"""CNN/MLP/LSTM backbones (paper Table I) and classifier/discriminator heads.

Every backbone outputs a 128-dim embedding; heads are a single ``Linear`` layer, so a classifier
(11-way) and discriminator (2-way) for a given backbone share every parameter except the final head
-- matching the plan's "SSFL-MLP/LSTM discriminators reuse their backbone + 2-way head".
"""

from __future__ import annotations

import torch
from torch import nn

from ssfl.config import Backbone
from ssfl.data.labels import LABEL_MAP, NUM_FEATURES

NUM_CLASSES = len(LABEL_MAP)
NUM_DISCRIMINATOR_CLASSES = 2
EMBEDDING_DIM = 128


class CNNBackbone(nn.Module):
    """Table I CNN. Input ``(batch, 23, 5)`` as ``Conv1d(in_channels=23, length=5)``.

    conv1-4: 64ch k3 s1 p1 (length stays 5). conv5-6: 128ch k3 s1 p1 (length stays 5).
    conv7-8: 128ch k3 s2 p1 (length 5->3->2). Flatten (128,2)=256 -> Dense128 -> ReLU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(23, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(128 * 2, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = x.flatten(start_dim=1)
        return self.relu(self.dense(x))


class MLPBackbone(nn.Module):
    """115 -> 512 -> 256 -> 128, ReLU between layers."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, EMBEDDING_DIM),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMBackbone(nn.Module):
    """2-layer LSTM, hidden=128, batch-first, over 5 timesteps x 23 features.

    Input is the Eq.19 reshape ``(batch, 23, 5)`` (rows=23, cols=5); transposed here to
    ``(batch, 5, 23)`` since the LSTM's "steps" are the 5 columns, each a 23-feature row group
    (the transpose of the CNN's channel-major reading of the same reshape).
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=23, hidden_size=EMBEDDING_DIM, num_layers=2, batch_first=True)
        self.dense = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, 23, 5) -> (batch, 5, 23)
        _, (h_n, _) = self.lstm(x)
        return self.relu(self.dense(h_n[-1]))


_BACKBONE_CLASSES: dict[Backbone, type[nn.Module]] = {
    Backbone.cnn: CNNBackbone,
    Backbone.mlp: MLPBackbone,
    Backbone.lstm: LSTMBackbone,
}


def uses_flat_features(backbone: Backbone) -> bool:
    """MLP consumes the flat 115-vector; CNN/LSTM consume the Eq.19 (23,5) reshape."""
    return backbone == Backbone.mlp


class SSFLModel(nn.Module):
    def __init__(self, backbone: nn.Module, out_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(EMBEDDING_DIM, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_classifier(backbone: Backbone) -> SSFLModel:
    return SSFLModel(_BACKBONE_CLASSES[backbone](), NUM_CLASSES)


def build_discriminator(backbone: Backbone) -> SSFLModel:
    return SSFLModel(_BACKBONE_CLASSES[backbone](), NUM_DISCRIMINATOR_CLASSES)


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def activation_shapes(model: nn.Module, example_input: torch.Tensor) -> dict[str, tuple[int, ...]]:
    """Leaf-module output shapes (batch dim excluded) for one forward pass -- the
    parameter-count/activation-shape report the M3 gate exercises against the CNN backbone."""
    shapes: dict[str, tuple[int, ...]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            shapes[name] = tuple(out.shape[1:])

        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(example_input)
    for h in hooks:
        h.remove()
    return shapes
