import torch

from ssfl.config import Backbone
from ssfl.models import (
    NUM_CLASSES,
    NUM_DISCRIMINATOR_CLASSES,
    activation_shapes,
    build_classifier,
    build_discriminator,
    num_parameters,
    uses_flat_features,
)


def test_cnn_intermediate_shapes() -> None:
    from ssfl.models import CNNBackbone

    backbone = CNNBackbone()
    shapes = activation_shapes(backbone, torch.zeros(2, 23, 5))
    assert shapes["conv4"] == (64, 5)
    assert shapes["conv6"] == (128, 5)
    assert shapes["conv7"] == (128, 3)
    assert shapes["conv8"] == (128, 2)


def test_cnn_classifier_and_discriminator_output_dims() -> None:
    x = torch.zeros(4, 23, 5)
    classifier = build_classifier(Backbone.cnn)
    discriminator = build_discriminator(Backbone.cnn)
    assert classifier(x).shape == (4, NUM_CLASSES) == (4, 11)
    assert discriminator(x).shape == (4, NUM_DISCRIMINATOR_CLASSES) == (4, 2)


def test_mlp_classifier_and_discriminator_output_dims() -> None:
    x = torch.zeros(4, 115)
    classifier = build_classifier(Backbone.mlp)
    discriminator = build_discriminator(Backbone.mlp)
    assert classifier(x).shape == (4, 11)
    assert discriminator(x).shape == (4, 2)


def test_lstm_classifier_and_discriminator_output_dims() -> None:
    x = torch.zeros(4, 23, 5)
    classifier = build_classifier(Backbone.lstm)
    discriminator = build_discriminator(Backbone.lstm)
    assert classifier(x).shape == (4, 11)
    assert discriminator(x).shape == (4, 2)


def test_uses_flat_features_only_for_mlp() -> None:
    assert uses_flat_features(Backbone.mlp) is True
    assert uses_flat_features(Backbone.cnn) is False
    assert uses_flat_features(Backbone.lstm) is False


def test_backbones_share_parameters_between_classifier_and_discriminator_shape() -> None:
    # Same backbone class -> same parameter count modulo the head (11-way vs 2-way).
    for backbone in Backbone:
        clf = build_classifier(backbone)
        disc = build_discriminator(backbone)
        backbone_params = num_parameters(clf.backbone)
        assert num_parameters(clf) - backbone_params == 128 * 11 + 11
        assert num_parameters(disc) - backbone_params == 128 * 2 + 2
        assert num_parameters(clf.backbone) == num_parameters(disc.backbone)
