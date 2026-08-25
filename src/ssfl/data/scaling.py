"""Min-max feature scaling and the Equation 19 (115 -> 23x5) reshape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MinMaxScaler:
    min_: np.ndarray  # (num_features,) float32
    max_: np.ndarray  # (num_features,) float32
    constant_mask: np.ndarray  # (num_features,) bool -- True where max == min

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float32)
        non_constant = ~self.constant_mask
        scale = (self.max_ - self.min_)[non_constant]
        out[:, non_constant] = (x[:, non_constant] - self.min_[non_constant]) / scale
        return out  # constant features stay 0.0 (already zero-initialized)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"min": self.min_, "max": self.max_, "constant_mask": self.constant_mask}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "MinMaxScaler":
        return cls(
            min_=arrays["min"].astype(np.float32),
            max_=arrays["max"].astype(np.float32),
            constant_mask=arrays["constant_mask"].astype(bool),
        )


def fit_scaler(matrix: np.ndarray) -> MinMaxScaler:
    min_ = matrix.min(axis=0).astype(np.float32)
    max_ = matrix.max(axis=0).astype(np.float32)
    constant_mask = (max_ - min_) == 0
    return MinMaxScaler(min_=min_, max_=max_, constant_mask=constant_mask)


def reshape_eq19(x: np.ndarray, rows: int = 23, cols: int = 5) -> np.ndarray:
    """Equation 19: flat feature vector -> (rows, cols) matrix with ``M[r, c] = v[r + rows*c]``.

    Equivalent to ``v.reshape((rows, cols), order='F')`` for a single vector; implemented via a
    C-order reshape-then-transpose so it vectorizes correctly over any number of leading batch
    dimensions (``order='F'`` reshape does not compose with a leading batch axis).
    """
    if x.shape[-1] != rows * cols:
        raise ValueError(f"expected last dimension {rows * cols}, got {x.shape[-1]}")
    leading = x.shape[:-1]
    reshaped = x.reshape(*leading, cols, rows)
    axes = list(range(reshaped.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return reshaped.transpose(axes)
