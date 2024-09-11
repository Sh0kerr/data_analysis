from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

AVAILABLE_FLIP_AXIS = [0, 1]


class Matrix2D:
    def __init__(self, vals: np.ndarray) -> None:
        self.vals = vals

    def __repr__(self) -> str:
        return np.array2string(self.vals)

    def __matmul__(self, other: Matrix2D) -> Matrix2D:
        return Matrix2D(self.vals @ other.vals)
    
    @classmethod
    def from_list(cls, vals: list) -> Matrix2D:
        return cls(np.array(vals))

    def rotate(self, angle: float) -> Matrix2D:
        return Matrix2D(self.vals @ self._rotation_matrix(np.deg2rad(angle)))
    
    def flip(self, axis: Optional[int]=None) -> Matrix2D:
        if axis not in AVAILABLE_FLIP_AXIS:
            logging.error("Flip failed check axis value asshole")
            return self

        flip_ = np.eye(self.vals.shape[-1])
        if axis is not None:
            flip_[axis-1, axis-1] = -1
        else:
            flip_ *= -1

        return Matrix2D(self.vals @ flip_)

    def scale(self, scaler: float) -> Matrix2D:
        return Matrix2D(scaler * self.vals)
    
    def inv(self) -> Matrix2D:
        try:
            _inv = np.linalg.inv(self.vals)
        except np.linalg.LinAlgError:
            logging.error("Inv failed fix your matrix hello hello hello")
            _inv = self.vals
        return Matrix2D(_inv)

    @staticmethod
    def _rotation_matrix(angle_rad: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(angle_rad), np.sin(angle_rad)],
                [-np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )


def plot_2d_letter(letter: Matrix2D) -> None:
    plt.plot(letter.vals[:, 0], letter.vals[:, 1])
