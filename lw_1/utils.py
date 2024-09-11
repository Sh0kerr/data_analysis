from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

AVAILABLE_FLIP_AXIS = [0, 1]


class Matrix:
    def __init__(self, vals: np.ndarray) -> None:
        self.vals = vals

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix(self.vals @ other.vals)
    
    @classmethod
    def from_list(cls, vals: list) -> Matrix:
        return cls(np.array(vals))

    def rotate2d(self, angle: float) -> Matrix:
        return Matrix(self.vals @ self._2d_rotation_matrix(np.deg2rad(angle)))
    
    def flip2d(self, axis: Optional[int]=None) -> Matrix:
        if axis not in AVAILABLE_FLIP_AXIS:
            logging.error("Flip failed check axis value asshole")
            return self

        flip_ = np.eye(self.vals.shape[-1])
        if axis is not None:
            flip_[axis-1, axis-1] = -1
        else:
            flip_ *= -1

        return Matrix(self.vals @ flip_)

    def scale(self, scaler: float) -> Matrix:
        return Matrix(scaler * self.vals)
    
    def inv(self) -> Matrix:
        try:
            _inv = np.linalg.inv(self.vals)
        except np.linalg.LinAlgError:
            logging.error("Inv failed fix your matrix hello hello hello")
            _inv = self.vals
        return Matrix(_inv)

    @staticmethod
    def _2d_rotation_matrix(angle_rad: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(angle_rad), np.sin(angle_rad)],
                [-np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )


def plot_2d_letter(letter: Matrix) -> None:
    plt.plot(letter.vals[:, 0], letter.vals[:, 1])
