import numpy as np
from dataclasses import dataclass
from enum import Enum, IntEnum, unique, auto


@unique
class CameraStream(Enum):
    LEFT = auto()
    RIGHT = auto()
    COLOR = auto()
    DEPTH = auto()


@unique
class CameraIndex(IntEnum):
    LEFT = 1
    RIGHT = 2
    COLOR = -1


@unique
class LaserMode(Enum):
    ON = auto()
    OFF = auto()


@dataclass
class StereoPair:
    left: np.ndarray
    right: np.ndarray

    def __iter__(self):
        return iter((self.left, self.right))


@unique
class CaptureMode(Enum):
    UNRECTIFIED = auto()
    RECTIFIED = auto()


class Size:
    """Represents image size"""

    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def __str__(self):
        return f"w: {self._width}, h: {self._height}"

    @classmethod
    def from_numpy_shape(cls, shape):
        return cls(*shape[1::-1])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def as_cv_size(self):
        return np.array((self._width, self._height), dtype=int)

    @property
    def dict(self):
        return {"width": self._width, "height": self._height}
