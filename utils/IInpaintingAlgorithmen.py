from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class IInpaintingAlgorithmen(ABC):

    def __init__(self, is_deep_learning=False):
        super().__init__()
        self.is_deep_learning = is_deep_learning

    @abstractmethod
    def inpaint(self, image, mask) -> np.ndarray:
        pass

    @abstractmethod
    def unload_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass
