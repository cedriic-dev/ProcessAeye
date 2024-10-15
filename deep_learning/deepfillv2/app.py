import os

import numpy as np

from deep_learning.deepfillv2.utils.inpainter import Inpainter
from utils.IInpaintingAlgorithmen import IInpaintingAlgorithmen
from PIL import Image


class DeepFillV2(IInpaintingAlgorithmen):

    def __init__(self):
        super().__init__(is_deep_learning=True)
        self.inpainter = Inpainter()
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(script_dir, 'model', 'models.yaml')

    def inpaint(self, image: Image, mask: Image) -> np.ndarray:

        self.check_loaded_models()

        model_names = list(self.inpainter.loaded_models.keys())
        model_names_str = ','.join(model_names)
        response_data = self.inpainter.inpaint(image, mask, model_names_str)
        return response_data[0]['output'][1]['data']

    def unload_model(self):
        self.inpainter.unload_models()

    def load_model(self):
        self.inpainter.load_models(self.model_path)
        print("model loaded")

    def check_loaded_models(self):
        if self.inpainter.loaded_models is None:
            self.load_model()
