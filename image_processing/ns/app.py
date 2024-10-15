import cv2
import numpy as np
from utils.IInpaintingAlgorithmen import IInpaintingAlgorithmen


class NSInpainter(IInpaintingAlgorithmen):
    def __init__(self):
        super().__init__()

    def inpaint(self, image: str, mask: str) -> np.ndarray:

        mask = mask.convert('L')

        mask = mask.resize(image.size)
        image_np = np.array(image)
        mask_np = np.array(mask).astype('uint8')

        inpainted_image_np = cv2.inpaint(image_np, mask_np, inpaintRadius=3, flags=cv2.INPAINT_NS)

        return inpainted_image_np

    def unload_model(self):
        pass

    def load_model(self):
        pass
