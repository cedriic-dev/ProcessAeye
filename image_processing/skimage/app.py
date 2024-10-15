from skimage import io
from skimage.restoration import inpaint
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte

from utils.IInpaintingAlgorithmen import IInpaintingAlgorithmen


class Skimage(IInpaintingAlgorithmen):
    def __init__(self):
        super().__init__()

    def inpaint(self, image: str, mask: str) -> np.ndarray:

        mask = mask.convert('L').resize(image.size)

        image_np = np.array(image)
        mask_np = np.array(mask)

        mask_np = np.where(mask_np > 1, 1, 0).astype(np.uint8)

        inpainted_image = inpaint.inpaint_biharmonic(image_np, mask_np, channel_axis=-1)

        return img_as_ubyte(inpainted_image)

    def unload_model(self):
        pass

    def load_model(self):
        pass
