import gc
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.nn as nn
from utils.error_calculation import calculate_frobenius_error

from deep_learning.misf.utils.config import Config
from deep_learning.misf.utils.models import InpaintingModel
from utils.IInpaintingAlgorithmen import IInpaintingAlgorithmen


class Misf(IInpaintingAlgorithmen):

    def __init__(self, device=None):
        super().__init__(is_deep_learning=True)
        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.inpaint_model = None
        self.config = Config(os.path.join(self.script_path, 'model', 'config.yml'))
        self.model_path = os.path.join(self.script_path, 'model', 'celebA_InpaintingModel_gen.pth')
        self.device = torch.device('cuda'
                                   if torch.cuda.is_available() else 'cpu') \
            if device is None else device

        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def inpaint(self, image, mask) -> np.ndarray:
        self.check_model()

        img_pil = Image.open(image)
        mask_pil = Image.open(mask)
        width, height = img_pil.size

        transform = T.Compose([
            T.Resize((height, width)),  # Resize to match the image size
            T.ToTensor()
        ])

        img_tensor = transform(img_pil)
        mask_tensor = transform(mask_pil)

        img_tensor = img_tensor[:3].unsqueeze(0)
        mask_tensor = mask_tensor[0:1].unsqueeze(0)

        if torch.cuda.is_available():
            img_tensor = img_tensor.half().cuda()
            mask_tensor = mask_tensor.half().cuda()

        with torch.no_grad():
            img_masked = img_tensor * (1 - mask_tensor)
            input = torch.cat((img_masked, mask_tensor), dim=1)
            output_tensor = self.inpaint_model.generator(input)

        # Convert tensors to NumPy arrays
        output_np = (output_tensor[0] * 255.0).clamp(0, 255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        return output_np

    def unload_model(self):
        del self.inpaint_model
        self.inpaint_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()

    def load_model(self):
        self.inpaint_model = InpaintingModel(self.config)
        data = torch.load(self.model_path, map_location=self.device)
        self.inpaint_model.generator.load_state_dict(data['generator'])

        # Move the model to the GPU
        if torch.cuda.is_available():
            self.inpaint_model = self.inpaint_model.half().cuda()

        # Wrap the model with DataParallel if more than one GPU is available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.inpaint_model = nn.DataParallel(self.inpaint_model)

        print('the model is loaded')

    @staticmethod
    def to_img(data):
        data = (data * 255.0).clamp(0, 255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        return Image.fromarray(data)

    def check_model(self):
        if self.inpaint_model is None:
            self.load_model()
