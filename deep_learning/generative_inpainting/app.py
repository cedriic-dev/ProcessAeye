import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from deep_learning.generative_inpainting.model.networks import Generator
from deep_learning.generative_inpainting.utils.tools import get_config, random_bbox, mask_image, is_image_file, \
    default_loader, normalize, get_model_list

from utils.IInpaintingAlgorithmen import IInpaintingAlgorithmen


class GenerativeInpainting(IInpaintingAlgorithmen):

    def __init__(self):

        super().__init__(is_deep_learning=True)
        self.config = None
        self.cuda = None
        self.netG = None

        script_dir = os.path.dirname(os.path.realpath(__file__))

        config_path = os.path.join(script_dir, 'config', 'config.yaml')
        self.checkpoint_path = os.path.join(script_dir, 'model')

        self.load_config(config_path)
        self.check_cuda()

    def inpaint(self, image: str, mask: str) -> np.ndarray:

        if self.netG is None:
            self.load_model()

        start_time = time.time()

        # Load and preprocess the image and mask
        x = default_loader(image)

        width, height = x.size

        mask = default_loader(mask)
        mask = transforms.Resize([width, height])(mask)
        x = transforms.ToTensor()(x)
        mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
        x = normalize(x)
        x = x * (1. - mask)
        x = x.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)

        if self.cuda:
            x = x.cuda()
            mask = mask.cuda()

        # Perform Inpainting
        with torch.no_grad():
            _, x2, _ = self.netG(x, mask)
            inpainted_result = x2 * mask + x * (1. - mask)

        script_dir = os.path.dirname(os.path.realpath(__file__))

        output_path = os.path.join(script_dir, 'examples', 'output.png')

        # Convert the result to a NumPy array
        inpainted_result_denorm = inpainted_result * 0.5 + 0.5  # Adjust this line based on your original normalization
        inpainted_result_np = inpainted_result_denorm.squeeze().cpu().numpy().transpose(1, 2, 0)
        inpainted_result_np = (inpainted_result_np * 255.0).clip(0, 255).astype(np.uint8)
        del x, x2, mask, inpainted_result

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time elapsed: " + str(elapsed_time))
        return inpainted_result_np

    def unload_model(self):
        del self.netG
        self.netG = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()

    def load_model(self):
        self.netG = Generator(self.config['netG'], self.cuda, self.config['gpu_ids'])
        if self.checkpoint_path:
            last_model_name = get_model_list(self.checkpoint_path, "gen")
            self.netG.load_state_dict(torch.load(last_model_name, map_location=torch.device('cpu')))

        if self.cuda:
            self.netG = nn.parallel.DataParallel(self.netG, device_ids=self.config['gpu_ids'])

    def check_cuda(self):
        # Check if CUDA is available
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            device_ids = self.config['gpu_ids']
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
            device_ids = list(range(len(device_ids)))
            self.config['gpu_ids'] = device_ids
            torch.backends.cudnn.benchmark = True

    def load_config(self, path):
        # Load configuration
        self.config = get_config(path)
