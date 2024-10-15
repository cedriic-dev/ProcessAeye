import gc
import os

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from deep_learning.deepfillv2.utils.misc import infer_deepfill
from ..model import load_model

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def _load_models(config_path, device='cuda', ):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader)

    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_dir = os.path.join(project_dir)

    for name, cfg in config.items():
        cfg['path'] = os.path.join(model_dir, cfg['path'])

    config = {name: cfg for name, cfg in config.items()
              if os.path.exists(cfg['path'])}

    loaded_models = {}

    for name, cfg in config.items():
        is_loaded = False
        if cfg['load_at_startup']:
            model = load_model(cfg['path'], device)
            loaded_models[name] = model
            is_loaded = True
        config[name]['is_loaded'] = is_loaded

    return config, loaded_models


class Inpainter:
    def __init__(self, device=None):
        self.available_models = None
        self.loaded_models = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    def load_models(self, config_path):
        self.available_models, self.loaded_models = _load_models(config_path, self.device)

    def get_model_info(self):
        model_data = []
        for name, cfg in self.available_models.items():
            model_dict = cfg.copy()
            model_dict['name'] = name
            model_dict['type'] = 'df'
            model_data.append(model_dict)

        return model_data

    def check_requested_models(self, models):
        for name in models:
            if name not in self.loaded_models:
                path = self.available_models[name]['path']
                model = load_model(path, self.device)
                if model:
                    self.loaded_models[name] = model
                    self.available_models[name]['is_loaded'] = True

    def unload_models(self):
        self.loaded_models = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def inpaint(self, image, mask, models):
        req_models = models.split(',')
        self.check_requested_models(req_models)

        image_pil = Image.open(image).convert('RGB')
        width, height = image_pil.size

        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor()
        ])

        image = transform(image_pil)
        mask = transform(Image.open(mask))

        response_data = []
        project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        for idx_model, model_name in enumerate(req_models):
            return_vals = self.available_models[model_name]['return_vals']
            model_output_list = []
            outputs = infer_deepfill(
                self.loaded_models[model_name],
                image.to(self.device),
                mask.to(self.device),
                return_vals=return_vals
            )

            for idx_out, output in enumerate(outputs):
                # Convert to a NumPy array
                output_np = np.array(output)

                model_output_list.append({
                    'name': return_vals[idx_out],
                    'data': output_np
                })

            model_output_dict = {
                'name': model_name,
                'output': model_output_list
            }
            response_data.append(model_output_dict)

        return response_data
