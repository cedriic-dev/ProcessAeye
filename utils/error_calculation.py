import numpy as np
from PIL import Image
import torchvision.transforms as T


def calculate_frobenius_error(ground_truth, inpainted, mask, is_deep_learning):
    ground_truth, mask = convert_to_numpy(ground_truth, mask, is_deep_learning)

    masked_ground_truth = ground_truth * mask
    masked_inpainted = inpainted * mask

    # Berechne die Differenz zwischen den maskierten Bereichen
    masked_ground_truth = masked_ground_truth.astype(np.int16)
    masked_inpainted = masked_inpainted.astype(np.int16)

    error_image = np.abs(masked_ground_truth - masked_inpainted)

    error_image = error_image.astype(np.uint8)

    # Berechne die Frobenius-Norm des Fehlerbildes
    frobenius_error = sum(
        np.linalg.norm(error_image[:, :, channel], ord='fro') for channel in range(error_image.shape[2]))

    normalized_error = frobenius_error / np.sum(mask)

    return normalized_error


def convert_to_numpy(ground_truth, mask, is_deep_learning):
    if is_deep_learning:
        ground_truth = Image.open(ground_truth)
        mask = Image.open(mask)

    mask = mask.convert('L')
    width, height = ground_truth.size

    transform = T.Compose([
        T.Resize((height, width)),  # Resize to match the image size
    ])

    if (width, height) != mask.size:
        mask = transform(mask)

    image_np = np.array(ground_truth)
    mask_np = np.array(mask)
    mask_np = (mask_np > 0).astype(np.uint8)

    if len(image_np.shape) == 3 and len(mask_np.shape) < 3:
        mask_np = np.stack([mask_np] * 3, axis=-1)

    return image_np, mask_np
