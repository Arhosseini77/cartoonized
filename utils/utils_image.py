import os

import cv2
import numpy as np


def resize_crop(image, target_size=720):
    h, w, _ = image.shape
    if min(h, w) > target_size:
        if h > w:
            new_w = target_size
            new_h = int(target_size * h / w)
        else:
            new_h = target_size
            new_w = int(target_size * w / h)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Crop dimensions to be multiples of 8
    h, w = image.shape[:2]
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image, save_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Expect image in RGB format
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_bgr)


def preprocess_image(image):
    # Resize and normalize to [-1,1]
    image = resize_crop(image)
    image = image.astype(np.float32) / 127.5 - 1.0
    return image
