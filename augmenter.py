import numpy as np
import cv2
import os

def augment_image(image_path, save_dir):
    image = cv2.imread(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]

    operations = {
        'flip_h': cv2.flip(image, 1),
        'flip_v': cv2.flip(image, 0),
        'rotate_45': cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        'blur': cv2.GaussianBlur(image, (5, 5), 0),
    }

    os.makedirs(save_dir, exist_ok=True)
    for op_name, img in operations.items():
        save_path = os.path.join(save_dir, f"{base}_{op_name}.png")
        cv2.imwrite(save_path, img)