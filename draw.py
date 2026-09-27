import os
import cv2
import numpy as np


def canvas(w, h, color=(0, 0, 0)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def set_pixel(img, x, y, color):
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        img[y, x] = color


def save(img, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    success, buf = cv2.imencode(ext, img[:, :, ::-1])
    if success:
        with open(path, "wb") as f:
            f.write(buf)
