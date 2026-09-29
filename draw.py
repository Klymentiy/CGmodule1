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


def _bresenham_points(x0, y0, x1, y1):
    pts = []
    if (x0, y0) > (x1, y1):
        x0, y0, x1, y1 = x1, y1, x0, y0

    dx = x1 - x0
    dy = abs(y1 - y0)
    sy = 1 if y1 >= y0 else -1

    if dy <= dx:
        d = 2 * dy - dx
        y = y0
        for x in range(x0, x1 + 1):
            pts.append((x, y))
            if d > 0:
                y += sy
                d += 2 * (dy - dx)
            else:
                d += 2 * dy
    else:
        d = 2 * dx - dy
        x = x0
        for y in range(y0, y1 + sy, sy):
            pts.append((x, y))
            if d > 0:
                x += 1
                d += 2 * (dx - dy)
            else:
                d += 2 * dx
    return pts


def line(img, *args):
    if len(args) == 3:
        (x0, y0), (x1, y1), color = args
    else:
        x0, y0, x1, y1, color = args

    for x, y in _bresenham_points(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))):
        set_pixel(img, x, y, color)


def thick_line(img, *args):
    if len(args) == 4:
        (x0, y0), (x1, y1), color, width = args
    else:
        x0, y0, x1, y1, color, width = args

    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    if width <= 1:
        line(img, x0, y0, x1, y1, color)
        return

    r = width // 2
    r2 = (width / 2.0) ** 2
    for px, py in _bresenham_points(x0, y0, x1, y1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r2:
                    set_pixel(img, px + dx, py + dy, color)
