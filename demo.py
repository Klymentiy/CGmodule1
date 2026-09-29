import math
import cv2
import numpy as np
from draw import canvas, set_pixel, save, line, thick_line
from color import to_linear, to_srgb, blend_colors, rgb_to_hsv, hsv_to_rgb


def task1():
    w, h = 700, 540
    img = canvas(w, h, color=(0, 0, 0))
    set_pixel(img, 0, 0, (255, 0, 0))
    set_pixel(img, -5, 3, (0, 255, 0))
    set_pixel(img, 1000, 1000, (0, 0, 255))
    save(img, "out/01-полотно.png")


def task2():
    w, h = 512, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    half_h = h // 2

    xs = np.linspace(0.0, 1.0, w)
    top_vals = np.rint(xs * 255.0).astype(np.uint8)
    bottom_vals = to_srgb(xs)

    for c in range(3):
        img[:half_h, :, c] = top_vals
        img[half_h:, :, c] = bottom_vals

    save(img, "out/02-смуги.png")


def task3():
    size = 400
    radius = 180
    cx, cy = size // 2, size // 2
    wheel_img = np.full((size, size, 3), 255, dtype=np.uint8)

    y_indices, x_indices = np.indices((size, size))
    dx = x_indices - cx
    dy = y_indices - cy
    dist = np.sqrt(dx**2 + dy**2)
    mask = dist <= radius

    angles = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
    saturations = dist / radius

    for y, x in zip(y_indices[mask], x_indices[mask]):
        wheel_img[y, x] = hsv_to_rgb(float(angles[y, x]), float(saturations[y, x]), 1.0)

    save(wheel_img, "out/03-коло-hsv.png")


def task5():
    w, h = 700, 540
    img = canvas(w, h, color=(255, 255, 255))
    cx, cy = w // 2, h // 2
    n_rays = 27
    r = 230
    ray_color = (30, 120, 220)

    for i in range(n_rays):
        angle = 2.0 * math.pi * i / n_rays
        x1 = int(round(cx + r * math.cos(angle)))
        y1 = int(round(cy + r * math.sin(angle)))
        line(img, cx, cy, x1, y1, ray_color)

    save(img, "out/05-зірка.png")


def task6():
    w, h = 700, 540
    img = canvas(w, h, color=(250, 250, 250))

    thick_line(img, 50, 60, 300, 60, (50, 50, 50), 1)
    thick_line(img, 50, 100, 300, 100, (50, 50, 50), 3)
    thick_line(img, 50, 150, 300, 150, (50, 50, 50), 5)
    thick_line(img, 50, 210, 300, 210, (50, 50, 50), 8)

    line(img, 50, 280, 300, 480, (0, 0, 255))
    cv2_img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.line(cv2_img, (60, 280), (310, 480), (255, 0, 0), 1)
    img[cv2_img > 0] = cv2_img[cv2_img > 0]

    block = 20
    ox, oy = 370, 140
    test_line = [(2, 2), (13, 5)]

    cv_test = np.zeros((8, 16), dtype=np.uint8)
    cv2.line(cv_test, test_line[0], test_line[1], 255, 1)

    our_test = canvas(16, 8, (0, 0, 0))
    line(our_test, test_line[0][0], test_line[0][1], test_line[1][0], test_line[1][1], (255, 255, 255))

    for py in range(8):
        for px in range(16):
            rx, ry = ox + px * block, oy + py * block
            c_val = (240, 240, 240)
            in_our = our_test[py, px, 0] > 0
            in_cv = cv_test[py, px] > 0

            if in_our and in_cv:
                c_val = (30, 100, 230)
            elif in_our:
                c_val = (220, 20, 60)
            elif in_cv:
                c_val = (40, 180, 50)

            img[ry:ry + block - 1, rx:rx + block - 1] = c_val

    save(img, "out/06-порівняння.png")


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task5()
    task6()
