import numpy as np
from draw import canvas, set_pixel, save
from color import to_linear, to_srgb, blend_colors, rgb_to_hsv, hsv_to_rgb


def task1():
    w, h = 700, 540
    img = canvas(w, h, color=(0, 0, 0))
    set_pixel(img, 0, 0, (255, 0, 0))
    set_pixel(img, -5, 3, (0, 255, 0))
    set_pixel(img, 1000, 1000, (0, 0, 255))
    save(img, "out/01-полотно.png")


def task2():
    print("0 + 255 blend:", blend_colors(0, 255))
    avg_linear = (to_linear(0) * 3 + to_linear(255)) / 4.0
    print("0, 0, 0, 255 avg:", to_srgb(avg_linear))

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
    h, s, v = rgb_to_hsv(255, 128, 0)
    print("RGB(255, 128, 0) -> HSV:", round(h, 2), s, v)

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


if __name__ == "__main__":
    task1()
    task2()
    task3()
