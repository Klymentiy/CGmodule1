import os
import cv2
import numpy as np
import pytest

from draw import canvas, set_pixel, save
from color import to_linear, to_srgb, blend_colors, rgb_to_hsv, hsv_to_rgb


def test_task1_canvas_and_pixel():
    w, h = 700, 540
    img = canvas(w, h, color=(0, 0, 0))
    assert img.shape == (540, 700, 3)
    assert img.dtype == np.uint8

    set_pixel(img, 0, 0, (255, 0, 0))
    set_pixel(img, -5, 3, (0, 255, 0))
    set_pixel(img, 700, 540, (0, 0, 255))
    set_pixel(img, 1000, -100, (128, 128, 128))

    assert tuple(img[0, 0]) == (255, 0, 0)
    red_count = np.count_nonzero((img == [255, 0, 0]).all(axis=2))
    assert red_count == 1

    test_out = "out/test_task1.png"
    save(img, test_out)
    assert os.path.exists(test_out)
    loaded = cv2.imdecode(np.fromfile(test_out, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert tuple(loaded[0, 0]) == (0, 0, 255)
    os.remove(test_out)


def test_task2_gamma():
    mixed = blend_colors(0, 255, 0.5)
    assert mixed == 186

    l_sum = to_linear(0) + to_linear(0) + to_linear(0) + to_linear(255)
    res = to_srgb(l_sum / 4.0)
    assert res == 136


def test_task3_hsv():
    h, s, v = rgb_to_hsv(255, 128, 0)
    assert abs(h - 30.1176) < 0.1
    assert abs(s - 1.0) < 1e-6
    assert abs(v - 1.0) < 1e-6

    h_gray, s_gray, v_gray = rgb_to_hsv(128, 128, 128)
    assert s_gray == 0.0
    assert abs(v_gray - 128 / 255.0) < 1e-4

    rng = np.random.default_rng(42)
    random_colors = rng.integers(0, 256, size=(10000, 3))

    for r, g, b in random_colors:
        h, s, v = rgb_to_hsv(int(r), int(g), int(b))
        r2, g2, b2 = hsv_to_rgb(h, s, v)
        assert abs(r - r2) <= 1
        assert abs(g - g2) <= 1
        assert abs(b - b2) <= 1
