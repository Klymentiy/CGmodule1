import os
import cv2
import numpy as np
import pytest

from draw import canvas, set_pixel, save, line, thick_line, _bresenham_points
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


def test_task4_bresenham_lecture():
    expected = [
        (2, 2), (3, 2), (4, 3), (5, 3), (6, 3), (7, 3),
        (8, 4), (9, 4), (10, 4), (11, 4), (12, 5), (13, 5)
    ]
    actual = _bresenham_points(2, 2, 13, 5)
    assert actual == expected

    img = canvas(20, 20, (0, 0, 0))
    line(img, 2, 2, 13, 5, (255, 255, 255))
    pts = list(zip(*np.where((img == [255, 255, 255]).all(axis=2))[::-1]))
    assert set(pts) == set(expected)


def test_task5_symmetry_and_octants():
    rng = np.random.default_rng(42)
    for _ in range(1000):
        p0 = tuple(rng.integers(-50, 50, size=2))
        p1 = tuple(rng.integers(-50, 50, size=2))
        pts1 = set(_bresenham_points(p0[0], p0[1], p1[0], p1[1]))
        pts2 = set(_bresenham_points(p1[0], p1[1], p0[0], p0[1]))
        assert pts1 == pts2

    assert _bresenham_points(5, 5, 5, 5) == [(5, 5)]
    assert _bresenham_points(2, 4, 8, 4) == [(x, 4) for x in range(2, 9)]
    assert _bresenham_points(4, 2, 4, 8) == [(4, y) for y in range(2, 9)]


def test_task6_thick_line_and_cv2():
    img = canvas(100, 100, (0, 0, 0))
    thick_line(img, 10, 10, 80, 80, (255, 255, 255), 5)
    count = np.count_nonzero((img == [255, 255, 255]).all(axis=2))
    assert count > 71

    our_pts = set(_bresenham_points(2, 2, 13, 5))
    cv_img = np.zeros((20, 20), dtype=np.uint8)
    cv2.line(cv_img, (2, 2), (13, 5), 255, 1)
    cv_pts = set(zip(*np.where(cv_img == 255)[::-1]))
    assert our_pts == cv_pts
