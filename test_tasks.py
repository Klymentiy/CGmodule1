import math
import os
import cv2
import numpy as np
import pytest

from draw import (
    canvas, set_pixel, save, line, thick_line, _bresenham_points,
    circle, fill_circle, fill_polygon, flood_fill,
    draw_polygon, draw_marker, draw_grid, put_number, dashed_line
)
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


def test_task7_circle():
    for r in range(1, 101):
        cx, cy = 105, 105
        img = canvas(210, 210, (0, 0, 0))
        circle(img, cx, cy, r, (255, 255, 255))
        pts = list(zip(*np.where((img == [255, 255, 255]).all(axis=2))[::-1]))
        assert len(pts) > 0
        for px, py in pts:
            dist = math.sqrt((px - cx)**2 + (py - cy)**2)
            assert abs(dist - r) < 0.5


def test_task8_polygon_and_circle_fill():
    img = canvas(50, 50, (0, 0, 0))
    rect = [(10, 10), (30, 10), (30, 20), (10, 20)]
    fill_polygon(img, rect, (255, 255, 255))
    count = np.count_nonzero((img == [255, 255, 255]).all(axis=2))
    assert count == 200

    img2 = canvas(50, 50, (0, 0, 0))
    fill_circle(img2, 25, 25, 10, (255, 255, 255))
    count2 = np.count_nonzero((img2 == [255, 255, 255]).all(axis=2))
    assert 300 < count2 < 325


def test_task9_flood_fill():
    img4 = canvas(100, 100, (255, 255, 255))
    line(img4, 0, 0, 99, 99, (0, 0, 0))
    flood_fill(img4, 90, 10, (255, 0, 0), connectivity=4)
    count4 = np.count_nonzero((img4 == [255, 0, 0]).all(axis=2))
    assert count4 == 4950

    img8 = canvas(100, 100, (255, 255, 255))
    line(img8, 0, 0, 99, 99, (0, 0, 0))
    flood_fill(img8, 90, 10, (255, 0, 0), connectivity=8)
    count8 = np.count_nonzero((img8 == [255, 0, 0]).all(axis=2))
    assert count8 == 9900


def test_task10_tools():
    img = canvas(200, 200, (255, 255, 255))
    corners = [(20, 20), (180, 20), (160, 180), (30, 160)]
    draw_polygon(img, corners, (0, 0, 0), thickness=2)
    draw_grid(img, corners, 3, (100, 100, 100))
    for i, pt in enumerate(corners):
        draw_marker(img, pt, (255, 0, 0), size=4)
        put_number(img, pt, i, (0, 0, 255))

    count = np.count_nonzero((img != [255, 255, 255]).any(axis=2))
    assert count > 200


def test_task11_dashed_line():
    img = canvas(100, 100, (0, 0, 0))
    dashed_line(img, 10, 10, 90, 90, (255, 255, 255), dash_len=5, gap_len=5)
    dashed_count = np.count_nonzero((img == [255, 255, 255]).all(axis=2))

    img_solid = canvas(100, 100, (0, 0, 0))
    line(img_solid, 10, 10, 90, 90, (255, 255, 255))
    solid_count = np.count_nonzero((img_solid == [255, 255, 255]).all(axis=2))

    assert 0 < dashed_count < solid_count
