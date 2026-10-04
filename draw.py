import math
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


def circle(img, cx, cy, r, color):
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    if r < 0:
        return
    if r == 0:
        set_pixel(img, cx, cy, color)
        return

    x = 0
    y = r
    d = 1 - r

    def plot8(px, py):
        set_pixel(img, cx + px, cy + py, color)
        set_pixel(img, cx - px, cy + py, color)
        set_pixel(img, cx + px, cy - py, color)
        set_pixel(img, cx - px, cy - py, color)
        set_pixel(img, cx + py, cy + px, color)
        set_pixel(img, cx - py, cy + px, color)
        set_pixel(img, cx + py, cy - px, color)
        set_pixel(img, cx - py, cy - px, color)

    plot8(x, y)
    while x < y:
        x += 1
        if d < 0:
            d += 2 * x + 1
        else:
            y -= 1
            d += 2 * (x - y) + 1
        plot8(x, y)


def fill_circle(img, cx, cy, r, color):
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    if r < 0:
        return
    r2 = r * r
    for dy in range(-r, r + 1):
        y = cy + dy
        if 0 <= y < img.shape[0]:
            dx_max = int(math.isqrt(r2 - dy * dy))
            x_start = max(0, cx - dx_max)
            x_end = min(img.shape[1] - 1, cx + dx_max)
            for x in range(x_start, x_end + 1):
                img[y, x] = color


def fill_polygon(img, points, color):
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 3:
        return

    min_y = int(math.floor(min(p[1] for p in pts)))
    max_y = int(math.ceil(max(p[1] for p in pts)))

    for y in range(min_y, max_y):
        scan_y = y + 0.5
        x_intersects = []
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            if p1[1] == p2[1]:
                continue
            y_low, y_high = (p1[1], p2[1]) if p1[1] < p2[1] else (p2[1], p1[1])
            if y_low <= scan_y < y_high:
                t = (scan_y - p1[1]) / (p2[1] - p1[1])
                x_intersects.append(p1[0] + t * (p2[0] - p1[0]))

        x_intersects.sort()
        for k in range(0, len(x_intersects) - 1, 2):
            x_start = int(math.ceil(x_intersects[k] - 0.5))
            x_end = int(math.floor(x_intersects[k + 1] - 0.5))
            for x in range(x_start, x_end + 1):
                set_pixel(img, x, y, color)


def flood_fill(img, x, y, color, connectivity=4):
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return

    target_color = tuple(img[y, x])
    fill_color = tuple(color)
    if target_color == fill_color:
        return

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    stack = [(x, y)]
    visited = set([(x, y)])

    while stack:
        cx, cy = stack.pop()
        img[cy, cx] = fill_color

        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                if (nx, ny) not in visited:
                    if tuple(img[ny, nx]) == target_color:
                        visited.add((nx, ny))
                        stack.append((nx, ny))


def draw_polygon(img, points, color, thickness=1):
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        if thickness <= 1:
            line(img, p1, p2, color)
        else:
            thick_line(img, p1, p2, color, thickness)


def draw_marker(img, point, color, size=5):
    x, y = int(round(point[0])), int(round(point[1]))
    line(img, x - size, y, x + size, y, color)
    line(img, x, y - size, x, y + size, color)


def draw_grid(img, corners, n, color):
    p0, p1, p2, p3 = [np.array(p, dtype=float) for p in corners]
    for i in range(n + 1):
        t = i / float(n)
        top = (1.0 - t) * p0 + t * p1
        bottom = (1.0 - t) * p3 + t * p2
        line(img, int(round(top[0])), int(round(top[1])), int(round(bottom[0])), int(round(bottom[1])), color)

        left = (1.0 - t) * p0 + t * p3
        right = (1.0 - t) * p1 + t * p2
        line(img, int(round(left[0])), int(round(left[1])), int(round(right[0])), int(round(right[1])), color)


def put_number(img, point, k, color, scale=1):
    x, y = int(round(point[0])), int(round(point[1]))
    w = 8 * scale
    h = 14 * scale
    hh = 7 * scale

    p_tl = (x, y)
    p_tr = (x + w, y)
    p_ml = (x, y + hh)
    p_mr = (x + w, y + hh)
    p_bl = (x, y + h)
    p_br = (x + w, y + h)

    segs = {
        'a': (p_tl, p_tr),
        'b': (p_tr, p_mr),
        'c': (p_mr, p_br),
        'd': (p_bl, p_br),
        'e': (p_ml, p_bl),
        'f': (p_tl, p_ml),
        'g': (p_ml, p_mr),
    }

    digits = {
        0: ['a', 'b', 'c', 'd', 'e', 'f'],
        1: ['b', 'c'],
        2: ['a', 'b', 'g', 'e', 'd'],
        3: ['a', 'b', 'g', 'c', 'd'],
        4: ['f', 'g', 'b', 'c'],
        5: ['a', 'f', 'g', 'c', 'd'],
        6: ['a', 'f', 'e', 'd', 'c', 'g'],
        7: ['a', 'b', 'c'],
        8: ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        9: ['a', 'b', 'c', 'd', 'f', 'g'],
    }

    for s in digits.get(int(k), []):
        p1, p2 = segs[s]
        line(img, p1, p2, color)


def dashed_line(img, *args, dash_len=8, gap_len=6):
    if len(args) == 3:
        (x0, y0), (x1, y1), color = args
    else:
        x0, y0, x1, y1, color = args

    pts = _bresenham_points(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    period = dash_len + gap_len
    for i, (x, y) in enumerate(pts):
        if (i % period) < dash_len:
            set_pixel(img, x, y, color)
