import numpy as np

GAMMA = 2.2


def to_linear(v):
    v_arr = np.asanyarray(v, dtype=np.float64) / 255.0
    v_arr = np.clip(v_arr, 0.0, 1.0)
    linear = v_arr ** GAMMA
    return float(linear) if np.isscalar(v) else linear


def to_srgb(l):
    l_arr = np.asanyarray(l, dtype=np.float64)
    l_arr = np.clip(l_arr, 0.0, 1.0)
    val = 255.0 * (l_arr ** (1.0 / GAMMA))
    rounded = np.rint(val).astype(int)
    clamped = np.clip(rounded, 0, 255)
    return int(clamped) if np.isscalar(l) else clamped


def blend_colors(c1, c2, t=0.5):
    l1 = to_linear(c1)
    l2 = to_linear(c2)
    return to_srgb((1.0 - t) * l1 + t * l2)


def rgb_to_hsv(r, g, b):
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax = max(rf, gf, bf)
    cmin = min(rf, gf, bf)
    delta = cmax - cmin

    v = cmax
    s = 0.0 if cmax == 0.0 else delta / cmax

    if delta == 0.0:
        h = 0.0
    elif cmax == rf:
        h = 60.0 * (((gf - bf) / delta) % 6.0)
    elif cmax == gf:
        h = 60.0 * (((bf - rf) / delta) + 2.0)
    else:
        h = 60.0 * (((rf - gf) / delta) + 4.0)

    h = h % 360.0
    if h < 0.0:
        h += 360.0

    return float(h), float(s), float(v)


def hsv_to_rgb(h, s, v):
    h_norm = (h % 360.0) / 60.0
    s_clamped = max(0.0, min(1.0, float(s)))
    v_clamped = max(0.0, min(1.0, float(v)))

    c = v_clamped * s_clamped
    x = c * (1.0 - abs((h_norm % 2.0) - 1.0))
    m = v_clamped - c

    if 0.0 <= h_norm < 1.0:
        rp, gp, bp = c, x, 0.0
    elif 1.0 <= h_norm < 2.0:
        rp, gp, bp = x, c, 0.0
    elif 2.0 <= h_norm < 3.0:
        rp, gp, bp = 0.0, c, x
    elif 3.0 <= h_norm < 4.0:
        rp, gp, bp = 0.0, x, c
    elif 4.0 <= h_norm < 5.0:
        rp, gp, bp = x, 0.0, c
    else:
        rp, gp, bp = c, 0.0, x

    r = int(round((rp + m) * 255.0))
    g = int(round((gp + m) * 255.0))
    b = int(round((bp + m) * 255.0))

    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
