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
