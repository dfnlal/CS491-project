"""Synthetic test images with known 3D shapes.

Used by the tests and by tools/make_samples.py. Each function returns the
image and the strokes a user would draw on it, so results can be checked
against the true size.
"""

import cv2
import numpy as np

from .geometry import euler_matrix

BACKGROUND = (235, 235, 235)


def cylinder(size=(400, 420), x0=150, x1=250, top=100, bottom=330, b=15):
    """Upright can seen slightly from above. True radius = (x1 - x0) / 2."""
    w, h = size
    img = np.full((h, w, 3), BACKGROUND, np.uint8)
    cx, a = (x0 + x1) // 2, (x1 - x0) // 2
    cv2.ellipse(img, (cx, bottom), (a, b), 0, 0, 360, (150, 90, 60), -1, cv2.LINE_AA)
    for x in range(x0, x1 + 1):
        shade = 0.6 + 0.4 * np.cos((x - cx) / a * 1.2)
        img[top:bottom, x] = (int(170 * shade), int(110 * shade), int(70 * shade))
    cv2.ellipse(img, (cx, top), (a, b), 0, 0, 360, (210, 160, 120), -1, cv2.LINE_AA)
    strokes = {
        "kind": "circle",
        "s1": (x0, top),
        "s2": (x1, top),
        "s3": (cx, top + b),
        "axis": [(cx, top), (cx, (top + bottom) / 2), (cx, bottom - 10)],
    }
    return img, strokes, {"radius": a, "length": bottom - 10 - top}


def cone(size=(400, 420), top=100, bottom=330, r_top=60, r_bottom=30):
    """Tapered can: radius shrinks linearly from top to bottom."""
    w, h = size
    img = np.full((h, w, 3), BACKGROUND, np.uint8)
    cx = w // 2
    poly = np.array([[cx - r_top, top], [cx + r_top, top], [cx + r_bottom, bottom], [cx - r_bottom, bottom]])
    cv2.fillPoly(img, [poly.astype(np.int32)], (60, 120, 40), cv2.LINE_AA)
    cv2.ellipse(img, (cx, bottom), (r_bottom, 8), 0, 0, 360, (60, 120, 40), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, top), (r_top, 14), 0, 0, 360, (120, 200, 90), -1, cv2.LINE_AA)
    strokes = {
        "kind": "circle",
        "s1": (cx - r_top, top),
        "s2": (cx + r_top, top),
        "s3": (cx, top + 14),
        "axis": [(cx, top), (cx, bottom - 10)],
    }
    return img, strokes, {"r_top": r_top, "r_bottom": r_bottom, "top": top, "bottom": bottom}


def bent_pipe(size=(500, 420), center=(250, 330), radius=150, thickness=50, start=200, end=340):
    """Tube bent along a circular arc in the image plane. True radius = thickness / 2."""
    w, h = size
    img = np.full((h, w, 3), BACKGROUND, np.uint8)
    cv2.ellipse(img, center, (radius, radius), 0, start, end, (40, 60, 160), thickness, cv2.LINE_8)
    cx, cy = center
    r_in, r_out = radius - thickness / 2, radius + thickness / 2

    def at(r, deg):
        t = np.radians(deg)
        return (cx + r * np.cos(t), cy + r * np.sin(t))

    s = start + 2
    tangent = np.array([-np.sin(np.radians(s)), np.cos(np.radians(s))])
    mid = np.array(at(radius, s))
    strokes = {
        "kind": "circle",
        "s1": at(r_in, s),
        "s2": at(r_out, s),
        "s3": tuple(mid + 6 * tangent),
        "axis": [at(radius, t) for t in np.linspace(s, end - 3, 40)],
    }
    return img, strokes, {"radius": thickness / 2}


def box(size=(420, 440), dims=(120, 150, 80), yaw=35.0, pitch=25.0, center=(210, 230)):
    """Box (width, height, depth) seen from above at an angle.

    The strokes follow two edges of the visible top face, then sweep down
    through the box's center line.
    """
    w, h = size
    W, H, D = dims
    img = np.full((h, w, 3), BACKGROUND, np.uint8)
    rot = euler_matrix(pitch, yaw, 0.0)
    cx, cy = center

    def to_img(p):
        q = rot @ np.asarray(p, float)
        return np.array([cx + q[0], cy - q[1]]), q

    signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    corner = {s: to_img((s[0] * W / 2, s[1] * H / 2, s[2] * D / 2))[0] for s in signs}
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            normal = np.zeros(3)
            normal[axis] = sign
            n_world = rot @ normal
            if n_world[2] <= 0:
                continue
            quad = [s for s in signs if s[axis] == sign]
            pts = np.array([corner[s] for s in quad])
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
            shade = 0.45 + 0.55 * n_world[2]
            faces.append((n_world[2], hull, shade))
    for _, hull, shade in sorted(faces, key=lambda f: f[0]):
        color = (int(80 * shade), int(160 * shade), int(220 * shade))
        cv2.fillPoly(img, [np.round(hull).astype(np.int32)], color, cv2.LINE_AA)

    t0 = corner[(-1, 1, 1)]  # top face, front-left
    t1 = corner[(1, 1, 1)]  # top face, front-right
    t2 = corner[(1, 1, -1)]  # top face, back-right
    top_center = to_img((0, H / 2, 0))[0]
    bottom_center = to_img((0, -H / 2, 0))[0]
    strokes = {
        "kind": "rect",
        "s1": tuple(t0),
        "s2": tuple(t1),
        "s3": tuple(t2),
        "axis": [tuple(top_center), tuple(bottom_center)],
    }
    return img, strokes, {"width": W, "height": H, "depth": D}
