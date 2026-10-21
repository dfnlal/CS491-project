"""Coordinate conventions and small vector helpers.

Image space: pixel coordinates, x to the right, y down (OpenCV / Qt).
World space: X to the right, Y up, Z toward the viewer.

The first version uses an orthographic camera: the world point (X, Y, Z)
lands on image pixel (X, -Y), so depth does not change the projection.
The paper uses a perspective camera (section 5 and appendix A); swapping
it in later only means replacing `project` and the depth steps in sweep.py.
"""

import numpy as np


def to_world2(p):
    """Image pixel coordinates -> world XY (flips the y axis)."""
    out = np.array(p, dtype=float, copy=True)
    out[..., 1] = -out[..., 1]
    return out


# The flip is its own inverse.
to_image = to_world2


def project(points3):
    """Orthographic projection of world points to image pixels."""
    p = np.asarray(points3, dtype=float)
    return to_image(p[..., :2])


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def perp(d):
    """2D vector rotated by +90 degrees."""
    d = np.asarray(d, dtype=float)
    return np.stack([-d[..., 1], d[..., 0]], axis=-1)


def rotation_between(a, b):
    """Smallest rotation matrix that turns direction a into direction b."""
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = normalize(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def polyline_length(pts):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def resample_polyline(pts, spacing):
    """Points spread evenly (about `spacing` apart) along a polyline."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], seg > 1e-9])]
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def tangents(pts, window=2):
    """Unit tangent at each point, from a centered difference."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    idx = np.arange(n)
    lo = np.clip(idx - window, 0, n - 1)
    hi = np.clip(idx + window, 0, n - 1)
    return normalize(pts[hi] - pts[lo])


def euler_matrix(rx, ry, rz):
    """Rotation matrix from angles in degrees, applied as X, then Y, then Z."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx
