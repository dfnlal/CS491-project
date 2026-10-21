"""3D profiles from the first two sweeps (paper section 4, "Profile").

Circle (generalized cylinder): S1-S2 is the major diameter of the ellipse
the circle projects to, and S3 marks the end of the minor diameter.
Rectangle (generalized cuboid): S1-S2 and S2-S3 follow two edges of the
cuboid's base.

The axis (sweep) direction picks which way the profile faces. The depth
sign is still ambiguous (the drawn face can point toward the camera or
away from it); by default the drawn face points toward the camera and
the part extends away from it. `flip` picks the other solution.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import normalize, perp

# Smallest allowed |axis_xy|. An axis pointing straight at the camera has
# no 2D direction to sweep along, so the tilt is kept away from that case.
MIN_TILT = 0.2


@dataclass
class Profile:
    kind: str  # "circle" or "rect"
    center: np.ndarray  # (3,) world position of the profile center
    axis: np.ndarray  # (3,) unit normal of the profile = sweep direction
    u: np.ndarray  # (3,) unit vector in the profile plane
    shape: np.ndarray  # (K, 2) outline in (u, v) coordinates, counterclockwise

    @property
    def v(self):
        return np.cross(self.axis, self.u)


def _tilted_axis(d, sin_t, flip):
    sin_t = float(np.clip(sin_t, MIN_TILT, 1.0))
    cos_t = np.sqrt(max(0.0, 1.0 - sin_t * sin_t))
    return np.array([sin_t * d[0], sin_t * d[1], cos_t if flip else -cos_t])


def circle_profile(s1, s2, s3, sweep_dir, flip=False, segments=48):
    """Circle in 3D whose projection is the ellipse given by S1-S2 and S3."""
    s1, s2, s3, d = (np.asarray(p, dtype=float) for p in (s1, s2, s3, sweep_dir))
    c = 0.5 * (s1 + s2)
    a = 0.5 * np.linalg.norm(s2 - s1)
    if a < 1.0:
        raise ValueError("Profile is too small.")
    um = (s2 - s1) / (2.0 * a)
    vm = perp(um)
    b = abs(np.dot(s3 - c, vm))
    ratio = min(b / a, 1.0)  # = |cos| of the tilt between circle normal and view direction
    side = 1.0 if np.dot(d, vm) >= 0 else -1.0
    axis = _tilted_axis(side * vm, np.sqrt(1.0 - ratio * ratio), flip)
    u = np.array([um[0], um[1], 0.0])  # the major axis stays parallel to the image
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    shape = a * np.stack([np.cos(phi), np.sin(phi)], axis=1)
    return Profile("circle", np.array([c[0], c[1], 0.0]), normalize(axis), u, shape)


def solve_rect_depths(u1, u2, d):
    """Depth offsets (z1, z2) of two base edges so the 3D base is a rectangle.

    The edges are e1 = (u1, z1) and e2 = (u2, z2). Two conditions fix z1, z2
    up to a common sign (orthographic version of the paper's appendix A):
      e1 . e2 = 0                    -> u1.u2 + z1 z2 = 0
      (e1 x e2) projects along d     -> z1 (u2.d) = z2 (u1.d)
    Returns None when there is no real solution.
    """
    p, q, k = float(u1 @ d), float(u2 @ d), float(u1 @ u2)
    if abs(p) < 1e-6 or abs(q) < 1e-6:
        return None
    z2_sq = -k * q / p
    if z2_sq < 0:
        return None
    z2 = np.sqrt(z2_sq)
    return z2 * p / q, z2


def rect_profile(s1, s2, s3, sweep_dir, flip=False):
    """Rectangle in 3D whose projection is the parallelogram S1, S2, S3, S1 + (S3 - S2)."""
    s1, s2, s3, d = (np.asarray(p, dtype=float) for p in (s1, s2, s3, sweep_dir))
    u1, u2 = s2 - s1, s3 - s2
    if np.linalg.norm(u1) < 2.0 or np.linalg.norm(u2) < 2.0:
        raise ValueError("Profile is too small.")
    d = normalize(d)

    sol = solve_rect_depths(u1, u2, d)
    if sol is not None:
        z1, z2 = sol
        for _ in range(2):
            e1 = np.array([u1[0], u1[1], z1])
            e2 = np.array([u2[0], u2[1], z2])
            axis = normalize(np.cross(e1, e2))
            if axis[:2] @ d < 0:
                axis = -axis
            if (axis[2] > 0) == flip:
                break
            z1, z2 = -z1, -z2  # the mirror solution
    else:
        # No exact rectangle fits: tilt the axis by a default amount and
        # lift both edges into the plane perpendicular to it.
        axis = _tilted_axis(d, 0.8, flip)
        e1 = np.array([u1[0], u1[1], -(u1 @ axis[:2]) / axis[2]])
        e2 = np.array([u2[0], u2[1], -(u2 @ axis[:2]) / axis[2]])
        e2 = e2 - (e2 @ normalize(e1)) * normalize(e1)

    p2 = np.array([s2[0], s2[1], 0.0])
    corners = np.array([p2 - e1, p2, p2 + e2, p2 - e1 + e2])
    center = corners.mean(axis=0)
    u = normalize(e1)
    v = np.cross(axis, u)
    rel = corners - center
    shape = np.stack([rel @ u, rel @ v], axis=1)
    area = 0.5 * np.sum(shape[:, 0] * np.roll(shape[:, 1], -1) - np.roll(shape[:, 0], -1) * shape[:, 1])
    if area < 0:
        shape = shape[::-1].copy()
    return Profile("rect", center, axis, u, shape)


def ellipse_outline(s1, s2, s3, segments=64):
    """2D outline of the ellipse defined by the profile strokes (for drawing)."""
    s1, s2, s3 = (np.asarray(p, dtype=float) for p in (s1, s2, s3))
    c = 0.5 * (s1 + s2)
    a = 0.5 * np.linalg.norm(s2 - s1)
    if a < 1e-6:
        return np.array([c])
    um = (s2 - s1) / (2.0 * a)
    vm = perp(um)
    b = abs(np.dot(s3 - c, vm))
    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    return c + a * np.cos(phi)[:, None] * um + b * np.sin(phi)[:, None] * vm
