"""Sweeping a profile along the third stroke (paper section 4, "Sweeping").

The axis stroke is sampled every few pixels. At each sample a copy of the
previous profile is moved there, turned to follow the stroke, and its size
and position are corrected by casting two rays sideways and snapping them
to the object's outline:
  - both sides hit an edge: resize to fit and recenter between the hits
  - one side hits: mirror that side's size to the other side
  - no hits: keep the previous size
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .geometry import (
    normalize,
    perp,
    polyline_length,
    project,
    resample_polyline,
    rotation_between,
    tangents,
    to_image,
    to_world2,
)
from .primitives import MIN_TILT, Profile, circle_profile, rect_profile

RADIUS_MODES = ("free", "constant", "linear")


@dataclass
class SweepOptions:
    straight_axis: bool = False  # sweep along the straight line from start to end
    radius_mode: str = "free"  # "free", "constant" or "linear" size along the axis
    flip_depth: bool = False  # use the other depth solution for the profile
    spacing: float = 5.0  # pixels between slices (paper: 5)
    max_change: float = 0.2  # largest size change between slices (paper: 20%)
    snap: bool = True


@dataclass
class SweepResult:
    profile: Profile
    centers: np.ndarray  # (N, 3)
    axes: np.ndarray  # (N, 3)
    us: np.ndarray  # (N, 3)
    scales: np.ndarray  # (N,) size of each slice relative to the profile
    hits: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # image points

    @property
    def kind(self):
        return self.profile.kind

    def rings(self):
        """(N, K, 3) outline of every slice in world space."""
        shape = self.profile.shape
        vs = np.cross(self.axes, self.us)
        return (
            self.centers[:, None, :]
            + self.scales[:, None, None]
            * (shape[None, :, 0:1] * self.us[:, None, :] + shape[None, :, 1:2] * vs[:, None, :])
        )

    def radii(self):
        """Approximate half-width of each slice (circle: the radius)."""
        return self.scales * np.linalg.norm(self.profile.shape, axis=1).max()

    def length(self):
        return float(np.linalg.norm(np.diff(self.centers, axis=0), axis=1).sum())


def build_primitive(kind, s1, s2, s3, axis_stroke, edge_map=None, options=None):
    """Build one part from the three sweeps. All inputs are image pixel coordinates."""
    options = options or SweepOptions()
    s1, s2, s3 = (np.asarray(p, dtype=float) for p in (s1, s2, s3))
    stroke_img = np.asarray(axis_stroke, dtype=float)
    if len(stroke_img) < 2 or polyline_length(stroke_img) < 2 * options.spacing:
        raise ValueError("The axis stroke is too short.")

    snapping = edge_map is not None and options.snap
    if snapping and kind == "circle":
        # Pull the diameter's ends onto the outline.
        reach = 0.1 * np.linalg.norm(s2 - s1) + 4.0
        s1_new = edge_map.snap_point(s1, s1 - s2, reach)
        s2_new = edge_map.snap_point(s2, s2 - s1, reach)
        shift = 0.5 * ((s1_new + s2_new) - (s1 + s2))
        s1, s2, s3 = s1_new, s2_new, s3 + shift

    stroke = to_world2(stroke_img)
    if options.straight_axis:
        stroke = stroke[[0, -1]]
    pts = resample_polyline(stroke, options.spacing)
    d0 = normalize(pts[min(len(pts) - 1, 4)] - pts[0])

    w1, w2, w3 = to_world2(s1), to_world2(s2), to_world2(s3)
    if kind == "circle":
        profile = circle_profile(w1, w2, w3, d0, flip=options.flip_depth)
    elif kind == "rect":
        profile = rect_profile(w1, w2, w3, d0, flip=options.flip_depth)
    else:
        raise ValueError(f"Unknown primitive kind: {kind}")

    # The sweep starts at the profile center, wherever the stroke began.
    pts = pts - pts[0] + profile.center[:2]
    tan = tangents(pts)

    sin0 = max(float(np.linalg.norm(profile.axis[:2])), MIN_TILT)
    tz = float(profile.axis[2])
    # Constant tilt out of the image plane: exact for straight axes, and the
    # paper assumes curved axes lie mostly parallel to the image anyway.
    depth_per_pixel = tz / sin0

    centers, axes, us, scales, hits = [profile.center.copy()], [profile.axis], [profile.u], [1.0], []
    offset = np.zeros(2)
    for i in range(1, len(pts)):
        d = tan[i]
        axis = normalize(np.array([d[0] * sin0, d[1] * sin0, tz]))
        u = rotation_between(axes[-1], axis) @ us[-1]
        u = normalize(u - (u @ axis) * axis)
        n2 = perp(d)

        cxy = pts[i] + (offset @ n2) * n2  # keep the previous sideways correction
        cz = centers[-1][2] + depth_per_pixel * np.linalg.norm(pts[i] - pts[i - 1])
        scale = scales[-1]

        if snapping:
            v = np.cross(axis, u)
            ring = profile.shape * scale
            side = (ring[:, 0:1] * u[:2] + ring[:, 1:2] * v[:2]) @ n2
            w_pos, w_neg = max(side.max(), 1.0), max(-side.min(), 1.0)
            m = options.max_change
            c_img, n_img = to_image(cxy), to_image(n2)
            h_pos = edge_map.cast_ray(c_img, n_img, (1 - m) * w_pos, (1 + m) * w_pos, expected=w_pos)
            h_neg = edge_map.cast_ray(c_img, -n_img, (1 - m) * w_neg, (1 + m) * w_neg, expected=w_neg)
            factor, shift = 1.0, 0.0
            if h_pos is not None and h_neg is not None:
                factor = (h_pos + h_neg) / (w_pos + w_neg)
                shift = h_pos - factor * w_pos
            elif h_pos is not None:
                factor = h_pos / w_pos
            elif h_neg is not None:
                factor = h_neg / w_neg
            if h_pos is not None:
                hits.append(c_img + h_pos * n_img)
            if h_neg is not None:
                hits.append(c_img - h_neg * n_img)
            scale *= factor
            cxy = cxy + shift * n2

        offset = cxy - pts[i]
        centers.append(np.array([cxy[0], cxy[1], cz]))
        axes.append(axis)
        us.append(u)
        scales.append(scale)

    centers = np.array(centers)
    scales = _apply_radius_mode(np.array(scales), centers, options.radius_mode)
    if len(centers) >= 5:
        centers[:, :2] = uniform_filter1d(centers[:, :2], 3, axis=0, mode="nearest")

    return SweepResult(
        profile=profile,
        centers=centers,
        axes=np.array(axes),
        us=np.array(us),
        scales=scales,
        hits=np.array(hits).reshape(-1, 2),
    )


def _apply_radius_mode(scales, centers, mode):
    """Size constraints from the paper: constant or linearly changing radius."""
    if mode == "constant":
        return np.full_like(scales, np.median(scales))
    if mode == "linear" and len(scales) >= 2:
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))])
        a, b = np.polyfit(s, scales, 1)
        return np.maximum(a * s + b, 1e-3)
    if len(scales) >= 5:
        # A few missed or wrong edge hits should not dent the surface.
        return median_filter(scales, size=5, mode="nearest")
    return scales


def projected_outline(result):
    """Image-space outlines of all slices, (N, K, 2)."""
    return project(result.rings())
