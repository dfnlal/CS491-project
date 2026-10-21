"""Image loading, edge detection and snapping to edges (paper section 3-4).

The paper uses gPb contours [Arbelaez et al. 2011] plus edge linking.
Here a Canny edge map is enough for a first version; a better edge
detector can replace `EdgeMap.__init__` without touching the rest.
"""

from pathlib import Path

import cv2
import numpy as np

from .geometry import normalize


def load_image(path, max_size=1600):
    """Read an image as BGR, shrinking it so its longer side is at most max_size.

    cv2.imread cannot open paths with non-ASCII characters on Windows
    (e.g. Turkish letters), so the bytes are read with NumPy first.
    """
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    s = max_size / max(h, w)
    if s < 1.0:
        image = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return image


class EdgeMap:
    """Edge pixels of a photo plus the image gradient used to check edge direction."""

    def __init__(self, image_bgr, sensitivity=1.0, blur_sigma=1.2):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        blurred = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
        otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high = max(float(otsu), 20.0) / max(sensitivity, 1e-3)
        self.edges = cv2.Canny(blurred, 0.5 * high, high, L2gradient=True) > 0
        # Canny edges are 1 px thin; a ray stepping half a pixel can slip
        # between diagonal pixels, so hits are tested on a slightly thicker mask.
        self.hit_mask = cv2.dilate(self.edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        self.gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        self.h, self.w = gray.shape

    def cast_ray(self, origin, direction, d_min, d_max, expected=None, step=0.5, max_angle_deg=60.0):
        """Distance along a ray (image space) to the best edge crossing, or None.

        Only edges roughly across the ray count: the angle between the ray
        and the edge normal must be under max_angle_deg (paper: pi/3).
        Among the crossings in [d_min, d_max], the one closest to `expected`
        wins, because the nearest edge is often the wrong one (paper section 4).
        """
        origin = np.asarray(origin, dtype=float)
        direction = normalize(direction)
        d_min = max(float(d_min), 0.0)
        if d_max <= d_min:
            return None
        ts = np.arange(d_min, d_max + 1e-9, step)
        pts = origin + ts[:, None] * direction
        xi = np.rint(pts[:, 0]).astype(int)
        yi = np.rint(pts[:, 1]).astype(int)
        inside = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h)
        if not inside.any():
            return None
        ts, xi, yi = ts[inside], xi[inside], yi[inside]
        g = np.stack([self.gx[yi, xi], self.gy[yi, xi]], axis=1)
        gn = np.linalg.norm(g, axis=1)
        cos_angle = np.abs(g @ direction) / np.maximum(gn, 1e-9)
        ok = self.hit_mask[yi, xi] & (gn > 1e-6) & (cos_angle >= np.cos(np.radians(max_angle_deg)))
        if not ok.any():
            return None
        cand = ts[ok]
        breaks = np.where(np.diff(cand) > 1.5 * step + 1e-9)[0]
        centers = np.array([grp.mean() for grp in np.split(cand, breaks + 1)])
        if expected is None:
            return float(centers[0])
        return float(centers[np.argmin(np.abs(centers - expected))])

    def snap_point(self, p, direction, radius):
        """Move p along +/- direction to the nearest edge within radius."""
        p = np.asarray(p, dtype=float)
        best = None
        for sign in (1.0, -1.0):
            t = self.cast_ray(p, sign * np.asarray(direction, dtype=float), 0.0, radius, expected=0.0)
            if t is not None and (best is None or t < best[0]):
                best = (t, sign)
        if best is None:
            return p
        return p + best[1] * best[0] * normalize(direction)

    def overlay_rgba(self, color=(255, 60, 60, 220)):
        """RGBA image with edge pixels colored, for drawing over the photo."""
        out = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        out[self.edges] = color
        return out
