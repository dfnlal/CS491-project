"""A photo plus the parts modeled on it.

Parts are stored as the user's strokes and options, not as meshes, so a
project file stays small and every part can be rebuilt (e.g. after
changing the edge sensitivity).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .edges import EdgeMap, load_image
from .mesh import part_mesh
from .sweep import SweepOptions, build_primitive


@dataclass
class PartSpec:
    kind: str  # "circle" or "rect"
    s1: list
    s2: list
    s3: list
    axis: list = field(default_factory=list)  # axis stroke, image pixels
    straight_axis: bool = False
    radius_mode: str = "free"
    flip_depth: bool = False
    # 3D edits made after modeling, applied around the part's own center
    translate: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotate: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # degrees
    scale: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def reset_edits(self):
        self.translate, self.rotate, self.scale = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    def options(self):
        return SweepOptions(
            straight_axis=self.straight_axis,
            radius_mode=self.radius_mode,
            flip_depth=self.flip_depth,
        )

    def to_dict(self):
        d = asdict(self)
        for key in ("s1", "s2", "s3"):
            d[key] = [float(x) for x in d[key]]
        d["axis"] = [[float(x), float(y)] for x, y in d["axis"]]
        for key in ("translate", "rotate", "scale"):
            d[key] = [float(x) for x in d[key]]
        return d


class Model:
    def __init__(self):
        self.image = None
        self.image_path = None
        self.edge_map = None
        self.sensitivity = 1.0
        self.specs = []
        self.results = []

    @property
    def image_size(self):
        h, w = self.image.shape[:2]
        return w, h

    def load_image(self, path, sensitivity=None):
        self.image = load_image(path)
        self.image_path = str(Path(path).resolve())
        self.specs, self.results = [], []
        self.set_sensitivity(self.sensitivity if sensitivity is None else sensitivity)

    def set_sensitivity(self, sensitivity):
        self.sensitivity = float(sensitivity)
        self.edge_map = EdgeMap(self.image, sensitivity=self.sensitivity)
        self.rebuild()

    def build(self, spec):
        return build_primitive(spec.kind, spec.s1, spec.s2, spec.s3, spec.axis, self.edge_map, spec.options())

    def add_part(self, spec):
        result = self.build(spec)
        self.specs.append(spec)
        self.results.append(result)
        return result

    def remove_last(self):
        if self.specs:
            self.specs.pop()
            self.results.pop()

    def clear(self):
        self.specs, self.results = [], []

    def rebuild(self):
        keep_specs, keep_results = [], []
        for spec in self.specs:
            try:
                keep_results.append(self.build(spec))
                keep_specs.append(spec)
            except ValueError:
                pass
        self.specs, self.results = keep_specs, keep_results

    def part_name(self, i):
        return f"part_{i}_{'cylinder' if self.results[i].kind == 'circle' else 'cuboid'}"

    def part_mesh(self, i):
        s = self.specs[i]
        return part_mesh(self.results[i], self.part_name(i), self.image_size, s.translate, s.rotate, s.scale)

    def part_meshes(self):
        return [self.part_mesh(i) for i in range(len(self.results))]

    def save(self, path):
        path = Path(path)
        image = Path(self.image_path)
        try:
            image_ref = str(image.relative_to(path.parent.resolve()))
        except ValueError:
            image_ref = str(image)
        data = {
            "version": 1,
            "image": image_ref,
            "sensitivity": self.sensitivity,
            "parts": [s.to_dict() for s in self.specs],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        image = Path(data["image"])
        if not image.is_absolute():
            image = path.parent / image
        self.load_image(image, data.get("sensitivity", 1.0))
        self.specs = [PartSpec(**p) for p in data["parts"]]
        self.rebuild()


def ring_outline_points(result, every=4):
    """Indices of slices to draw as outlines over the photo."""
    n = len(result.centers)
    idx = list(range(0, n, every))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return np.array(idx)
