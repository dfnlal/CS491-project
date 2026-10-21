"""3D view and part editor inside the app (pyqtgraph + OpenGL).

Pick a part in the list and move, rotate or scale it; the other parts
stay as they are. Edits are stored in the model, so Export and Save
project include them.

Drag with the left mouse button to orbit, the wheel zooms, middle drag
(or Ctrl + left drag) pans.
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph.opengl as gl

from .mesh import primitive_mesh, transform_vertices


# Light from the front, above and to the right (pyqtgraph space, Z up,
# -Y toward where the photo was taken).
LIGHT = np.array([0.35, -0.8, 0.5]) / np.linalg.norm([0.35, -0.8, 0.5])


def photo_colors(part, image_rgb, display_vertices):
    """Per-vertex colors: the photo's pixel under each vertex, with simple
    lighting baked in (pyqtgraph's own shaded mode lights the side facing
    the camera very dark)."""
    h, w = image_rgb.shape[:2]
    x = np.clip((part.uv[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip(((1.0 - part.uv[:, 1]) * h).astype(int), 0, h - 1)
    rgb = image_rgb[y, x].astype(np.float32) / 255.0
    normals = gl.MeshData(vertexes=display_vertices, faces=part.faces).vertexNormals()
    light = 0.55 + 0.45 * np.clip(normals @ LIGHT, 0.0, 1.0)
    rgb = np.clip(rgb * light[:, None], 0.0, 1.0)
    return np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=1)


def to_display(vertices, origin):
    """Model space (Y up) -> pyqtgraph space (Z up), relative to origin."""
    v = vertices - origin
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


class View3D(QWidget):
    def __init__(self, model, export_callback=None):
        super().__init__()
        self.setWindowTitle("3-Sweep - 3D view")
        self.resize(1100, 720)
        self.model = model
        self.export_callback = export_callback
        self.items = []
        self.origin = np.zeros(3)
        self.extent = 100.0
        self.image_rgb = None

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#1e1e1e")
        self.grid = gl.GLGridItem()
        self.grid.setColor((255, 255, 255, 40))
        self.view.addItem(self.grid)

        self.part_list = QListWidget()
        self.part_list.currentRowChanged.connect(self.select_part)

        self.spins = {}
        edit_box = QGroupBox("Selected part")
        grid = QGridLayout(edit_box)
        for col, axis in enumerate("XYZ"):
            grid.addWidget(QLabel(axis), 0, col + 1, alignment=Qt.AlignmentFlag.AlignCenter)
        for row, (key, label, lo, hi, step) in enumerate(
            (
                ("translate", "Move", -1e4, 1e4, 5.0),
                ("rotate", "Rotate", -180.0, 180.0, 5.0),
                ("scale", "Scale", 0.05, 20.0, 0.05),
            ),
            start=1,
        ):
            grid.addWidget(QLabel(label), row, 0)
            for col in range(3):
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setSingleStep(step)
                spin.setDecimals(2)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda value, k=key, c=col: self.edit(k, c, value))
                grid.addWidget(spin, row, col + 1)
                self.spins[(key, col)] = spin
        reset = QPushButton("Reset part")
        reset.clicked.connect(self.reset_part)
        grid.addWidget(reset, 4, 0, 1, 4)

        hint = QLabel(
            "Each part is edited on its own: for example, make a table top\n"
            "wider without touching the legs.\n\n"
            "Mouse: left drag orbits, wheel zooms,\nmiddle drag pans."
        )
        hint.setStyleSheet("color: #888;")
        export = QPushButton("Export model (.glb / .obj)...")
        export.clicked.connect(lambda: self.export_callback and self.export_callback())
        home = QPushButton("Reset camera")
        home.clicked.connect(self.reset_camera)

        side = QVBoxLayout()
        side.addWidget(QLabel("Parts"))
        side.addWidget(self.part_list, 1)
        side.addWidget(edit_box)
        side.addWidget(home)
        side.addWidget(export)
        side.addWidget(hint)
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(330)

        layout = QHBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(panel)

    # --- scene -----------------------------------------------------------

    def refresh(self, reset_camera=False):
        for item in self.items:
            self.view.removeItem(item)
        self.items = []
        if self.model.image is None or not self.model.results:
            self.part_list.clear()
            return
        self.image_rgb = self.model.image[..., ::-1]
        meshes = self.model.part_meshes()
        allv = np.concatenate([m.vertices for m in meshes])
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        self.origin = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
        self.extent = float((hi - lo).max())
        self.grid.setSize(self.extent * 2, self.extent * 2)
        self.grid.setSpacing(self.extent / 10, self.extent / 10)

        for mesh in meshes:
            verts = to_display(mesh.vertices, self.origin)
            item = gl.GLMeshItem(
                vertexes=verts,
                faces=mesh.faces,
                vertexColors=photo_colors(mesh, self.image_rgb, verts),
                smooth=mesh.kind == "circle",
                shader=None,
                glOptions="opaque",
            )
            self.view.addItem(item)
            self.items.append(item)
        # the selected part's axis, drawn on top
        self.axis_line = gl.GLLinePlotItem(color=(1.0, 0.85, 0.2, 1.0), width=3, antialias=True, glOptions="additive")
        self.view.addItem(self.axis_line)
        self.items.append(self.axis_line)

        row = self.part_list.currentRow()
        self.part_list.blockSignals(True)
        self.part_list.clear()
        self.part_list.addItems([self.model.part_name(i) for i in range(len(meshes))])
        self.part_list.blockSignals(False)
        self.part_list.setCurrentRow(min(max(row, 0), len(meshes) - 1))
        self.select_part(self.part_list.currentRow())
        if reset_camera:
            self.reset_camera()

    def reset_camera(self):
        # azimuth -90 puts the camera where the photo was taken from
        self.view.setCameraPosition(
            pos=QVector3D(0.0, 0.0, self.extent * 0.4), distance=self.extent * 2.2, elevation=20, azimuth=-90
        )

    def select_part(self, row):
        if not (0 <= row < len(self.model.specs)):
            return
        self.show_axis(row)
        spec = self.model.specs[row]
        for (key, col), spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(spec, key)[col])
            spin.blockSignals(False)

    def edit(self, key, col, value):
        row = self.part_list.currentRow()
        if not (0 <= row < len(self.model.specs)):
            return
        getattr(self.model.specs[row], key)[col] = float(value)
        self.update_part(row)

    def reset_part(self):
        row = self.part_list.currentRow()
        if 0 <= row < len(self.model.specs):
            self.model.specs[row].reset_edits()
            self.select_part(row)
            self.update_part(row)

    def update_part(self, row):
        mesh = self.model.part_mesh(row)
        verts = to_display(mesh.vertices, self.origin)
        self.items[row].setMeshData(
            vertexes=verts, faces=mesh.faces, vertexColors=photo_colors(mesh, self.image_rgb, verts)
        )
        self.show_axis(row)

    def show_axis(self, row):
        """Draw the part's axis, moved along with the part's edits."""
        result, spec = self.model.results[row], self.model.specs[row]
        pivot = primitive_mesh(result)[0].mean(axis=0)  # same pivot as the part's mesh
        centers = transform_vertices(result.centers, spec.translate, spec.rotate, spec.scale, pivot=pivot)
        self.axis_line.setData(pos=to_display(centers, self.origin))
