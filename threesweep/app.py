"""Modeling window: draw the three sweeps over a photo (PyQt6).

Drawing one part:
  1. Press and drag across the part: the first profile line (S1 -> S2).
     Cylinder: the full width of the round end. Cuboid: one edge of its base.
  2. Move the mouse and click: the second profile line (S3).
     Cylinder: how open the ellipse is. Cuboid: the base's second edge.
  3. Press and drag along the part: the axis. The part snaps to the
     photo's outlines while you drag; releasing the mouse keeps it.
Esc cancels. Right or middle drag pans, the wheel zooms.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPolygonF, QTransform
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QToolBar,
    QWidget,
)

from .export import export_model, save_scene
from .geometry import project
from .model import Model, PartSpec, ring_outline_points
from .primitives import ellipse_outline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PART_COLORS = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176", "#4db6ac", "#f06292"]

IDLE, PROFILE, MINOR, READY, SWEEP = range(5)
HINTS = {
    IDLE: "Step 1/3 - press and drag across the part (cylinder: its width at one end; cuboid: one base edge).",
    PROFILE: "Step 1/3 - release to finish the first profile line.",
    MINOR: "Step 2/3 - move the mouse to shape the profile, then click.",
    READY: "Step 3/3 - press and drag along the part's axis.",
    SWEEP: "Step 3/3 - keep dragging along the axis, release to keep the part.",
}


def to_qimage(rgb_or_rgba):
    arr = np.ascontiguousarray(rgb_or_rgba)
    h, w, c = arr.shape
    fmt = QImage.Format.Format_RGB888 if c == 3 else QImage.Format.Format_RGBA8888
    return QImage(arr.data, w, h, c * w, fmt).copy()


def polygon(points):
    return QPolygonF([QPointF(float(x), float(y)) for x, y in points])


def pen(color, width=1.5, style=Qt.PenStyle.SolidLine):
    p = QPen(QColor(color), width, style)
    p.setCosmetic(True)  # width in screen pixels, whatever the zoom
    return p


class Canvas(QWidget):
    def __init__(self, window):
        super().__init__()
        self.win = window
        self.model = window.model
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        self.photo = None
        self.edge_overlay = None
        self.show_edges = False
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._pan_from = None
        self.reset_stroke()

    # --- view ------------------------------------------------------------

    def set_image(self, image_bgr):
        self.photo = to_qimage(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self.refresh_edges()
        self.reset_stroke()
        self.fit_view()

    def refresh_edges(self):
        if self.model.edge_map is not None:
            self.edge_overlay = to_qimage(self.model.edge_map.overlay_rgba())
        self.update()

    def fit_view(self):
        if self.photo is None:
            return
        w, h = self.photo.width(), self.photo.height()
        self.zoom = 0.95 * min(self.width() / w, self.height() / h)
        self.pan = QPointF((self.width() - w * self.zoom) / 2, (self.height() - h * self.zoom) / 2)
        self.update()

    def to_image(self, pos):
        return np.array([(pos.x() - self.pan.x()) / self.zoom, (pos.y() - self.pan.y()) / self.zoom])

    def wheelEvent(self, event):
        factor = 1.15 ** (event.angleDelta().y() / 120)
        p = event.position()
        img = self.to_image(p)
        self.zoom = float(np.clip(self.zoom * factor, 0.05, 40))
        self.pan = QPointF(p.x() - img[0] * self.zoom, p.y() - img[1] * self.zoom)
        self.update()

    # --- strokes ---------------------------------------------------------

    def reset_stroke(self):
        self.state = IDLE
        self.s1 = self.s2 = self.s3 = None
        self.axis = []
        self.preview = None
        self.win.show_hint(HINTS[IDLE])
        self.update()

    def current_spec(self):
        return PartSpec(
            kind=self.win.kind(),
            s1=list(self.s1),
            s2=list(self.s2),
            s3=list(self.s3),
            axis=[list(p) for p in self.axis],
            **self.win.part_options(),
        )

    def set_state(self, state):
        self.state = state
        self.win.show_hint(HINTS[state])

    def mousePressEvent(self, event):
        if self.photo is None:
            return
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_from = event.position()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        p = self.to_image(event.position())
        if self.state == IDLE:
            self.s1, self.s2 = p, p.copy()
            self.set_state(PROFILE)
        elif self.state == MINOR:
            self.s3 = p
            self.set_state(READY)
        elif self.state == READY:
            self.axis = [p]
            self.set_state(SWEEP)
        self.update()

    def mouseMoveEvent(self, event):
        if self._pan_from is not None:
            self.pan += event.position() - self._pan_from
            self._pan_from = event.position()
            self.update()
            return
        if self.photo is None:
            return
        p = self.to_image(event.position())
        if self.state == PROFILE:
            self.s2 = p
        elif self.state == MINOR:
            self.s3 = p
        elif self.state == SWEEP and np.linalg.norm(p - self.axis[-1]) >= 2.0:
            self.axis.append(p)
            self.update_preview()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_from = None
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.state == PROFILE:
            if np.linalg.norm(self.s2 - self.s1) < 4:
                self.reset_stroke()
            else:
                self.s3 = self.s2.copy()
                self.set_state(MINOR)
        elif self.state == SWEEP:
            self.finish_part()
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reset_stroke()
        else:
            super().keyPressEvent(event)

    def update_preview(self):
        try:
            self.preview = self.model.build(self.current_spec())
        except ValueError:
            self.preview = None

    def finish_part(self):
        try:
            self.model.add_part(self.current_spec())
        except ValueError as err:
            self.win.statusBar().showMessage(f"Part not added: {err}", 5000)
        self.reset_stroke()
        self.win.parts_changed()

    # --- drawing ---------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2b2b2b"))
        if self.photo is None:
            painter.setPen(QColor("#bbbbbb"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a photo to start (File > Open photo)")
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setTransform(QTransform(self.zoom, 0, 0, self.zoom, self.pan.x(), self.pan.y()))
        painter.drawImage(0, 0, self.photo)
        if self.show_edges and self.edge_overlay is not None:
            painter.drawImage(0, 0, self.edge_overlay)

        for i, result in enumerate(self.model.results):
            self.draw_result(painter, result, PART_COLORS[i % len(PART_COLORS)])
        if self.preview is not None:
            self.draw_result(painter, self.preview, "#ffffff", show_hits=True)
        self.draw_strokes(painter)

    def draw_result(self, painter, result, color, show_hits=False):
        rings = project(result.rings())
        idx = ring_outline_points(result)
        painter.setPen(pen(color, 1.0))
        for i in idx[1:-1]:
            painter.drawPolygon(polygon(rings[i]))
        painter.setPen(pen(color, 2.0))
        painter.drawPolygon(polygon(rings[0]))
        painter.drawPolygon(polygon(rings[-1]))
        # silhouette: the outermost ring points on each side of the axis
        centers = project(result.centers)
        left, right = [], []
        for i in range(len(rings)):
            d = centers[min(i + 1, len(centers) - 1)] - centers[max(i - 1, 0)]
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            side = (rings[i] - centers[i]) @ n
            left.append(rings[i][np.argmin(side)])
            right.append(rings[i][np.argmax(side)])
        painter.drawPolyline(polygon(left))
        painter.drawPolyline(polygon(right))
        painter.setPen(pen(color, 1.0, Qt.PenStyle.DashLine))
        painter.drawPolyline(polygon(centers))
        if show_hits and len(result.hits):
            painter.setPen(pen("#ff4081", 4.0))
            painter.drawPoints(polygon(result.hits))

    def draw_strokes(self, painter):
        if self.state == IDLE:
            return
        kind = self.win.kind()
        painter.setPen(pen("#ff5252", 2.0))
        painter.drawLine(QPointF(*self.s1), QPointF(*self.s2))
        if self.state in (MINOR, READY, SWEEP) and self.s3 is not None:
            painter.setPen(pen("#69f0ae", 2.0))
            if kind == "circle":
                c = 0.5 * (self.s1 + self.s2)
                painter.drawLine(QPointF(*c), QPointF(*self.s3))
                painter.setPen(pen("#448aff", 2.0))
                painter.drawPolyline(polygon(ellipse_outline(self.s1, self.s2, self.s3)))
            else:
                painter.drawLine(QPointF(*self.s2), QPointF(*self.s3))
                s4 = self.s1 + (self.s3 - self.s2)
                painter.setPen(pen("#448aff", 1.5, Qt.PenStyle.DashLine))
                painter.drawPolyline(polygon([self.s3, s4, self.s1]))
        if self.state == SWEEP and len(self.axis) > 1:
            painter.setPen(pen("#448aff", 2.0))
            painter.drawPolyline(polygon(self.axis))


class MainWindow(QMainWindow):
    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("3-Sweep")
        self.model = Model()
        self.hint_label = QLabel()
        self.statusBar().addWidget(self.hint_label, 1)
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self._build_menus()
        self._build_toolbar()
        self.resize(1300, 860)
        self.viewer_procs = []
        self.view3d = None
        if image_path:
            self.load_image(image_path)

    # --- UI setup --------------------------------------------------------

    def _action(self, menu, text, slot, shortcut=None):
        act = QAction(text, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        menu.addAction(act)
        return act

    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "Open photo...", self.open_image, "Ctrl+O")
        self._action(file_menu, "Open project...", self.open_project, "Ctrl+Shift+O")
        self._action(file_menu, "Save project...", self.save_project, "Ctrl+S")
        file_menu.addSeparator()
        self._action(file_menu, "Export model (.glb / .obj)...", self.export_model, "Ctrl+E")
        edit_menu = self.menuBar().addMenu("&Edit")
        self._action(edit_menu, "Undo last part", self.undo_part, "Ctrl+Z")
        self._action(edit_menu, "Delete all parts", self.clear_parts)
        view_menu = self.menuBar().addMenu("&View")
        self._action(view_menu, "Fit photo to window", lambda: self.canvas.fit_view(), "F")
        self._action(view_menu, "Show 3D", self.show_3d, "Ctrl+3")
        self._action(view_menu, "Show 3D in Polyscope", self.show_3d_polyscope)

    def _build_toolbar(self):
        tb = QToolBar("Tools")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Primitive: "))
        self.kind_box = QComboBox()
        self.kind_box.addItems(["Cylinder", "Cuboid"])
        self.kind_box.currentIndexChanged.connect(lambda _: self.canvas.reset_stroke())
        tb.addWidget(self.kind_box)

        self.straight_box = QCheckBox("Straight axis")
        tb.addWidget(self.straight_box)

        tb.addWidget(QLabel("  Size along axis: "))
        self.radius_box = QComboBox()
        self.radius_box.addItems(["Free", "Constant", "Linear"])
        tb.addWidget(self.radius_box)

        self.flip_box = QCheckBox("Flip depth")
        self.flip_box.setToolTip("Use when the part comes out tilted the wrong way (toward the camera instead of away).")
        tb.addWidget(self.flip_box)
        tb.addSeparator()

        self.edges_box = QCheckBox("Show edges")
        self.edges_box.toggled.connect(self.toggle_edges)
        tb.addWidget(self.edges_box)
        tb.addWidget(QLabel(" Edge sensitivity: "))
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(3, 30)
        self.sens_slider.setValue(10)
        self.sens_slider.setFixedWidth(120)
        self.sens_slider.sliderReleased.connect(self.sensitivity_changed)
        tb.addWidget(self.sens_slider)
        tb.addSeparator()

        for text, slot in (("Undo part", self.undo_part), ("Show 3D", self.show_3d), ("Export", self.export_model)):
            act = QAction(text, self)
            act.triggered.connect(slot)
            tb.addAction(act)

    # --- state for the canvas -------------------------------------------

    def kind(self):
        return "circle" if self.kind_box.currentIndex() == 0 else "rect"

    def part_options(self):
        return {
            "straight_axis": self.straight_box.isChecked(),
            "radius_mode": ("free", "constant", "linear")[self.radius_box.currentIndex()],
            "flip_depth": self.flip_box.isChecked(),
        }

    def show_hint(self, text):
        self.hint_label.setText(text)

    def parts_changed(self):
        n = len(self.model.results)
        self.setWindowTitle(f"3-Sweep - {Path(self.model.image_path).name} - {n} part{'s' if n != 1 else ''}")
        self.canvas.update()
        if self.view3d is not None and self.view3d.isVisible():
            self.view3d.refresh()

    # --- actions ---------------------------------------------------------

    def require_image(self):
        if self.model.image is None:
            QMessageBox.information(self, "3-Sweep", "Open a photo first.")
            return False
        return True

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open photo", str(PROJECT_ROOT / "samples"), "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            self.model.load_image(path, self.sens_slider.value() / 10)
        except ValueError as err:
            QMessageBox.warning(self, "3-Sweep", str(err))
            return
        self.canvas.set_image(self.model.image)
        self.parts_changed()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "3-Sweep project (*.json)")
        if not path:
            return
        try:
            self.model.load(path)
        except (OSError, ValueError, KeyError) as err:
            QMessageBox.warning(self, "3-Sweep", f"Could not open project: {err}")
            return
        self.sens_slider.setValue(round(self.model.sensitivity * 10))
        self.canvas.set_image(self.model.image)
        self.parts_changed()

    def save_project(self):
        if not self.require_image():
            return
        default = str(Path(self.model.image_path).with_suffix(".json"))
        path, _ = QFileDialog.getSaveFileName(self, "Save project", default, "3-Sweep project (*.json)")
        if path:
            self.model.save(path)
            self.statusBar().showMessage(f"Saved {path}", 4000)

    def undo_part(self):
        self.model.remove_last()
        self.canvas.reset_stroke()
        if self.model.image is not None:
            self.parts_changed()

    def clear_parts(self):
        self.model.clear()
        self.canvas.reset_stroke()
        if self.model.image is not None:
            self.parts_changed()

    def toggle_edges(self, on):
        self.canvas.show_edges = on
        self.canvas.update()

    def sensitivity_changed(self):
        if self.model.image is None:
            return
        self.model.set_sensitivity(self.sens_slider.value() / 10)
        self.canvas.refresh_edges()
        self.parts_changed()

    def show_3d(self):
        if not self.require_image():
            return
        if not self.model.results:
            QMessageBox.information(self, "3-Sweep", "Model at least one part first.")
            return
        try:
            if self.view3d is None:
                from .view3d import View3D

                self.view3d = View3D(self.model, export_callback=self.export_model)
            self.view3d.refresh(reset_camera=not self.view3d.isVisible())
            self.view3d.show()
            self.view3d.raise_()
            self.view3d.activateWindow()
        except Exception as err:  # e.g. no OpenGL: fall back to the Polyscope window
            self.view3d = None
            self.statusBar().showMessage(f"Built-in 3D view failed ({err}); trying Polyscope.", 6000)
            self.show_3d_polyscope()

    def show_3d_polyscope(self):
        """Polyscope viewer in a separate process (Polyscope runs its own main loop).

        Windows Smart App Control can block Polyscope's DLL; the built-in
        view above does not need it.
        """
        if not self.require_image() or not self.model.results:
            return
        folder = Path(tempfile.mkdtemp(prefix="threesweep_"))
        scene = folder / "scene.npz"
        save_scene(scene, self.model.part_meshes(), self.model.image)
        out = Path(self.model.image_path).with_name(Path(self.model.image_path).stem + "_edited.glb")
        proc = subprocess.Popen(
            [sys.executable, "-m", "threesweep.viewer", str(scene), "--out", str(out)], cwd=str(PROJECT_ROOT)
        )
        self.viewer_procs.append(proc)
        self.statusBar().showMessage("Polyscope viewer opened in a new window. Its exports go next to the photo.", 6000)

    def export_model(self):
        if not self.require_image():
            return
        if not self.model.results:
            QMessageBox.information(self, "3-Sweep", "Model at least one part first.")
            return
        default = str(Path(self.model.image_path).with_suffix(".glb"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export model", default, "glTF binary (*.glb);;Wavefront OBJ (*.obj)"
        )
        if not path:
            return
        try:
            saved = export_model(path, self.model.part_meshes(), self.model.image)
        except (OSError, ValueError) as err:
            QMessageBox.warning(self, "3-Sweep", f"Export failed: {err}")
            return
        self.statusBar().showMessage(f"Exported {saved}", 5000)

    def closeEvent(self, event):
        for proc in self.viewer_procs:
            if proc.poll() is None:
                proc.terminate()
        if self.view3d is not None:
            self.view3d.close()
        super().closeEvent(event)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1])
    window = MainWindow(argv[0] if argv else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
