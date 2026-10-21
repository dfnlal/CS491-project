"""Drives the modeling window with synthetic mouse events (no screen needed)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from threesweep import synthetic  # noqa: E402
from threesweep.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def mouse(canvas, kind, img_point, buttons=Qt.MouseButton.LeftButton):
    x, y = img_point
    pos = QPointF(x * canvas.zoom + canvas.pan.x(), y * canvas.zoom + canvas.pan.y())
    types = {
        "press": QMouseEvent.Type.MouseButtonPress,
        "move": QMouseEvent.Type.MouseMove,
        "release": QMouseEvent.Type.MouseButtonRelease,
    }
    button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    held = Qt.MouseButton.NoButton if kind == "release" else buttons
    event = QMouseEvent(types[kind], pos, pos, button, held, Qt.KeyboardModifier.NoModifier)
    getattr(canvas, {"press": "mousePressEvent", "move": "mouseMoveEvent", "release": "mouseReleaseEvent"}[kind])(event)


def draw_part(canvas, strokes):
    mouse(canvas, "press", strokes["s1"])
    mouse(canvas, "move", strokes["s2"])
    mouse(canvas, "release", strokes["s2"])
    mouse(canvas, "move", strokes["s3"])
    mouse(canvas, "press", strokes["s3"])
    mouse(canvas, "release", strokes["s3"])
    axis = np.array(strokes["axis"], float)
    t = np.linspace(0, 1, 60)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(axis, axis=0), axis=1))])
    pts = np.stack([np.interp(t * s[-1], s, axis[:, 0]), np.interp(t * s[-1], s, axis[:, 1])], 1)
    mouse(canvas, "press", pts[0])
    for p in pts[1:]:
        mouse(canvas, "move", p)
    mouse(canvas, "release", pts[-1])


def test_draw_cylinder_and_cuboid(qapp, tmp_path):
    import cv2

    img, strokes, truth = synthetic.cylinder()
    path = tmp_path / "cyl.png"
    cv2.imwrite(str(path), img)
    win = MainWindow(str(path))
    win.resize(900, 700)
    win.canvas.resize(900, 650)
    win.canvas.fit_view()

    draw_part(win.canvas, strokes)
    assert len(win.model.results) == 1
    assert np.median(win.model.results[0].radii()) == pytest.approx(truth["radius"], rel=0.06)

    win.kind_box.setCurrentIndex(1)
    box_img, box_strokes, _ = synthetic.box()
    draw_part(win.canvas, box_strokes)  # strokes on the wrong photo still make a part
    assert len(win.model.results) == 2
    assert win.model.results[1].kind == "rect"

    win.canvas.grab().save(str(tmp_path / "canvas.png"))
    win.undo_part()
    assert len(win.model.results) == 1
    win.close()
