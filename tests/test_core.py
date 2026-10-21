import numpy as np
import pytest
import trimesh

from threesweep import synthetic
from threesweep.edges import EdgeMap
from threesweep.export import export_glb, export_obj, load_scene, save_scene
from threesweep.geometry import euler_matrix, normalize, project, rotation_between
from threesweep.mesh import part_mesh, primitive_mesh, signed_volume
from threesweep.primitives import circle_profile, rect_profile, solve_rect_depths
from threesweep.sweep import SweepOptions, build_primitive


def build(img, strokes, **opts):
    edge_map = EdgeMap(img)
    return build_primitive(
        strokes["kind"], strokes["s1"], strokes["s2"], strokes["s3"], strokes["axis"],
        edge_map, SweepOptions(**opts),
    )


# --- geometry -------------------------------------------------------------

def test_rotation_between_maps_a_to_b():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = normalize(rng.normal(size=3)), normalize(rng.normal(size=3))
        assert np.allclose(rotation_between(a, b) @ a, b, atol=1e-9)
    a = np.array([0.0, 0.0, 1.0])
    assert np.allclose(rotation_between(a, -a) @ a, -a)


def test_circle_profile_projects_to_drawn_ellipse():
    s1, s2, s3 = np.array([-50.0, 0]), np.array([50.0, 0]), np.array([0.0, 20])
    prof = circle_profile(s1, s2, s3, sweep_dir=[0, -1])
    pts = prof.center + prof.shape[:, 0:1] * prof.u + prof.shape[:, 1:2] * prof.v
    assert np.isclose(np.abs(pts[:, 0]).max(), 50, atol=1e-6)
    assert np.isclose(np.abs(pts[:, 1]).max(), 20, atol=0.5)
    # axis points down the image and away from the camera
    assert prof.axis[1] < 0 and prof.axis[2] < 0
    assert np.isclose(np.linalg.norm(prof.axis), 1)


def test_rect_depths_recover_a_true_rectangle():
    rng = np.random.default_rng(1)
    for _ in range(50):
        rot = euler_matrix(*rng.uniform(-60, 60, size=3))
        e1, e2 = rot @ [80.0, 0, 0], rot @ [0, 0, 50.0]
        t = normalize(np.cross(e1, e2))
        if np.linalg.norm(t[:2]) < 0.3 or abs(t[2]) < 0.1:
            continue
        sol = solve_rect_depths(e1[:2], e2[:2], normalize(t[:2]))
        assert sol is not None
        z1, z2 = sol
        assert np.allclose(sorted([(z1, z2), (-z1, -z2)])[0], sorted([(e1[2], e2[2]), (-e1[2], -e2[2])])[0], atol=1e-6)


def test_rect_profile_is_orthogonal():
    prof = rect_profile([0, 0], [80, -10], [110, 20], sweep_dir=[0.1, -1])
    assert abs(prof.u @ prof.axis) < 1e-9
    assert abs(prof.v @ prof.axis) < 1e-9
    assert len(prof.shape) == 4


# --- sweeping on synthetic photos -----------------------------------------

def test_cylinder_radius_and_length():
    img, strokes, truth = synthetic.cylinder()
    res = build(img, strokes)
    r = res.radii()
    assert np.median(r) == pytest.approx(truth["radius"], rel=0.05)
    assert r.min() > 0.85 * truth["radius"] and r.max() < 1.15 * truth["radius"]
    # 2D length matches the stroke; the 3D length is longer because the axis tilts away
    assert np.linalg.norm(np.diff(res.centers[:, :2], axis=0), axis=1).sum() == pytest.approx(truth["length"], rel=0.05)
    assert res.length() > truth["length"]


def test_cylinder_recenters_offset_stroke():
    img, strokes, truth = synthetic.cylinder()
    strokes = dict(strokes, axis=[(x + 15, y) for x, y in strokes["axis"]])
    res = build(img, strokes)
    center_x = project(res.centers)[:, 0]
    assert np.median(center_x[5:]) == pytest.approx(200, abs=2)


def test_cone_tapers():
    img, strokes, truth = synthetic.cone()
    res = build(img, strokes, radius_mode="linear")
    r = res.radii()
    assert r[0] == pytest.approx(truth["r_top"], rel=0.08)
    expected_end = truth["r_top"] + (truth["r_bottom"] - truth["r_top"]) * (strokes["axis"][-1][1] - truth["top"]) / (truth["bottom"] - truth["top"])
    assert r[-1] == pytest.approx(expected_end, rel=0.12)


def test_bent_pipe_follows_curve():
    img, strokes, truth = synthetic.bent_pipe()
    res = build(img, strokes)
    assert np.median(res.radii()) == pytest.approx(truth["radius"], rel=0.1)
    assert len(res.centers) > 20


def test_box_dimensions():
    img, strokes, truth = synthetic.box()
    res = build(img, strokes, radius_mode="constant", straight_axis=True)
    shape = res.profile.shape * res.scales[0]
    sides = np.linalg.norm(np.roll(shape, -1, axis=0) - shape, axis=1)
    assert sorted([sides[0], sides[1]]) == pytest.approx(sorted([truth["width"], truth["depth"]]), rel=0.1)
    assert res.length() == pytest.approx(truth["height"], rel=0.1)


def test_flip_depth_mirrors_the_part():
    img, strokes, _ = synthetic.cylinder()
    a = build(img, strokes)
    b = build(img, strokes, flip_depth=True)
    assert a.centers[-1, 2] < 0 < b.centers[-1, 2]
    assert np.allclose(a.radii(), b.radii(), rtol=0.05)


def test_short_stroke_is_rejected():
    img, strokes, _ = synthetic.cylinder()
    with pytest.raises(ValueError):
        build(img, dict(strokes, axis=[(200, 100), (200, 103)]))


# --- meshes and export ----------------------------------------------------

@pytest.mark.parametrize("maker", [synthetic.cylinder, synthetic.box, synthetic.bent_pipe])
def test_mesh_is_closed_and_outward(maker):
    img, strokes, _ = maker()
    res = build(img, strokes)
    v, f = primitive_mesh(res)
    assert signed_volume(v, f) > 0
    mesh = trimesh.Trimesh(v, f, process=True)  # merges duplicated vertices
    assert mesh.is_watertight


def test_export_round_trip(tmp_path):
    img, strokes, _ = synthetic.cylinder()
    res = build(img, strokes)
    h, w = img.shape[:2]
    parts = [part_mesh(res, "part_0", (w, h))]
    assert parts[0].uv.min() >= -0.01 and parts[0].uv.max() <= 1.01

    glb = export_glb(tmp_path / "model.glb", parts, img)
    scene = trimesh.load(glb)
    assert len(scene.geometry) == 1
    bounds = scene.bounds
    assert bounds[0][1] == pytest.approx(0, abs=1e-6)
    assert (bounds[1] - bounds[0]).max() == pytest.approx(1, abs=1e-6)

    obj = export_obj(tmp_path / "model.obj", parts, img)
    assert (tmp_path / "model.mtl").exists() and (tmp_path / "model_texture.png").exists()
    assert "o part_0" in obj.read_text()

    save_scene(tmp_path / "scene.npz", parts, img)
    loaded, tex = load_scene(tmp_path / "scene.npz")
    assert loaded[0].name == "part_0" and tex.shape == img.shape


def test_project_save_load_keeps_parts_and_edits(tmp_path):
    import cv2

    from threesweep.model import Model, PartSpec

    img, strokes, truth = synthetic.cylinder()
    cv2.imwrite(str(tmp_path / "photo.png"), img)
    model = Model()
    model.load_image(tmp_path / "photo.png")
    spec = PartSpec(kind="circle", s1=list(strokes["s1"]), s2=list(strokes["s2"]), s3=list(strokes["s3"]),
                    axis=[list(p) for p in strokes["axis"]])
    model.add_part(spec)
    before = model.part_mesh(0).vertices
    spec.scale = [2.0, 1.0, 1.0]
    spec.translate = [10.0, 0.0, 0.0]
    edited = model.part_mesh(0).vertices
    width = lambda v: np.ptp(v[:, 0])
    assert width(edited) == pytest.approx(2 * width(before), rel=1e-6)
    assert model.part_mesh(0).uv == pytest.approx(part_mesh(model.results[0], "p", model.image_size).uv)

    model.save(tmp_path / "project.json")
    other = Model()
    other.load(tmp_path / "project.json")
    assert len(other.results) == 1
    assert other.specs[0].scale == [2.0, 1.0, 1.0]
    assert np.allclose(other.part_mesh(0).vertices, edited)
