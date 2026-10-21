"""3D viewer and part editor (Polyscope), an alternative to the built-in view.

Windows Smart App Control blocks Polyscope's DLL on some machines; the
app's built-in 3D view (view3d.py) works there. This one runs in its own
process, because Polyscope and Qt each want to own the main loop:

    python -m threesweep.viewer scene.npz [--out exported.glb]

Each part can be moved, rotated and scaled on its own, then the edited
model can be exported.
"""

import argparse
from pathlib import Path

import numpy as np

from .export import export_model, load_scene
from .mesh import PartMesh, transform_vertices


class EditablePart:
    def __init__(self, part):
        self.part = part
        self.reset()

    def reset(self):
        self.translate = [0.0, 0.0, 0.0]
        self.rotate = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]

    def vertices(self):
        return transform_vertices(self.part.vertices, self.translate, self.rotate, self.scale)

    def edited(self):
        p = self.part
        return PartMesh(p.name, p.kind, self.vertices(), p.faces, p.uv)


def _add_texture(ps_mesh, part, texture_rgb):
    if texture_rgb is None:
        return
    try:
        ps_mesh.add_parameterization_quantity("uv", part.uv, defined_on="vertices")
        tex = texture_rgb.astype(np.float32) / 255.0
        try:
            ps_mesh.add_color_quantity(
                "photo", tex, defined_on="texture", param_name="uv", image_origin="upper_left", enabled=True
            )
        except TypeError:  # older Polyscope without image_origin
            ps_mesh.add_color_quantity("photo", tex[::-1], defined_on="texture", param_name="uv", enabled=True)
    except Exception as err:  # texture is optional; keep the viewer usable
        print(f"Texture not shown: {err}")


def run(parts, image_bgr=None, out_path=None, screenshot=None):
    import polyscope as ps
    import polyscope.imgui as psim

    ps.set_program_name("3-Sweep viewer")
    ps.init()
    ps.set_up_dir("y_up")
    try:
        ps.set_front_dir("z_front")
    except Exception:
        pass
    ps.set_ground_plane_mode("shadow_only")

    texture_rgb = image_bgr[..., ::-1].copy() if image_bgr is not None else None
    editable = [EditablePart(p) for p in parts]
    meshes = []
    for e in editable:
        m = ps.register_surface_mesh(e.part.name, e.vertices(), e.part.faces, smooth_shade=e.part.kind == "circle")
        _add_texture(m, e.part, texture_rgb)
        meshes.append(m)

    extent = float(np.ptp(np.concatenate([p.vertices for p in parts]), axis=0).max()) if parts else 1.0
    state = {"selected": 0, "message": ""}
    out_path = Path(out_path) if out_path else Path.cwd() / "model_edited.glb"

    def slider3(label, values, lo, hi):
        changed = False
        for i, axis in enumerate("XYZ"):
            c, values[i] = psim.SliderFloat(f"{label} {axis}", values[i], lo, hi)
            changed |= c
        return changed

    def callback():
        if not editable:
            psim.Text("No parts.")
            return
        sel = state["selected"]
        psim.Text("Edit one part at a time; the others stay as they are.")
        if psim.BeginCombo("Part", editable[sel].part.name):
            for i, e in enumerate(editable):
                clicked, _ = psim.Selectable(e.part.name, i == sel)
                if clicked:
                    state["selected"] = sel = i
            psim.EndCombo()
        e = editable[sel]
        changed = slider3("Move", e.translate, -extent, extent)
        changed |= slider3("Rotate", e.rotate, -180.0, 180.0)
        changed |= slider3("Scale", e.scale, 0.1, 3.0)
        if psim.Button("Reset part"):
            e.reset()
            changed = True
        if changed:
            meshes[sel].update_vertex_positions(e.vertices())
        psim.Separator()
        if psim.Button("Export GLB"):
            state["message"] = _export(out_path.with_suffix(".glb"), editable, image_bgr)
        psim.SameLine()
        if psim.Button("Export OBJ"):
            state["message"] = _export(out_path.with_suffix(".obj"), editable, image_bgr)
        if state["message"]:
            psim.TextWrapped(state["message"])

    ps.set_user_callback(callback)
    if screenshot:
        ps.reset_camera_to_home_view()
        ps.screenshot(str(screenshot))
        return
    ps.show()


def _export(path, editable, image_bgr):
    try:
        export_model(path, [e.edited() for e in editable], image_bgr)
        return f"Saved: {path}"
    except Exception as err:
        return f"Export failed: {err}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="3-Sweep 3D viewer")
    parser.add_argument("scene", help=".npz file written by the modeling app")
    parser.add_argument("--out", help="where the Export buttons save (default: ./model_edited.glb)")
    parser.add_argument("--screenshot", help="save one rendered frame to this png and exit")
    args = parser.parse_args(argv)
    parts, image = load_scene(args.scene)
    run(parts, image, args.out, args.screenshot)


if __name__ == "__main__":
    main()
