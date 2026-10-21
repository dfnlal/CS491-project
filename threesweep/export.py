"""Saving parts for game engines and other tools.

Each part stays a separate object in the file, so it can still be edited
on its own (e.g. scale a table top without touching the legs).

Exported models are Y-up, +Z toward the front (glTF convention), scaled
so the whole object fits in a 1 x 1 x 1 box, centered on X/Z and resting
on Y = 0.
"""

from pathlib import Path

import cv2
import numpy as np

from .mesh import PartMesh


def to_export_space(parts):
    """Copies of the parts moved and scaled into export units."""
    if not parts:
        return []
    allv = np.concatenate([p.vertices for p in parts])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    scale = 1.0 / max(float((hi - lo).max()), 1e-9)
    offset = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return [
        PartMesh(p.name, p.kind, (p.vertices - offset) * scale, p.faces, p.uv)
        for p in parts
    ]


def _write_png(path, image_bgr):
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise IOError("Could not encode texture")
    buf.tofile(str(path))


def export_obj(path, parts, image_bgr=None):
    """Wavefront OBJ with one object per part, plus .mtl and texture png."""
    path = Path(path)
    parts = to_export_space(parts)
    lines = []
    if image_bgr is not None:
        tex_name = path.stem + "_texture.png"
        mtl_name = path.stem + ".mtl"
        _write_png(path.with_name(tex_name), image_bgr)
        path.with_name(mtl_name).write_text(
            "newmtl photo\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd " + tex_name + "\n",
            encoding="utf-8",
        )
        lines.append(f"mtllib {mtl_name}")
    base = 1
    for p in parts:
        lines.append(f"o {p.name}")
        if image_bgr is not None:
            lines.append("usemtl photo")
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in p.vertices)
        lines.extend(f"vt {u:.6f} {v:.6f}" for u, v in p.uv)
        for a, b, c in p.faces + base:
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
        base += len(p.vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_glb(path, parts, image_bgr=None):
    """Binary glTF (.glb), the easiest format for Godot, Unity and Blender."""
    import trimesh
    from PIL import Image

    path = Path(path)
    parts = to_export_space(parts)
    texture = None
    if image_bgr is not None:
        texture = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    scene = trimesh.Scene()
    for p in parts:
        if texture is not None:
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0
            )
            visual = trimesh.visual.TextureVisuals(uv=p.uv, material=material)
        else:
            visual = None
        mesh = trimesh.Trimesh(p.vertices, p.faces, visual=visual, process=False)
        scene.add_geometry(mesh, node_name=p.name, geom_name=p.name)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def export_model(path, parts, image_bgr=None):
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return export_obj(path, parts, image_bgr)
    if suffix in (".glb", ".gltf"):
        return export_glb(Path(path).with_suffix(".glb"), parts, image_bgr)
    raise ValueError(f"Unsupported format: {suffix} (use .glb or .obj)")


def save_scene(path, parts, image_bgr=None):
    """Bundle parts and texture into one .npz file for the 3D viewer process."""
    data = {"count": np.array(len(parts))}
    for i, p in enumerate(parts):
        data[f"name_{i}"] = np.array(p.name)
        data[f"kind_{i}"] = np.array(p.kind)
        data[f"vertices_{i}"] = p.vertices
        data[f"faces_{i}"] = p.faces
        data[f"uv_{i}"] = p.uv
    if image_bgr is not None:
        data["texture"] = image_bgr
    np.savez_compressed(path, **data)


def load_scene(path):
    data = np.load(path, allow_pickle=False)
    parts = [
        PartMesh(
            str(data[f"name_{i}"]),
            str(data[f"kind_{i}"]),
            data[f"vertices_{i}"],
            data[f"faces_{i}"],
            data[f"uv_{i}"],
        )
        for i in range(int(data["count"]))
    ]
    image = data["texture"] if "texture" in data.files else None
    return parts, image
