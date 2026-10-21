"""Triangle meshes from swept parts."""

from dataclasses import dataclass

import numpy as np

from .geometry import euler_matrix, project


@dataclass
class PartMesh:
    name: str
    kind: str
    vertices: np.ndarray  # (V, 3) world space, units = pixels
    faces: np.ndarray  # (F, 3) counterclockwise seen from outside
    uv: np.ndarray  # (V, 2) texture coordinates into the photo


def primitive_mesh(result):
    """Closed mesh of one part: the side surface plus a cap at each end.

    Cuboid sides and all caps get their own vertices so they shade flat;
    cylinder sides share vertices so they shade smooth.
    """
    rings = result.rings()  # (N, K, 3)
    n, k = rings.shape[:2]
    verts, faces = [], []

    def add(v, f):
        base = sum(len(x) for x in verts)
        verts.append(np.asarray(v, dtype=float))
        faces.append(np.asarray(f, dtype=np.int64) + base)

    if result.kind == "rect":
        for j in range(k):
            strip = np.concatenate([rings[:, j], rings[:, (j + 1) % k]])  # 2N vertices
            i = np.arange(n - 1)
            a, b, c, d = i, n + i, n + i + 1, i + 1
            add(strip, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]))
    else:
        i = np.arange(n - 1)[:, None]
        j = np.arange(k)[None, :]
        a, b = i * k + j, i * k + (j + 1) % k
        c, d = (i + 1) * k + (j + 1) % k, (i + 1) * k + j
        f = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        add(rings.reshape(-1, 3), f)

    j = np.arange(k)
    for end, outward in ((0, False), (n - 1, True)):
        ring = rings[end]
        v = np.concatenate([ring, ring.mean(axis=0, keepdims=True)])
        if outward:
            f = np.stack([j, (j + 1) % k, np.full(k, k)], 1)
        else:
            f = np.stack([(j + 1) % k, j, np.full(k, k)], 1)
        add(v, f)

    vertices = np.concatenate(verts)
    faces = np.concatenate(faces)
    if signed_volume(vertices, faces) < 0:
        faces = faces[:, ::-1].copy()
    return vertices, faces


def signed_volume(vertices, faces):
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def photo_uv(vertices, image_size):
    """Texture coordinates by projecting each vertex onto the photo.

    With the orthographic camera, a back vertex lands on the same pixel as
    the front vertex in front of it, so the back simply mirrors the front
    texture, as in the paper (section 6).
    """
    w, h = image_size
    p = project(vertices)
    return np.stack([p[:, 0] / w, 1.0 - p[:, 1] / h], axis=1)


def transform_vertices(vertices, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1), pivot=None):
    """Scale, rotate (degrees, about X then Y then Z) and move points around
    a pivot, by default their own center."""
    pivot = vertices.mean(axis=0) if pivot is None else pivot
    local = (vertices - pivot) * np.asarray(scale, dtype=float)
    return local @ euler_matrix(*rotate).T + pivot + np.asarray(translate, dtype=float)


def part_mesh(result, name, image_size, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1)):
    """Mesh of one part. The texture comes from where the part sits in the
    photo, so it stays attached when the part is later moved or scaled."""
    v, f = primitive_mesh(result)
    uv = photo_uv(v, image_size)
    v = transform_vertices(v, translate, rotate, scale)
    return PartMesh(name=name, kind=result.kind, vertices=v, faces=f, uv=uv)
