"""Coincident faces between shapes: the surfaces a renderer flickers between (z-fighting).

Two shapes that each have a face on the same plane, facing the same way, over the same area, are drawn
in whichever order the depth buffer happens to pick from frame to frame, so the shared area flickers as
the camera moves. Inside one part this is ordinary overlap and no collision check sees it. It is reported
only where the shared area is visible: a sample lifted just off the face must not be inside any shape.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import igl
import numpy as np

PLANE_TOL = 2e-4  # faces within 0.2 mm of one plane coincide
NORMAL_DOT = 0.9995  # and face within ~1.8 degrees of one direction
STEP = 0.002  # the shared area is sampled on a 2 mm grid
LIFT = 3e-4  # a sample is visible when the point 0.3 mm off the face is inside no shape


@dataclass(frozen=True)
class CoplanarFaces:
    shape_a: str
    shape_b: str
    normal: tuple[float, float, float]
    offset: float  # the plane is normal . p = offset
    visible_area: float
    point: tuple[float, float, float]  # a visible point on the shared area


@dataclass(frozen=True)
class _Plane:
    normal: np.ndarray
    offset: float
    triangles: np.ndarray  # (k, 3, 3) world vertices


def coplanar_faces(
    meshes: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    min_area: float = 1e-4,
    step: float = STEP,
) -> list[CoplanarFaces]:
    """Coincident, same-facing, visible faces between every pair of the given (label, vertices, faces)."""

    meshes = [
        (label, np.asarray(v, float), _outward(np.asarray(v, float), np.asarray(f, np.int64)))
        for label, v, f in meshes
    ]
    planes = [_planes(v, f) for _label, v, f in meshes]
    normals = [np.array([plane.normal for plane in group]).reshape(-1, 3) for group in planes]
    offsets = [np.array([plane.offset for plane in group]) for group in planes]
    lows = [v.min(axis=0) for _label, v, _f in meshes]
    highs = [v.max(axis=0) for _label, v, _f in meshes]
    found: list[CoplanarFaces] = []
    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            if np.any(lows[i] > highs[j] + PLANE_TOL) or np.any(lows[j] > highs[i] + PLANE_TOL):
                continue
            # Every plane of one against every plane of the other at once: a curved mesh has a plane
            # per facet, and a Python loop over their pairs costs seconds.
            matches = (normals[i] @ normals[j].T >= NORMAL_DOT) & (
                np.abs(offsets[i][:, None] - offsets[j][None, :]) <= PLANE_TOL
            )
            for a, b in zip(*np.nonzero(matches), strict=True):
                plane_a = planes[i][a]
                hit = _shared_visible(plane_a, planes[j][b], meshes, lows, highs, step)
                if hit is not None and hit[0] >= min_area:
                    n, p = plane_a.normal, hit[1]
                    normal = (float(n[0]), float(n[1]), float(n[2]))
                    point = (float(p[0]), float(p[1]), float(p[2]))
                    found.append(
                        CoplanarFaces(
                            meshes[i][0], meshes[j][0], normal, plane_a.offset, hit[0], point
                        )
                    )
    return sorted(found, key=lambda item: -item.visible_area)


def _outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The faces wound outward: a closed mesh with a negative signed volume is turned inside out."""

    tri = vertices[faces]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
    return faces[:, ::-1] if volume < 0 else faces


def _planes(vertices: np.ndarray, faces: np.ndarray) -> list[_Plane]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > 1e-12
    tri, normal = tri[keep], cross[keep] / double_area[keep, None]
    offset = np.einsum("ij,ij->i", normal, tri[:, 0])
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, (n, d) in enumerate(
        zip(np.round(normal, 3), np.round(offset / PLANE_TOL), strict=True)
    ):
        groups[(*n, d)].append(index)
    return [
        _Plane(
            normal[idx].mean(axis=0) / np.linalg.norm(normal[idx].mean(axis=0)),
            float(offset[idx].mean()),
            tri[idx],
        )
        for idx in (np.asarray(members) for members in groups.values())
    ]


def _shared_visible(
    plane_a: _Plane,
    plane_b: _Plane,
    meshes: Sequence[tuple[str, np.ndarray, np.ndarray]],
    lows: list[np.ndarray],
    highs: list[np.ndarray],
    step: float,
) -> tuple[float, np.ndarray] | None:
    n = plane_a.normal
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    flat_a = np.stack([plane_a.triangles @ u, plane_a.triangles @ v], axis=-1)
    flat_b = np.stack([plane_b.triangles @ u, plane_b.triangles @ v], axis=-1)
    low = np.maximum(flat_a.reshape(-1, 2).min(axis=0), flat_b.reshape(-1, 2).min(axis=0))
    high = np.minimum(flat_a.reshape(-1, 2).max(axis=0), flat_b.reshape(-1, 2).max(axis=0))
    extent = high - low
    if np.any(extent <= 0):
        return None
    # At most `step` apart on each axis, and at least one row across a strip narrower than a step.
    counts = np.maximum(np.ceil(extent / step), 1).astype(int)
    spacing = extent / counts
    grid = np.stack(
        np.meshgrid(*(low[k] + spacing[k] * (np.arange(counts[k]) + 0.5) for k in range(2))),
        axis=-1,
    )
    shared = _covered(grid, low, spacing, flat_a) & _covered(grid, low, spacing, flat_b)
    grid = grid[shared]
    if len(grid) == 0:
        return None
    points = grid[:, :1] * u + grid[:, 1:] * v + (plane_a.offset + LIFT) * n
    hidden = np.zeros(len(points), bool)
    for (_label, vertices, faces), lo, hi in zip(meshes, lows, highs, strict=True):
        if np.any(points.min(axis=0) > hi) or np.any(points.max(axis=0) < lo):
            continue
        winding = igl.fast_winding_number(
            np.asarray(vertices, float), np.asarray(faces, np.int64), points
        )
        # A mesh wound inward reads -1 inside, so the magnitude decides.
        hidden |= np.abs(np.asarray(winding)) > 0.5
    visible = points[~hidden]
    if len(visible) == 0:
        return None
    return len(visible) * spacing[0] * spacing[1], visible[len(visible) // 2] - LIFT * n


def _covered(
    grid: np.ndarray, low: np.ndarray, spacing: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Which points of the regular grid (rows, columns, 2) lie in any of the 2D triangles (k, 3, 2).

    Each triangle is tested only against the grid points inside its own bounds, so the cost follows
    the area covered rather than points times triangles.
    """

    rows, columns = grid.shape[:2]
    result = np.zeros((rows, columns), bool)
    first = np.floor((triangles.min(axis=1) - low) / spacing - 0.5).astype(int) + 1
    last = np.ceil((triangles.max(axis=1) - low) / spacing - 0.5).astype(int)
    first = np.maximum(first - 1, 0)
    last = np.minimum(last + 1, [columns - 1, rows - 1])
    for tri, (c0, r0), (c1, r1) in zip(triangles, first, last, strict=True):
        if c1 < c0 or r1 < r0:
            continue
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        result[r0 : r1 + 1, c0 : c1 + 1] |= _in_triangle(window, tri)
    return result


def _in_triangle(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    v0, v1, v2 = tri[2] - tri[0], tri[1] - tri[0], points - tri[0]
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) <= 1e-18:
        return np.zeros(points.shape[:-1], bool)
    d02, d12 = v2 @ v0, v2 @ v1
    s = (d11 * d02 - d01 * d12) / den
    t = (d00 * d12 - d01 * d02) / den
    return (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9)
