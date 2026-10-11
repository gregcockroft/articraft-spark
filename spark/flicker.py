"""Do any two shapes share a visible face? Reads the USDZ, like spark/handles.py.

Needs pxr, so run with the project venv:
    .venv/bin/python spark/flicker.py <run-dir | file.usdz>

The same rule as the compiler's warn_if_coplanar_faces(), applied to the exported object at its rest pose:
two shapes with a face on the same plane, facing the same way, over the same area that is not buried inside
another shape. A renderer flickers between such faces as the camera moves (z-fighting) - the stripe a
turntable shows across a dresser's side when its rails were built to the outer width, a foot's face flush
with an apron, a bottom board's end flush with a side. Exit status 0 when nothing is found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from articraft.sdk._coplanar import coplanar_faces

MIN_AREA = 1e-4  # warn_if_coplanar_faces()'s default, in m2


def resolve(target: Path) -> Path:
    if target.suffix == ".usdz":
        return target
    record = json.loads((target / "record.json").read_text())
    if not record.get("result"):
        sys.exit(f"{target}: no result (status={record.get('status')})")
    return target / record["result"]


def meshes(stage: Usd.Stage) -> list[tuple[str, np.ndarray, np.ndarray]]:
    time = Usd.TimeCode.Default()
    out = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(time), dtype=float)
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time), dtype=float)
        points = points @ matrix[:3, :3] + matrix[3, :3]
        counts = mesh.GetFaceVertexCountsAttr().Get(time)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time)
        triangles, k = [], 0
        for count in counts:
            triangles += [
                (indices[k], indices[k + j], indices[k + j + 1]) for j in range(1, count - 1)
            ]
            k += count
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        label = f"{body.GetName() if body else '?'}/{prim.GetName()}"
        out.append((label, points, np.asarray(triangles, dtype=np.int64)))
    return out


def main() -> int:
    stage = Usd.Stage.Open(str(resolve(Path(sys.argv[1]))))
    # The compiler's rule: every coincident plane counts, and a shape pair is reported when its
    # planes together reach MIN_AREA.
    every = coplanar_faces(meshes(stage), min_area=0.0)
    totals: dict[tuple[str, str], float] = {}
    for item in every:
        totals[(item.shape_a, item.shape_b)] = (
            totals.get((item.shape_a, item.shape_b), 0.0) + item.visible_area
        )
    found = [item for item in every if totals[(item.shape_a, item.shape_b)] >= MIN_AREA]
    for item in found:
        n, p = item.normal, item.point
        print(
            f"  {item.shape_a:28s} ~ {item.shape_b:28s} facing ({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}) "
            f"{item.visible_area * 1e4:6.1f} cm2 near ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
        )
    total = sum(item.visible_area for item in found) * 1e4
    verdict = "PASS" if not found else "FAIL"
    print(
        f"flicker  {len(found)} shape pair face(s) coincide, {total:.0f} cm2 visible  ->  {verdict}"
    )
    return 0 if not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
