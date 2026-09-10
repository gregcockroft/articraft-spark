"""Score an Articraft run against a bench ask: joints by type, then where the drawers actually move.

A clean compile with the right joint count is not enough. A local model can pass both with an object
whose parts do not fit together, so every prismatic joint is also checked in world space:

  axis  its world axis is horizontal: |axis . up| < 0.1
  out   at its upper limit the child moves away from the parent's centre, horizontally

Revolute and other joints are counted, not geometry-checked. Passing is necessary, not sufficient:
look at the render as well (spark/render/render.sh).

    python spark/score.py <run-dir | file.usdz> [--ask spark/bench/dresser/ask.json]

Exit status 0 means the ask and both checks passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

KINDS = {
    "PhysicsPrismaticJoint": "prismatic",
    "PhysicsRevoluteJoint": "revolute",
    "PhysicsFixedJoint": "fixed",
    "PhysicsSphericalJoint": "spherical",
    "PhysicsDistanceJoint": "distance",
}
AXES = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}


def resolve(target: Path) -> tuple[Path, dict]:
    if target.suffix == ".usdz":
        return target, {}
    record = json.loads((target / "record.json").read_text())
    if not record.get("result"):
        sys.exit(f"{target}: no result (status={record.get('status')}, {record.get('terminate_reason')})")
    return target / record["result"], record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--ask", type=Path)
    a = ap.parse_args()
    usdz, record = resolve(a.target)
    stage = Usd.Stage.Open(str(usdz))
    t = Usd.TimeCode.Default()
    up = AXES[UsdGeom.GetStageUpAxis(stage)]
    boxes = UsdGeom.BBoxCache(t, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def centre(path) -> Gf.Vec3d:
        return boxes.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMidpoint()

    def horizontal(v: Gf.Vec3d) -> float:
        return (v - up * (v * up)).GetLength()

    by_kind: dict[str, int] = {}
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        kind = KINDS.get(prim.GetTypeName(), "other")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if kind != "prismatic":
            continue
        j = UsdPhysics.PrismaticJoint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets()[0], j.GetBody1Rel().GetTargets()[0]
        rot0 = Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get() or Gf.Quatf(1)))
        frame = UsdGeom.Xformable(stage.GetPrimAtPath(b0)).ComputeLocalToWorldTransform(t)
        axis = frame.ExtractRotation().TransformDir(rot0.TransformDir(AXES[j.GetAxisAttr().Get()]))
        upper = j.GetUpperLimitAttr().Get() or 0.0
        before = centre(b1) - centre(b0)
        after = before + axis * upper
        rows.append((prim.GetName(), axis, abs(axis * up) < 0.1, horizontal(after) > horizontal(before) + 1e-6))

    total = sum(by_kind.values())
    print(f"file     {usdz}")
    if record:
        print(f"record   status={record.get('status')}  terminate_reason={record.get('terminate_reason')}")
    print(f"joints   {total}  " + "  ".join(f"{k} {n}" for k, n in sorted(by_kind.items())))
    ok = True
    if a.ask:
        ask = json.loads(a.ask.read_text())["ask"]
        need = int(ask.get("min_joints") or 0)
        short = {k: (n, by_kind.get(k, 0)) for k, n in (ask.get("required") or {}).items() if by_kind.get(k, 0) < n}
        ask_ok = total >= need and not short
        ok &= ask_ok
        print(f"ask      min {need}, required {ask.get('required')}  ->  {'PASS' if ask_ok else 'FAIL'}"
              + (f"  short: {short}" if short else ""))
    for name, axis, flat, out in rows:
        print(f"  {name:22s} world axis ({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})"
              f"  {'horizontal' if flat else 'NOT horizontal'}  {'moves out' if out else 'does NOT move out'}")
    if rows:
        n_flat = sum(r[2] for r in rows)
        n_out = sum(r[3] for r in rows)
        ok &= n_flat == len(rows) and n_out == len(rows)
        print(f"axis     {n_flat}/{len(rows)} prismatic axes horizontal  ->  {'PASS' if n_flat == len(rows) else 'FAIL'}")
        print(f"out      {n_out}/{len(rows)} move away from their parent  ->  {'PASS' if n_out == len(rows) else 'FAIL'}")
    print(f"RESULT   {'PASS' if ok else 'FAIL'}" + ("  (now look at the render)" if ok else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
