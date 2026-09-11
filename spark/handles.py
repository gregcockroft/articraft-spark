"""Does every drawer carry a handle that stands proud of its front? Reads the USDZ physics prims.

Needs pxr, so run with the articraft conda python:
    $HOME/micromamba/envs/articraft/bin/python handles.py <run-dir | file.usdz>

For each prismatic joint, body1 is the drawer and the joint's world axis is the opening direction.
A drawer passes when at least one of its shapes named like a handle (handle/pull/knob) reaches
further along the opening axis than every other shape of that drawer: the handle is the frontmost
thing on the drawer. Exit status 0 when every drawer passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

HANDLE_WORDS = ("handle", "pull", "knob", "grip")


def resolve(target: Path) -> Path:
    if target.suffix == ".usdz":
        return target
    record = json.loads((target / "record.json").read_text())
    if not record.get("result"):
        sys.exit(f"{target}: no result (status={record.get('status')})")
    return target / record["result"]


def main() -> int:
    usdz = resolve(Path(sys.argv[1]))
    stage = Usd.Stage.Open(str(usdz))
    t = Usd.TimeCode.Default()
    boxes = UsdGeom.BBoxCache(t, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    xf = UsdGeom.XformCache(t)

    def reach(prim, axis: Gf.Vec3d) -> float:
        r = boxes.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = r.GetMin(), r.GetMax()
        return max(
            Gf.Vec3d(x, y, z) * axis for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])
        )

    drawers = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        joint = UsdPhysics.PrismaticJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0 or not body1:
            continue
        axis_name = joint.GetAxisAttr().Get() or "X"
        local_axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[axis_name]
        rot0 = joint.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
        world0 = xf.GetLocalToWorldTransform(stage.GetPrimAtPath(body0[0]))
        axis = Gf.Rotation(Gf.Quatd(rot0)).TransformDir(local_axis)
        axis = world0.TransformDir(axis).GetNormalized()
        upper = joint.GetUpperLimitAttr().Get()
        if upper is not None and upper < 0:
            axis = -axis
        drawers.append((prim.GetName(), stage.GetPrimAtPath(body1[0]), axis))

    ok = 0
    for joint_name, drawer, axis in drawers:
        shapes = [p for p in Usd.PrimRange(drawer) if p.IsA(UsdGeom.Gprim)]
        handles = [p for p in shapes if any(w in p.GetName().lower() for w in HANDLE_WORDS)]
        others = [p for p in shapes if p not in handles]
        if not handles:
            print(f"  {drawer.GetName():14s} NO handle shape   ({len(shapes)} shapes: {', '.join(p.GetName() for p in shapes)[:80]})")
            continue
        front_h = max(reach(p, axis) for p in handles)
        front_o = max(reach(p, axis) for p in others) if others else -1e9
        proud = front_h - front_o
        good = proud > 0.001
        ok += good
        print(f"  {drawer.GetName():14s} handle(s) {', '.join(p.GetName() for p in handles)[:40]:40s} proud of the front by {proud * 1000:+.1f} mm  {'OK' if good else 'NOT proud'}")
    print(f"handles  {ok}/{len(drawers)} drawers have a handle standing proud of the front  ->  {'PASS' if drawers and ok == len(drawers) else 'FAIL'}")
    return 0 if drawers and ok == len(drawers) else 1


if __name__ == "__main__":
    sys.exit(main())
