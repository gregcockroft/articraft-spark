"""Flatten an Articraft USDZ into one world-space OBJ that Blender can import without USD support.

Every Mesh and Cube prim becomes an OBJ object named <body>__<shape>, where <body> is the prismatic
child it rides on ("root" for everything else). A sidecar <out>.joints.json lists each prismatic
body's world axis and upper limit, so an animation can open the drawers, and a sidecar <out>.mtl
carries each bound UsdPreviewSurface's diffuse colour, metallic and roughness - without it a shaded
render loses the paint, the wood and the brass and every part comes out the same grey.

    python usdz2obj.py model.usdz out.obj [--open 1.0]

--open F moves every prismatic child F of the way to its upper limit before writing.
"""

import json
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

src, dst = sys.argv[1], sys.argv[2]
frac = float(sys.argv[sys.argv.index("--open") + 1]) if "--open" in sys.argv else 0.0
stage = Usd.Stage.Open(src)
t = Usd.TimeCode.Default()
AXES = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}


def world(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t)


joints = {}
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.PrismaticJoint):
        continue
    j = UsdPhysics.PrismaticJoint(prim)
    b0 = stage.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
    b1 = str(j.GetBody1Rel().GetTargets()[0])
    rot0 = Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get() or Gf.Quatf(1)))
    axis = world(b0).ExtractRotation().TransformDir(rot0.TransformDir(AXES[j.GetAxisAttr().Get()]))
    joints[b1] = {"name": prim.GetName(), "body": b1.split("/")[-1],
                  "axis": list(axis), "upper": float(j.GetUpperLimitAttr().Get() or 0.0)}

# The bound UsdPreviewSurface per prim, written as an .mtl beside the OBJ. A prim with no bound
# material, or one whose surface has no diffuseColor, falls back to "default" rather than to nothing:
# an OBJ that names a material the .mtl lacks imports with no material at all.
DEFAULT = ("default", (0.8, 0.8, 0.8), 0.0, 0.5)
materials = {}


def material_of(prim):
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not mat:
        return DEFAULT[0]
    surface = UsdShade.Material(mat).ComputeSurfaceSource()[0]
    if not surface:
        return DEFAULT[0]
    name = mat.GetPath().name

    def value(key, fallback):
        got = surface.GetInput(key)
        got = got.Get() if got else None
        return fallback if got is None else got

    colour = value("diffuseColor", None)
    if colour is None:
        return DEFAULT[0]
    materials[name] = (name, tuple(colour), float(value("metallic", 0.0)), float(value("roughness", 0.5)))
    return name


CUBE_V = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
CUBE_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
up = UsdGeom.GetStageUpAxis(stage)
with open(dst, "w") as out:
    out.write(f"mtllib {dst.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.mtl\n")
    base = 1
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts, counts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        elif prim.IsA(UsdGeom.Cube):
            h = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            pts, counts, idx = [Gf.Vec3d(*(c * h for c in v)) for v in CUBE_V], [4] * 6, [i for f in CUBE_F for i in f]
        else:
            continue
        if not pts:
            continue
        path = str(prim.GetPath())
        owner = next((b for b in joints if path.startswith(b + "/")), None)
        shift = Gf.Vec3d(*joints[owner]["axis"]) * joints[owner]["upper"] * frac if owner else Gf.Vec3d(0)
        xf = world(prim)
        out.write(f"o {joints[owner]['body'] if owner else 'root'}__{prim.GetName()}\n")
        out.write(f"usemtl {material_of(prim)}\n")
        for p in pts:
            w = xf.Transform(Gf.Vec3d(p)) + shift
            if up == "Y":  # Blender is Z-up
                w = Gf.Vec3d(w[0], -w[2], w[1])
            out.write(f"v {w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n")
        k = 0
        for c in counts:
            out.write("f " + " ".join(str(base + idx[k + i]) for i in range(c)) + "\n")
            k += c
        base += len(pts)
Path(dst.rsplit(".", 1)[0] + ".mtl").write_text("".join(
    f"newmtl {name}\nKd {colour[0]:.6f} {colour[1]:.6f} {colour[2]:.6f}\n"
    f"Pm {metallic:.4f}\nPr {roughness:.4f}\nillum 2\n\n"
    for name, colour, metallic, roughness in [DEFAULT, *materials.values()]))
with open(dst.rsplit(".", 1)[0] + ".joints.json", "w") as f:
    json.dump({"up": up, "prismatic": list(joints.values())}, f, indent=1)
