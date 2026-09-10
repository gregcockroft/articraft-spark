"""Flatten a URDF built from box and cylinder primitives into one world-space OBJ, like usdz2obj.py.

Earlier Articraft versions emitted URDF rather than USDZ. Objects are named <link>__<n>, where <link>
is the prismatic child it rides on ("root" otherwise); <out>.joints.json lists each prismatic child's
world axis and upper limit. It also prints the same two checks spark/score.py applies to a USDZ.

    python urdf2obj.py model.urdf out.obj [--open 1.0]
"""

import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
frac = float(sys.argv[sys.argv.index("--open") + 1]) if "--open" in sys.argv else 0.0
root = ET.parse(src).getroot()


def origin(el):
    o = el.find("origin") if el is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    r, p, y = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    m = np.eye(4)
    m[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    m[:3, 3] = xyz
    return m


joints = {j.find("child").get("link"): j for j in root.findall("joint")}
links = {link.get("name"): link for link in root.findall("link")}


def world(name):
    if name not in joints:
        return np.eye(4)
    j = joints[name]
    return world(j.find("parent").get("link")) @ origin(j)


def rider(name):  # the prismatic joint a link moves with, if any
    while name in joints:
        if joints[name].get("type") == "prismatic":
            return name
        name = joints[name].find("parent").get("link")
    return None


slides = {}
for child, j in joints.items():
    if j.get("type") != "prismatic":
        continue
    axis = np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0").split()])
    axis = world(j.find("parent").get("link"))[:3, :3] @ (origin(j)[:3, :3] @ axis)
    axis /= np.linalg.norm(axis)
    limit = j.find("limit")
    slides[child] = {"name": j.get("name"), "body": child, "parent": j.find("parent").get("link"),
                     "axis": axis.tolist(), "upper": float(limit.get("upper", 0)) if limit is not None else 0.0}

BOX_V = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
BOX_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
points = {}  # link -> world points, for the checks
with open(dst, "w") as out:
    base = 1
    for name, link in links.items():
        owner = rider(name)
        shift = np.array(slides[owner]["axis"]) * slides[owner]["upper"] * frac if owner else np.zeros(3)
        for k, vis in enumerate(link.findall("visual")):
            g = vis.find("geometry")
            if g.find("box") is not None:
                v = BOX_V * np.array([float(s) for s in g.find("box").get("size").split()]) / 2
                faces = BOX_F
            elif g.find("cylinder") is not None:
                r, h, n = float(g.find("cylinder").get("radius")), float(g.find("cylinder").get("length")), 24
                ring = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
                v = np.array([(x, y, -h / 2) for x, y in ring] + [(x, y, h / 2) for x, y in ring])
                faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
                faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
            else:
                continue
            m = world(name) @ origin(vis)
            w = (m[:3, :3] @ v.T).T + m[:3, 3] + shift
            points.setdefault(owner or "root", []).append(w)
            out.write(f"o {owner or 'root'}__{name}_{k}\n")
            out.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in w)
            out.writelines("f " + " ".join(str(base + i) for i in f) + "\n" for f in faces)
            base += len(w)
with open(dst.rsplit(".", 1)[0] + ".joints.json", "w") as f:
    json.dump({"up": "Z", "prismatic": list(slides.values())}, f, indent=1)

if frac == 0.0 and slides:
    centre = {k: (np.vstack(v).min(0) + np.vstack(v).max(0)) / 2 for k, v in points.items()}
    flat = out_ok = 0
    for s in slides.values():
        a = np.array(s["axis"])
        before = centre[s["body"]] - centre["root"]
        after = before + a * s["upper"]
        f_ok, o_ok = abs(a[2]) < 0.1, np.linalg.norm(after[:2]) > np.linalg.norm(before[:2]) + 1e-6
        flat += f_ok
        out_ok += o_ok
        print(f"  {s['name']:22s} world axis ({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f})"
              f"  {'horizontal' if f_ok else 'NOT horizontal'}  {'moves out' if o_ok else 'does NOT move out'}")
    print(f"axis     {flat}/{len(slides)} prismatic axes horizontal  ->  {'PASS' if flat == len(slides) else 'FAIL'}")
    print(f"out      {out_ok}/{len(slides)} move away from the carcass  ->  {'PASS' if out_ok == len(slides) else 'FAIL'}")
