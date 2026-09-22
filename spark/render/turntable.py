"""Blender: RGBA frames of an object swinging in a studio while its drawers ripple open, one by one.

    blender -b --python-exit-code 1 --python turntable.py -- model.obj frames_dir [frames]

This is the picture the README leads with. It was made by hand on another machine until now, from a
studio .blend that lived in no repository; the studio is rebuilt here in code from the values that
blend carried - one area light, a flat grey world, a 47 mm camera, AgX - so a clone can make the
same picture with nothing but this file.

Reads <model>.joints.json (from usdz2obj.py) for each prismatic body's world axis and upper limit,
and <model>.mtl for the appearances. The frames are RGBA over a transparent film: gif.py lays the
background in, the same way the published GIFs were made.

Env: SAMPLES (24), RES (1080), CYCLES_DEVICE (GPU, or CPU to force it).
"""

import json
import math
import os
import sys
from pathlib import Path

import bpy
import mathutils

args = sys.argv[sys.argv.index("--") + 1:]
obj, out = args[0], args[1]
n = int(args[2]) if len(args) > 2 else 150
samples, res = int(os.environ.get("SAMPLES", "24")), int(os.environ.get("RES", "1080"))
joints = json.loads(Path(obj.rsplit(".", 1)[0] + ".joints.json").read_text())["prismatic"]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=obj, forward_axis="Y", up_axis="Z")
scene = bpy.context.scene
parts = [o for o in scene.objects if o.type == "MESH"]
if not parts:
    raise SystemExit(f"turntable: no meshes in {obj}")


def bounds(objects, offsets=()):
    lo, hi = mathutils.Vector((1e9,) * 3), mathutils.Vector((-1e9,) * 3)
    for o in objects:
        for shift in (mathutils.Vector((0, 0, 0)), *offsets):
            for c in o.bound_box:
                w = (o.matrix_world @ mathutils.Vector(c)) + shift
                lo, hi = mathutils.Vector(map(min, lo, w)), mathutils.Vector(map(max, hi, w))
    return lo, hi


# Each prismatic body rides a parent empty that slides along its own world axis; the empties all hang
# off one pivot, which is what swings. The drawers open in a snake - top row left to right, the next
# row back the other way - which reads as one movement rather than nine.
rigs = {}
for j in joints:
    rig = bpy.data.objects.new("slide_" + j["body"], None)
    scene.collection.objects.link(rig)
    rig["axis"], rig["travel"] = list(j["axis"]), j["upper"] * 0.8   # 0.8: out, but not off its rails
    for o in parts:
        if o.name.split("__")[0] == j["body"]:
            o.parent = rig
    rigs[j["body"]] = rig
travel_max = max((r["travel"] for r in rigs.values()), default=0.0)
lo, hi = bounds(parts, [mathutils.Vector(r["axis"]) * r["travel"] for r in rigs.values()])
centre, height = (lo + hi) / 2, hi.z - lo.z
radius = max(math.hypot(x, y) for x in (lo.x, hi.x) for y in (lo.y, hi.y))

pivot = bpy.data.objects.new("pivot", None)
pivot.location = (centre.x, centre.y, lo.z)
scene.collection.objects.link(pivot)
for o in list(rigs.values()) + [o for o in parts if o.parent is None]:
    o.parent = pivot
    o.matrix_parent_inverse = pivot.matrix_world.inverted()


def centre_of(rig):
    kids = [o for o in parts if o.parent is rig]
    pts = [o.matrix_world @ mathutils.Vector(c) for o in kids for c in o.bound_box]
    return sum(pts, mathutils.Vector()) / len(pts)


order = []
if rigs:
    at = {b: centre_of(r) for b, r in rigs.items()}
    rows, row = [], []
    for body in sorted(rigs, key=lambda b: -at[b].z):
        if row and abs(at[row[-1]].z - at[body].z) > 0.02 * max(height, 0.01):
            rows.append(row)
            row = []
        row.append(body)
    rows.append(row)
    for i, r in enumerate(rows):
        order += sorted(r, key=lambda b: at[b].x, reverse=bool(i % 2))
print("TURNTABLE snake", order)

# The timing the published GIFs carry, written as fractions of the frame count so any length works:
# each drawer starts 0.0733 n after the one before, opens over 0.0667 n, holds 0.08 n, closes; then
# they all open together at 0.787 n and shut by the end.
spacing = min(0.0733 * n, (0.60 * n - 0.0667 * n) / max(1, len(order) - 1))


def key(rig, frame, amount):
    rig.location = mathutils.Vector(rig["axis"]) * (rig["travel"] * amount)
    rig.keyframe_insert("location", frame=int(frame))


for i, body in enumerate(order):
    rig, t0 = rigs[body], 0.0667 * n + i * spacing
    for frame, amount in ((1, 0), (t0, 0), (t0 + 0.0667 * n, 1), (t0 + 0.147 * n, 1), (t0 + 0.213 * n, 0)):
        key(rig, min(frame, n), amount)
for rig in rigs.values():
    for frame, amount in ((0.787 * n, 0), (0.867 * n, 1), (0.933 * n, 1), (n, 0)):
        key(rig, frame, amount)

for frame, deg in ((1, -35), (n // 2, 35), (n, -35)):
    pivot.rotation_euler = (0, 0, math.radians(deg))
    pivot.keyframe_insert("rotation_euler", index=2, frame=frame)

# The studio: one area light, a flat grey world at 1.175, a 47 mm camera, AgX - the values the
# hand-built scene these GIFs were shot in carried, so a clone reproduces the same picture. The light is placed and powered
# for a 1.05 m dresser, so it scales with the object - distance by s, energy by s squared.
s = max(height, 2 * radius) / 1.05
world = bpy.data.worlds.new("studio")
scene.world = world
world.node_tree.nodes["Background"].inputs[0].default_value = (0.28, 0.29, 0.31, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.175
light = bpy.data.lights.new("studio_key", "AREA")
light.energy, light.size = 105.0 * s * s, 1.1 * s
key_light = bpy.data.objects.new("studio_key", light)
key_light.location = centre + mathutils.Vector((-0.5, -1.0, 1.52)) * s
scene.collection.objects.link(key_light)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens = 47
scene.collection.objects.link(cam)
scene.camera = cam
target = mathutils.Vector((centre.x, centre.y, lo.z + 0.43 * height))
half = math.atan(18.0 / cam.data.lens)                 # 36 mm sensor, square frame
# 1.51 is not a taste: it is the number that puts the object on as much of the frame as the
# published GIFs do - coverage 0.230 of the frame against their 0.230, measured at the same phase.
fit = max(height, 2 * radius, 2 * travel_max) / 2 / math.tan(half) * 1.51
elevation = math.radians(19)
cam.location = target + mathutils.Vector((0, math.cos(elevation), math.sin(elevation))) * fit
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()

scene.render.engine = "CYCLES"
scene.cycles.samples, scene.cycles.use_denoising = samples, True
device = "CPU"
if os.environ.get("CYCLES_DEVICE", "GPU").upper() != "CPU":
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except TypeError:
            continue
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device, device = "GPU", kind
            break
scene.view_settings.view_transform = "AgX"
scene.render.resolution_x = scene.render.resolution_y = res
scene.render.film_transparent = True          # the background is laid in by the encoder
scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGBA"
scene.render.fps, scene.frame_start, scene.frame_end = 30, 1, n
Path(out).mkdir(parents=True, exist_ok=True)
scene.render.filepath = out.rstrip("/") + "/frame_####"
if os.environ.get("ONEFRAME"):                # a single frame, for checking the framing cheaply
    scene.frame_start = scene.frame_end = int(os.environ["ONEFRAME"])
print(f"TURNTABLE device={device} samples={samples} res={res} "
      f"frames={scene.frame_start}-{scene.frame_end} drawers={len(order)} scale={s:.2f}")
bpy.ops.render.render(animation=True)
print("TURNTABLE_OK")
