"""Blender: PNG frames of the object turning while its drawers open and close, one after another.

    blender -b --python-exit-code 1 --python animate.py -- model.obj frames_dir [seconds]

Frames are written as PNG so this works on a Blender built without FFmpeg; render.sh assembles
them into an MP4 when ffmpeg is on PATH.

Reads <model>.joints.json written by usdz2obj.py for each drawer's world axis and upper limit.
"""

import json
import math
import sys

import bpy
import mathutils

args = sys.argv[sys.argv.index("--") + 1:]
obj, frames = args[0], args[1]
seconds = float(args[2]) if len(args) > 2 else 10.0
joints = {j["body"]: j for j in json.load(open(obj.rsplit(".", 1)[0] + ".joints.json"))["prismatic"]}
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=obj, forward_axis="Y", up_axis="Z")
scene = bpy.context.scene
fps = 30
scene.render.fps, scene.frame_start, scene.frame_end = fps, 1, int(seconds * fps)
parts = [o for o in scene.objects if o.type == "MESH"]
lo = mathutils.Vector((1e9,) * 3)
hi = -lo
for o in parts:
    for c in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(c)
        lo, hi = mathutils.Vector(map(min, lo, w)), mathutils.Vector(map(max, hi, w))
centre, size = (lo + hi) / 2, (hi - lo).length

# Each drawer body gets a parent empty that slides along its axis: out, hold, back, staggered.
order = sorted(joints)
n = scene.frame_end
for i, body in enumerate(order):
    j = joints[body]
    rig = bpy.data.objects.new(f"slide_{body}", None)
    scene.collection.objects.link(rig)
    for o in parts:
        if o.name.split("__")[0] == body:
            o.parent = rig
    axis = mathutils.Vector(j["axis"]) * j["upper"]
    start = int(n * 0.15 + i * n * 0.04)
    for frame, amount in ((start, 0), (start + fps // 2, 1), (int(n * 0.75), 1), (int(n * 0.75) + fps // 2, 0)):
        rig.location = axis * amount
        rig.keyframe_insert("location", frame=frame)

for o in parts:
    body = o.name.split("__")[0]
    o.color = (0.80, 0.72, 0.58, 1) if body == "root" else ((0.60, 0.46, 0.33, 1), (0.66, 0.52, 0.38, 1))[hash(body) % 2]

# Camera on a turntable: a quarter turn each way around the front, slightly above.
pivot = bpy.data.objects.new("pivot", None)
pivot.location = centre
scene.collection.objects.link(pivot)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
cam.parent = pivot
cam.location = mathutils.Vector((0, 1, 0.45)).normalized() * size * 2.0
cam.rotation_euler = (-cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
for frame, angle in ((1, -35), (n // 2, 35), (n, -35)):
    pivot.rotation_euler = (0, 0, math.radians(angle))
    pivot.keyframe_insert("rotation_euler", frame=frame)

scene.render.engine = "BLENDER_WORKBENCH"
shading = scene.display.shading
shading.light, shading.color_type = "STUDIO", "OBJECT"
shading.show_cavity = shading.show_object_outline = shading.show_shadows = True
shading.shadow_intensity = 0.35
scene.render.resolution_x, scene.render.resolution_y = 1280, 720
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = f"{frames}/frame_####"
bpy.ops.render.render(animation=True)
