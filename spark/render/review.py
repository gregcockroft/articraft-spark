"""Blender: render an OBJ from four fixed cameras (3/4, front, side, top) with neutral shading.

    blender -b --python review.py -- model.obj out_prefix
"""

import sys

import bpy
import mathutils

obj, out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=obj, forward_axis="Y", up_axis="Z")
parts = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in parts:
    body = o.name.split("__")[0]
    o.color = (0.80, 0.72, 0.58, 1) if body == "root" else ((0.60, 0.46, 0.33, 1), (0.66, 0.52, 0.38, 1))[hash(body) % 2]
lo = mathutils.Vector((1e9,) * 3)
hi = -lo
for o in parts:
    for c in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(c)
        lo, hi = mathutils.Vector(map(min, lo, w)), mathutils.Vector(map(max, hi, w))
centre, size = (lo + hi) / 2, (hi - lo).length
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
shading = scene.display.shading
shading.light, shading.color_type = "STUDIO", "OBJECT"
shading.show_cavity = shading.show_object_outline = shading.show_shadows = True
shading.shadow_intensity = 0.35
scene.render.resolution_x = scene.render.resolution_y = 420
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
for tag, direction, ortho in (("34", (0.55, 1, 0.5), False), ("front", (0, 1, 0.08), True),
                              ("side", (1, 0.08, 0.08), True), ("top", (0.02, 0.2, 1), True)):
    cam.location = centre + mathutils.Vector(direction).normalized() * size * 1.8
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO" if ortho else "PERSP"
    cam.data.ortho_scale = size * 1.05
    scene.render.filepath = f"{out}_{tag}.png"
    bpy.ops.render.render(write_still=True)
