# Dresser, earlier Articraft, Qwen3.6 with thinking on

The same frozen prompt and reference photo, run on 2026-09-05 through the earlier Articraft codebase
(`mattzh72/articraft` at `959f1455`, the record's `articraft_commit`) against the same local
`RedHatAI/Qwen3.6-35B-A3B-NVFP4`, thinking on. It finished on its own final response after 65 turns and
3019 s. `build.json` is the record's summary.

That harness wrote URDF with box and cylinder primitives rather than USDZ, so `score.txt` comes from
`spark/render/urdf2obj.py model.urdf`, which applies the same two checks as `spark/score.py`: all nine
slide axes horizontal (+Y), all nine drawers move out of the carcass. `sheet.png` is rendered with the
same cameras as every other sheet here; `hero.png` is a separate Blender render of the same URDF, and
`dresser.gif` / `dresser.mp4` are `spark/render/animate.py` driving its nine prismatic joints.

A second draw of the same configuration ran to the 100-turn limit after 2 h 31 min without producing
a record.
