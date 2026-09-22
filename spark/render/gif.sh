#!/usr/bin/env bash
# Render a run's final USDZ as the README's turntable GIF: the object swinging in a studio while its
# drawers ripple open. Everything happens on this machine - no second box, no ffmpeg, no .blend file.
#
#   spark/render/gif.sh <run-dir | file.usdz> [out.gif]
#
# Output goes to <run-dir>/render/turntable.gif (or beside the .usdz); set RENDER_OUT to put the
# working files elsewhere. Needs Blender (BLENDER=/path/to/blender, or on PATH) and the project venv's
# python for pxr and PIL. FRAMES=150 SAMPLES=24 RES=1080 tune the render; the GIF is always 480x480
# and 60 frames, because that is the format the two published GIFs carry.
#
# Exit codes:
#   0   the GIF was written
#   2   it could not start: no Blender, no python, no USDZ, no run record
#   3   Blender rendered no frames, or the encode failed
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ARTICRAFT_PYTHON:-${ARTICRAFT_VENV:-$(cd "$HERE/../.." && pwd)/.venv}/bin/python}
BLENDER=${BLENDER:-$(command -v blender || true)}
FRAMES=${FRAMES:-150}
[ -x "$BLENDER" ] || { echo "Blender not found: set BLENDER=/path/to/blender" >&2; exit 2; }
[ -x "$PY" ] || { echo "no python at $PY: run spark/install.sh, or set ARTICRAFT_VENV=/path/to/venv" >&2; exit 2; }
TARGET=${1:?usage: gif.sh <run-dir | file.usdz> [out.gif]}
if [[ $TARGET == *.usdz ]]; then
  USDZ=$TARGET; OUT=${RENDER_OUT:-$(dirname "$TARGET")/render}
else
  [ -f "$TARGET/record.json" ] || { echo "no record.json in $TARGET" >&2; exit 2; }
  USDZ=$TARGET/$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['result'])" "$TARGET/record.json")
  OUT=${RENDER_OUT:-$TARGET/render}
fi
[ -f "$USDZ" ] || { echo "no USDZ at $USDZ" >&2; exit 2; }
GIF=${2:-$OUT/turntable.gif}
mkdir -p "$OUT"

# The USDZ is flattened to OBJ+MTL first: the Blender that runs on a Spark has no USD importer, and
# without the MTL every part renders the same grey.
"$PY" "$HERE/usdz2obj.py" "$USDZ" "$OUT/turntable.obj" || exit 3
rm -rf "$OUT/frames"
"$BLENDER" -b --python-exit-code 1 --python "$HERE/turntable.py" -- \
  "$OUT/turntable.obj" "$OUT/frames" "$FRAMES" > "$OUT/turntable.log" 2>&1
rc=$?
n=$(ls "$OUT/frames"/frame_*.png 2>/dev/null | wc -l)
if [ "$rc" != 0 ] || [ "$n" -lt 60 ]; then
  echo "turntable render failed (exit $rc, $n frames): see $OUT/turntable.log" >&2
  tail -3 "$OUT/turntable.log" >&2
  exit 3
fi
"$PY" "$HERE/gif.py" "$OUT/frames" "$GIF" || exit 3
