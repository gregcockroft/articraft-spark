#!/usr/bin/env bash
# Render a run's final USDZ: a four-view sheet with drawers closed and open, and optionally an MP4.
#
#   spark/render/render.sh <run-dir | file.usdz> [--video]
#
# Output goes to <run-dir>/render/ (or beside the .usdz); set RENDER_OUT to put it elsewhere.
# Needs Blender on PATH (or BLENDER=/path/to/blender) and the articraft env's python (for pxr and PIL).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ARTICRAFT_PYTHON:-${ARTICRAFT_VENV:-$(cd "$(dirname "$0")/../.." && pwd)/.venv}/bin/python}
BLENDER=${BLENDER:-$(command -v blender || true)}
[ -x "$BLENDER" ] || { echo "Blender not found: set BLENDER=/path/to/blender" >&2; exit 2; }
TARGET=${1:?usage: render.sh <run-dir | file.usdz> [--video]}
if [[ $TARGET == *.usdz ]]; then USDZ=$TARGET; OUT=${RENDER_OUT:-$(dirname "$TARGET")/render}
else USDZ=$TARGET/$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['result'])" "$TARGET/record.json"); OUT=${RENDER_OUT:-$TARGET/render}; fi
mkdir -p "$OUT"
for state in 0 1; do
  "$PY" "$HERE/usdz2obj.py" "$USDZ" "$OUT/state$state.obj" --open $state
  "$BLENDER" -b --python-exit-code 1 --python "$HERE/review.py" -- "$OUT/state$state.obj" "$OUT/state$state" >/dev/null 2>&1
done
"$PY" - "$OUT" <<'PYEOF'
import sys
from PIL import Image, ImageDraw
out = sys.argv[1]; W = 420
sheet = Image.new("RGB", (W * 4, 2 * (W + 26)), "white"); d = ImageDraw.Draw(sheet)
for r, label in enumerate(("closed", "open (every prismatic joint at its upper limit)")):
    y = r * (W + 26); d.rectangle([0, y, W * 4, y + 26], fill=(40, 40, 48)); d.text((8, y + 7), label, fill="white")
    for c, view in enumerate(("34", "front", "side", "top")):
        sheet.paste(Image.open(f"{out}/state{r}_{view}.png").convert("RGB"), (c * W, y + 26))
sheet.save(f"{out}/sheet.png"); print(f"{out}/sheet.png")
PYEOF
if [ "${2:-}" = --video ]; then
  rm -rf "$OUT/frames"; mkdir -p "$OUT/frames"
  "$BLENDER" -b --python-exit-code 1 --python "$HERE/animate.py" -- "$OUT/state0.obj" "$OUT/frames" 10 >/dev/null 2>&1
  ENCODE=(ffmpeg -y -loglevel error -framerate 30 -i "$OUT/frames/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT/demo.mp4")
  if command -v ffmpeg >/dev/null; then "${ENCODE[@]}" && echo "$OUT/demo.mp4"
  else echo "frames in $OUT/frames/ (no ffmpeg on PATH; to encode: ${ENCODE[*]})"; fi
fi
