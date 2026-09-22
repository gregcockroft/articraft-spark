#!/usr/bin/env bash
# The dresser demo: a local model on this machine builds an articulated nine-drawer chest from one
# reference photo, then the result is scored and rendered. No API key, no network after the weights.
#
#   spark/serve.sh qwen3.6-35b-a3b-nvfp4      # once; leaves the server running
#   spark/demo_dresser.sh qwen3.6-35b-a3b-nvfp4
#
# PROMPT=demo (default) sends bench/dresser/prompt_demo.txt, the one-paragraph description the
# result in spark/results/ was built from. PROMPT=frozen sends the short benchmark prompt.
#
# Output: runs/spark-demo/<run-id>/ with the USDZ, the conversation, score.txt, render/sheet.png and
# render/turntable.gif - the shaded turntable the README leads with. GIF=0 skips the GIF (it is a few
# minutes of Cycles on top of an hour-long run); everything else is unchanged by it.
# Expect 30-120 minutes on a DGX Spark; the agent works in up to ARTICRAFT_MAX_TURNS turns.
#
# Exit codes, which used to be unstated - and a thing nobody states is a thing nobody notices breaking:
#   0   the object was built and scored, and the render (if Blender was found) wrote its sheet
#   2   this script could not start: no model file, no articraft, no server on the port, bad PROMPT
#   3   the frozen dresser inputs do not match spark/bench/dresser/SHA256SUMS
#   4   the object scored PASS but THE RENDER FAILED, so there is no sheet to look at
#   5   the object scored PASS and the sheet is there, but THE GIF FAILED
#   other   spark/score.py's verdict on the object
# A failed render always prints a line on stderr, including when a non-zero score keeps the exit code:
# the code can only carry one of the two facts, and the line is what makes the other one visible.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
KEY=${1:-qwen3.6-35b-a3b-nvfp4}
ENV_FILE=$HERE/models/$KEY.env
[ -f "$ENV_FILE" ] || { echo "no model file $ENV_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

PREFIX=${ARTICRAFT_VENV:-$(cd "$(dirname "$0")/.." && pwd)/.venv}
ART=$PREFIX/bin/articraft
PY=$PREFIX/bin/python
PORT=${ARTICRAFT_SERVE_PORT:-8001}
BENCH=$HERE/bench/dresser
case ${PROMPT:-demo} in
  demo) PROMPT_FILE=$BENCH/prompt_demo.txt ;;
  frozen) PROMPT_FILE=$BENCH/prompt.txt ;;
  *) echo "PROMPT must be demo or frozen" >&2; exit 2 ;;
esac
OUT=${OUT:-$ROOT/runs/spark-demo}
[ -x "$ART" ] || { echo "articraft not installed at $ART; run spark/install.sh" >&2; exit 2; }
curl -sf -m 5 "localhost:$PORT/v1/models" >/dev/null || { echo "no server on :$PORT; run spark/serve.sh $KEY" >&2; exit 2; }
(cd "$BENCH" && sha256sum --quiet -c SHA256SUMS) || { echo "the frozen dresser inputs do not match SHA256SUMS" >&2; exit 3; }

export ARTICRAFT_PROVIDER=openrouter
export ARTICRAFT_OPENROUTER_BASE_URL=http://localhost:$PORT/v1
export ARTICRAFT_OPENROUTER_MODEL=$SERVE_NAME
export ARTICRAFT_OPENROUTER_REQUEST_TIMEOUT_SECONDS=${ARTICRAFT_OPENROUTER_REQUEST_TIMEOUT_SECONDS:-3600}
export ARTICRAFT_MAX_TURNS=${ARTICRAFT_MAX_TURNS:-100}
# An empty CHAT_TEMPLATE_KWARGS must not be sent as an empty string.
[ -n "${ARTICRAFT_OPENROUTER_CHAT_TEMPLATE_KWARGS:-}" ] || unset ARTICRAFT_OPENROUTER_CHAT_TEMPLATE_KWARGS
unset FORCE_COLOR CLICOLOR_FORCE COLORTERM

# The first generation after a server start decodes at about a third of warm speed for ~50 s.
curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVE_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say ready.\"}],\"max_tokens\":16}" -o /dev/null

echo "[$(date +%H:%M:%S)] $KEY building the dresser from $(basename "$PROMPT_FILE") (max $ARTICRAFT_MAX_TURNS turns)"
start=$(date +%s)
"$ART" generate "$(cat "$PROMPT_FILE")" --image "$BENCH/reference.png" \
  --provider openrouter --model "$SERVE_NAME" --no-tui --output-dir "$OUT"
echo "[$(date +%H:%M:%S)] finished in $(( ($(date +%s) - start) / 60 )) min"

RUN=$(ls -dt "$OUT"/*/ | head -1)
"$PY" "$HERE/score.py" "$RUN" --ask "$BENCH/ask.json" | tee "$RUN/score.txt"
verdict=${PIPESTATUS[0]}
if [ -n "${BLENDER:-}" ] || command -v blender >/dev/null; then
  # The render's status is not decoration: this script's own docs promise a sheet, and it used to throw
  # the status away and exit with the scorer's verdict, so a failed render reported success.
  "$HERE/render/render.sh" "$RUN" ${VIDEO:+--video}
  render_rc=$?
  if [ "$render_rc" != 0 ]; then
    echo "RENDER FAILED (exit $render_rc): no sheet was written for $RUN." >&2
    if [ "$verdict" = 0 ]; then
      echo "  the object scored PASS; this script exits 4 to say the sheet is missing." >&2
      verdict=4
    else
      echo "  the object also failed scoring, so the exit code stays $verdict - the more important fact." >&2
    fi
  fi
  if [ "${GIF:-1}" != 0 ]; then
    # The GIF is what the README shows, so a missing one is reported like a missing sheet rather than
    # left to be noticed: it is the output somebody actually looks at.
    "$HERE/render/gif.sh" "$RUN"
    gif_rc=$?
    if [ "$gif_rc" != 0 ]; then
      echo "GIF FAILED (exit $gif_rc): no turntable.gif was written for $RUN." >&2
      if [ "$verdict" = 0 ]; then
        echo "  the object scored PASS and the sheet is there; this script exits 5 to say the GIF is missing." >&2
        verdict=5
      else
        echo "  the exit code stays $verdict - the more important fact." >&2
      fi
    fi
  fi
else
  echo "no Blender: skipping the render and the GIF (set BLENDER=/path/to/blender for render/sheet.png and render/turntable.gif)"
fi
exit "$verdict"
