#!/usr/bin/env bash
# Clone this fork fresh into a subdirectory and run the dresser demo against a local model, using a
# Hugging Face cache you already have (no re-download). Standalone on purpose - it doesn't need to
# run from inside a checkout of this repo, so copy it out or fetch it directly:
#
#   curl -O https://raw.githubusercontent.com/gregcockroft/articraft-spark/main/spark/clone_and_demo.sh
#   chmod +x clone_and_demo.sh
#
# Usage, from an empty folder:
#
#   ./clone_and_demo.sh <model-key> <hf-cache-dir> [demo|frozen]
#
# <model-key>    one of: qwen3.6-35b-a3b-nvfp4 (default choice below), qwen3.8-27b-inferact,
#                qwen3.8-flash-next-nvfp4, qwen3.8-27b (unpinned — will hit the network regardless
#                of the cache, since it always resolves `main` at the hub)
# <hf-cache-dir> the HF_CACHE folder that already has this model's weights, e.g. your existing
#                ~/.cache/huggingface — passed straight through to spark/cache.sh and spark/serve.sh
# [demo|frozen]  which dresser prompt to send (default: demo, the one-paragraph description)
#
# Requires: git, docker (with GPU support: --gpus all), an internet connection for the clone
# itself (a few MB) but NOT for the weights if hf-cache-dir already has them — spark/cache.sh
# checks that up front and refuses to serve.sh if anything would be fetched.
#
# What it does:
#   1. git clone the fork into ./articraft-spark
#   2. spark/install.sh            (uv venv + OpenUSD; set FRESH_USD=1 to force a from-source
#                                   OpenUSD build under this folder instead of reusing any build
#                                   already on the box, so the build itself is actually exercised)
#   3. spark/cache.sh <model-key>  (report only; aborts here if HF_CACHE is missing the weights)
#   4. spark/serve.sh <model-key>  (vLLM in Docker, offline, bind-mounts your HF_CACHE) — skipped
#                                   if :8001 (or $ARTICRAFT_SERVE_PORT) already answers as this
#                                   model's SERVE_NAME, and skipped with guidance printed if the
#                                   model's .env documents that spark/serve.sh does not serve it
#                                   (e.g. qwen3.8-flash-next-nvfp4 needs its own server/image)
#   5. spark/demo_dresser.sh <model-key>   (renders render/sheet.png and render/turntable.gif too,
#                                           if Blender is found — see below)
#
# Expect 30-120 minutes for step 5 on a DGX-Spark-class box; the agent works in up to
# ARTICRAFT_MAX_TURNS turns (override that env var before running this script to shorten it).
# A vLLM container started by this script (step 4) is left running afterward — `docker rm -f
# articraft-serve` when you're done. A server this script found already running (skipped step 4)
# is left exactly as it was.
#
# Rendering: spark/demo_dresser.sh renders automatically if `blender` is on PATH or $BLENDER is
# set. If you already export BLENDER, this script leaves it alone. Otherwise it looks in a couple
# of places a Blender build for this kind of box commonly lands (nothing is installed or fetched -
# these are existence checks only) before falling back to no render, exactly as demo_dresser.sh
# would on its own.
set -euo pipefail

REPO_URL=${ARTICRAFT_SPARK_REPO:-https://github.com/gregcockroft/articraft-spark.git}
MODEL_KEY=${1:?usage: $0 <model-key> <hf-cache-dir> [demo|frozen]}
RAW_CACHE=${2:?usage: $0 <model-key> <hf-cache-dir> [demo|frozen]}
PROMPT_MODE=${3:-demo}
CLONE_DIR=articraft-spark
PORT=${ARTICRAFT_SERVE_PORT:-8001}

[ -d "$CLONE_DIR" ] && { echo "refusing: ./$CLONE_DIR already exists" >&2; exit 1; }
[ -d "$RAW_CACHE" ] || { echo "no such HF cache dir: $RAW_CACHE" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
HF_CACHE=$(cd "$RAW_CACHE" && pwd)   # serve.sh bind-mounts this into the container; must be absolute

if [ -z "${BLENDER:-}" ] && ! command -v blender >/dev/null; then
  for candidate in "$HOME/tools/blender-gb10/blender" "$HOME/blender/blender"; do
    [ -x "$candidate" ] && { BLENDER=$candidate; break; }
  done
fi
if [ -n "${BLENDER:-}" ]; then
  export BLENDER
  echo "==> rendering with BLENDER=$BLENDER"
else
  echo "==> no Blender found (checked PATH and the usual local spots); demo_dresser.sh will skip the render"
fi

echo "==> cloning $REPO_URL into ./$CLONE_DIR"
git clone "$REPO_URL" "$CLONE_DIR"
cd "$CLONE_DIR"

echo "==> spark/install.sh"
if [ "${FRESH_USD:-0}" = 1 ]; then
  USD_PREFIX=$(pwd)/usd-prefix
  echo "FRESH_USD=1: building OpenUSD from source under $USD_PREFIX, not reusing any prior build"
  env -u FORCE_COLOR USD_PREFIX="$USD_PREFIX" spark/install.sh
else
  env -u FORCE_COLOR spark/install.sh
fi

echo "==> spark/cache.sh $MODEL_KEY   (checking $HF_CACHE, nothing fetched)"
if ! HF_CACHE="$HF_CACHE" spark/cache.sh "$MODEL_KEY"; then
  echo
  echo "cache.sh says $HF_CACHE is missing weights $MODEL_KEY needs (see above)." >&2
  echo "Point arg 2 at a cache that has them, or run 'spark/cache.sh $MODEL_KEY --fetch' yourself first." >&2
  exit 2
fi

SERVE_NAME=$( (set -a; . "spark/models/$MODEL_KEY.env"; echo "$SERVE_NAME") )
if curl -sf -m 3 "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "\"id\":\"$SERVE_NAME\""; then
  echo "==> :$PORT already serving '$SERVE_NAME' — reusing it, not calling spark/serve.sh"
else
  echo "==> spark/serve.sh $MODEL_KEY   (vLLM in Docker, offline, from $HF_CACHE)"
  if ! HF_CACHE="$HF_CACHE" spark/serve.sh "$MODEL_KEY"; then
    rc=$?
    echo
    echo "spark/serve.sh exited $rc for $MODEL_KEY — some model files document that this script" >&2
    echo "does not serve them (see spark/models/$MODEL_KEY.env). Start that model's own server," >&2
    echo "then re-run this script; it will detect it answering on :$PORT and skip straight to the demo." >&2
    exit "$rc"
  fi
fi

echo "==> spark/demo_dresser.sh $MODEL_KEY   (PROMPT=$PROMPT_MODE)"
PROMPT=$PROMPT_MODE spark/demo_dresser.sh "$MODEL_KEY"
