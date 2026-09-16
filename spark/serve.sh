#!/usr/bin/env bash
# Serve a local model for Articraft: vLLM in Docker, OpenAI-compatible, on localhost.
#
#   spark/serve.sh qwen3.6-35b-a3b-nvfp4
#
# Every model setting comes from spark/models/<key>.env. The first start downloads the weights
# into the Hugging Face cache; set HF_HUB_OFFLINE=1 afterwards to serve without the network.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=${1:?usage: spark/serve.sh <model-key>   (one of: $(cd "$HERE/models" && ls *.env | sed 's/\.env$//' | tr '\n' ' '))}
ENV_FILE=$HERE/models/$KEY.env
[ -f "$ENV_FILE" ] || { echo "no model file $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"

PORT=${ARTICRAFT_SERVE_PORT:-8001}
NAME=${ARTICRAFT_SERVE_NAME:-articraft-serve}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
LOG_DIR=${ARTICRAFT_SERVE_LOGS:-$HERE/../runs/serve}
mkdir -p "$LOG_DIR" "$HF_CACHE"

[ "${MODEL_STATUS:-}" = tested ] || echo "note: $KEY is MODEL_STATUS=${MODEL_STATUS:-unset}; no result in spark/results/ was measured with it" >&2
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

# One model needs a different server entirely: Flash-Next is 135 GB of weights and stock vLLM cannot
# fit it on one 121 GB Spark. A model file that pins SERVE_EXTERNAL_* names the project that can, and
# spark/serve_external.sh drives it - a separate launcher, because it shares no argument with the vLLM
# one below. Same contract from the caller's side: it exits 0 when :$PORT answers.
if [ -n "${SERVE_EXTERNAL_REPO:-}" ]; then
  exec "$HERE/serve_external.sh" "$KEY"
fi

# A model file that pins neither a vLLM image nor an external server cannot be started at all - some
# ship only SERVE_MODEL/SERVE_REVISION/SERVE_NAME for spark/cache.sh and spark/demo_dresser.sh to
# read. Fail with that pointer instead of an unbound-variable crash from the vLLM arg list below.
for v in SERVE_IMAGE SERVE_MAX_MODEL_LEN SERVE_GPU_UTIL SERVE_MAX_SEQS SERVE_TOOL_PARSER; do
  if [ -z "${!v:-}" ]; then
    echo "$KEY.env sets no $v: this model is not served by spark/serve.sh." >&2
    if [ -n "${SERVE_EXTERNAL_REPO:-}" ]; then
      echo "  served by:  ${SERVE_EXTERNAL_REPO} at ${SERVE_EXTERNAL_COMMIT:-unpinned}" >&2
      echo "  base image: ${SERVE_EXTERNAL_BASE_IMAGE:-unpinned}" >&2
      echo "  built as:   ${SERVE_EXTERNAL_TAG:-unpinned}" >&2
    fi
    echo "Read the top of $ENV_FILE for how to start its server, then run spark/demo_dresser.sh $KEY once it answers on :\${ARTICRAFT_SERVE_PORT:-8001}." >&2
    exit 3
  fi
done

# Serve offline when the pinned snapshot is already here. The cache path is the one spark/cache.sh
# reports on, so the two agree by construction. An explicit HF_HUB_OFFLINE from the caller always wins,
# in both directions: a caller who asks for 0 on a cached model gets 0.
if [ -n "${HF_HUB_OFFLINE:-}" ]; then
  echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE (from the caller)" >&2
elif [ -z "${SERVE_REVISION:-}" ]; then
  # Unpinned: there is no snapshot path to test, because a serve resolves `main` at the hub.
  HF_HUB_OFFLINE=0
  echo "HF_HUB_OFFLINE=0: $KEY pins no SERVE_REVISION, so this serve resolves '$SERVE_MODEL' at the hub and fetches it" >&2
elif [ -d "$HF_CACHE/hub/models--${SERVE_MODEL//\//--}/snapshots/$SERVE_REVISION" ]; then
  HF_HUB_OFFLINE=1
  echo "HF_HUB_OFFLINE=1: $SERVE_MODEL at $SERVE_REVISION is in $HF_CACHE, so this serve needs no network" >&2
else
  HF_HUB_OFFLINE=0
  echo "HF_HUB_OFFLINE=0: no snapshot at $HF_CACHE/hub/models--${SERVE_MODEL//\//--}/snapshots/$SERVE_REVISION, so the weights will be downloaded. \`spark/cache.sh $KEY\` says how much" >&2
fi

args=("$SERVE_MODEL" --served-model-name "$SERVE_NAME"
      --max-model-len "$SERVE_MAX_MODEL_LEN" --gpu-memory-utilization "$SERVE_GPU_UTIL"
      --max-num-seqs "$SERVE_MAX_SEQS"
      --enable-auto-tool-choice --tool-call-parser "$SERVE_TOOL_PARSER")
[ -n "${SERVE_REVISION:-}" ] && args+=(--revision "$SERVE_REVISION")
[ -n "${SERVE_REASONING_PARSER:-}" ] && args+=(--reasoning-parser "$SERVE_REASONING_PARSER")
[ -n "${SERVE_SPECULATIVE:-}" ] && args+=(--speculative-config "$SERVE_SPECULATIVE")
[ -n "${SERVE_LIMIT_MM:-}" ] && args+=(--limit-mm-per-prompt "$SERVE_LIMIT_MM")
# shellcheck disable=SC2206
[ -n "${SERVE_EXTRA_ARGS:-}" ] && args+=($SERVE_EXTRA_ARGS)

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host -p "$PORT:8000" \
  -e HF_HUB_OFFLINE="$HF_HUB_OFFLINE" -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "$SERVE_IMAGE" "${args[@]}" >/dev/null
LOG=$LOG_DIR/$KEY.log
setsid nohup docker logs -f "$NAME" > "$LOG" 2>&1 < /dev/null & disown
echo "serving $SERVE_MODEL as '$SERVE_NAME' on :$PORT  (log: $LOG)"

printf 'waiting for the server'
for _ in $(seq 1 240); do
  if curl -sf -m 2 "localhost:$PORT/v1/models" >/dev/null; then echo " ready"; exit 0; fi
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo; echo "container exited:" >&2; tail -30 "$LOG" >&2; exit 1; }
  printf '.'; sleep 10
done
echo; echo "not ready after 40 min, see $LOG" >&2; exit 1
