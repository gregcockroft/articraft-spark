#!/usr/bin/env bash
# Serve a model whose server is not stock vLLM, by driving the project the model file pins.
#
#   spark/serve_external.sh qwen3.8-flash-next-nvfp4     # or: spark/serve.sh <same key>, which dispatches here
#
# One model needs this: Qwen3.8-Flash-Next is 135 GB of weights and stock vLLM cannot fit it on one
# 121 GB Spark. blazux/qwen3.8-Flash-DGX patches the official image so the n-gram table is mmapped
# from NVMe and ~76-78 GiB sit on the card. That project is Apache-2.0 and it is DRIVEN, never
# vendored and never patched: this script clones it at the commit the model file pins, builds its
# image if this box has none, and runs its scripts/serve.sh with our settings. If the clone is not
# exactly at that commit and clean, this script stops - it does not repair a checkout it does not own.
#
# Everything it passes comes from spark/models/<key>.env, so the container this leaves running is the
# one the published result was measured on. There is no fallback path: if the external server cannot
# be built or started, that is the failure, and spark/serve.sh does not quietly serve something else.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=${1:?usage: spark/serve_external.sh <model-key>}
ENV_FILE=$HERE/models/$KEY.env
[ -f "$ENV_FILE" ] || { echo "no model file $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"
[ -n "${SERVE_EXTERNAL_REPO:-}" ] || { echo "$KEY.env pins no SERVE_EXTERNAL_REPO: it is not served by an external project. Use spark/serve.sh $KEY." >&2; exit 2; }
for v in SERVE_EXTERNAL_COMMIT SERVE_EXTERNAL_TAG SERVE_EXTERNAL_BASE_IMAGE SERVE_MODEL SERVE_REVISION \
         SERVE_MAX_MODEL_LEN SERVE_GPU_UTIL SERVE_MAX_SEQS; do
  [ -n "${!v:-}" ] || { echo "$KEY.env sets no $v, and this launcher hardcodes nothing" >&2; exit 2; }
done

PORT=${ARTICRAFT_SERVE_PORT:-8001}
NAME=${ARTICRAFT_SERVE_NAME:-articraft-serve}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
LOG_DIR=${ARTICRAFT_SERVE_LOGS:-$HERE/../runs/serve}
DIR=${SERVE_EXTERNAL_DIR:-$HOME/src/$(basename "$SERVE_EXTERNAL_REPO" .git)}
mkdir -p "$LOG_DIR" "$HF_CACHE"
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
command -v git >/dev/null || { echo "git is required, to clone $SERVE_EXTERNAL_REPO at its pin" >&2; exit 2; }

# A backgrounded follower outlives this script, so it must not inherit the caller's file descriptors:
# `flock <lock> spark/serve.sh <key>` otherwise leaves the lock HELD FOR THE CONTAINER'S LIFE by a
# ppid-1 process that `fuser` reports as "docker", which reads like the daemon and is not. Close every
# fd above 2 in the subshell the tail is started from.
closefds() { local fd n; for fd in /proc/$BASHPID/fd/*; do n=${fd##*/}; [ "$n" -gt 2 ] && eval "exec $n>&-"; done 2>/dev/null; return 0; }

# --- the weights ----------------------------------------------------------------------------------
# Their scripts/serve.sh bind-mounts the cache and runs with HF_HUB_OFFLINE=1, so a missing snapshot
# surfaces as "checkpoint not found" from inside their script. Say it here instead, with the size.
SNAP=$HF_CACHE/hub/models--${SERVE_MODEL//\//--}/snapshots/$SERVE_REVISION
[ -d "$SNAP" ] || { echo "no snapshot at $SNAP" >&2
  echo "\`spark/cache.sh $KEY\` says what it would cost; this script fetches no weights." >&2; exit 2; }
# The same sentence spark/serve.sh:45-57 prints, and true for the same reason: their script runs the
# container with HF_HUB_OFFLINE=1 and mounts the cache, so a serve from here touches no hub.
echo "HF_HUB_OFFLINE=1: $SERVE_MODEL at $SERVE_REVISION is in $HF_CACHE, so this serve needs no network" >&2

# --- the clone, at the pin, unmodified ------------------------------------------------------------
# Absent: clone it. Present: it must BE the pin and be clean. Not a pull, not a checkout -f, not a
# stash - a clone that has been edited is a different server from the one the numbers describe, and
# repairing it would destroy whatever the person who edited it was doing.
if [ ! -d "$DIR/.git" ]; then
  echo "cloning $SERVE_EXTERNAL_REPO into $DIR (a few MB)"
  git clone -q "$SERVE_EXTERNAL_REPO" "$DIR"
  git -C "$DIR" checkout -q --detach "$SERVE_EXTERNAL_COMMIT"
fi
HEAD_AT=$(git -C "$DIR" rev-parse HEAD)
if [ "$HEAD_AT" != "$SERVE_EXTERNAL_COMMIT" ]; then
  echo "$DIR is at $HEAD_AT, not the pinned $SERVE_EXTERNAL_COMMIT." >&2
  echo "This script does not move a clone it did not make. Check it out yourself, or point" >&2
  echo "SERVE_EXTERNAL_DIR at a fresh path and it will clone the pin there." >&2
  exit 4
fi
if [ -n "$(git -C "$DIR" status --porcelain)" ]; then
  echo "$DIR has local modifications: it is driven, never patched, so the image built from it would" >&2
  echo "not be the one $KEY's published result was measured on. \`git -C $DIR status\` to see them." >&2
  exit 4
fi

# --- the image ------------------------------------------------------------------------------------
# The build is ~1 min on top of a base image that is ~20 GB and comes from a registry. Say so before
# it starts, the same courtesy spark/serve.sh pays for weights: nobody should discover a 20 GB pull
# from their bandwidth bill.
if ! docker image inspect "$SERVE_EXTERNAL_TAG" >/dev/null 2>&1; then
  grep -qF "FROM $SERVE_EXTERNAL_BASE_IMAGE" "$DIR/Dockerfile" \
    || { echo "$DIR/Dockerfile at $SERVE_EXTERNAL_COMMIT does not build FROM the pinned base image" >&2
         echo "  pinned here: $SERVE_EXTERNAL_BASE_IMAGE" >&2; exit 4; }
  if docker image inspect "$SERVE_EXTERNAL_BASE_IMAGE" >/dev/null 2>&1; then
    echo "building $SERVE_EXTERNAL_TAG from $DIR at $SERVE_EXTERNAL_COMMIT (~1 min; the base image is here)" >&2
  else
    echo "this box has neither $SERVE_EXTERNAL_TAG nor its base image." >&2
    echo "The build will FIRST PULL ABOUT 20 GB: $SERVE_EXTERNAL_BASE_IMAGE" >&2
    echo "Then it patches it into $SERVE_EXTERNAL_TAG (~1 min). Ctrl-C now if that is not what you want." >&2
  fi
  docker build -t "$SERVE_EXTERNAL_TAG" "$DIR"
fi

# --- start it -------------------------------------------------------------------------------------
# Their defaults are not ours and two of the differences are load-bearing. MTP: their default is 2
# (scripts/serve.sh:62) and the container that produced spark/results/dresser-flash-next-medium/
# carries no --speculative-config at all, so SERVE_EXTERNAL_MTP=0 is what the published numbers were
# measured with. SEQS: theirs is 8, ours is 4. Their $EXTRA is expanded unquoted, so the image limit
# must be a single word - the spaces come out of the JSON rather than splitting it into two flags.
docker rm -f "$NAME" >/dev/null 2>&1 || true
env NAME="$NAME" IMAGE="$SERVE_EXTERNAL_TAG" MODEL="$SERVE_MODEL" HF_CACHE="$HF_CACHE" PORT="$PORT" \
    CTX="$SERVE_MAX_MODEL_LEN" SEQS="$SERVE_MAX_SEQS" GPU_MEM="$SERVE_GPU_UTIL" \
    MTP="${SERVE_EXTERNAL_MTP:-0}" \
    EXTRA="${SERVE_LIMIT_MM:+--limit-mm-per-prompt ${SERVE_LIMIT_MM// /}}" \
    "$DIR/scripts/serve.sh"
# scripts/serve.sh:140 hardcodes --restart unless-stopped and offers no env knob for it. A measured
# server that resurrects itself after a reboot is not a measured server; drop the policy at once.
docker update --restart=no "$NAME" >/dev/null
# Their script serves whichever snapshot the cache's refs/main points at, which need not be the one
# this model file pins. Read what the container actually got, not what we asked for.
GOT=$(docker inspect -f '{{index .Config.Cmd 0}}' "$NAME" 2>/dev/null || true)
case "$GOT" in
  */"$SERVE_REVISION") ;;
  *) echo "warning: $NAME is serving '$GOT', not the pinned SERVE_REVISION=$SERVE_REVISION" >&2 ;;
esac
LOG=$LOG_DIR/$KEY.log
( closefds; exec setsid nohup docker logs -f "$NAME" > "$LOG" 2>&1 < /dev/null ) & disown
echo "serving $SERVE_MODEL as '$SERVE_NAME' on :$PORT from $SERVE_EXTERNAL_TAG  (log: $LOG)"

# The same readiness contract as spark/serve.sh:79-85, deliberately duplicated rather than shared:
# 240 x 10 s = 40 min, and the container is checked each time so an exit is reported as an exit. Their
# script returns as soon as docker does and tells you to watch the log for "Application startup
# complete"; a caller of this one can wait on its exit code. Flash-Next's first boot mmaps ~76 GiB and
# was measured ready at 742-752 s, where a small model takes ~4 min.
printf 'waiting for the server'
for _ in $(seq 1 240); do
  if curl -sf -m 2 "localhost:$PORT/v1/models" >/dev/null; then echo " ready"; exit 0; fi
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo; echo "container exited:" >&2; tail -30 "$LOG" >&2; exit 1; }
  printf '.'; sleep 10
done
echo; echo "not ready after 40 min, see $LOG" >&2; exit 1
