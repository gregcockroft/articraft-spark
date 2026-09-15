#!/usr/bin/env bash
# What a local serve needs, and whether this box already has it.
#
#   spark/cache.sh qwen3.6-35b-a3b-nvfp4          # report; exit 0 if nothing would be downloaded, else 2
#   spark/cache.sh qwen3.8-27b-inferact --fetch   # download the missing pieces, then report again
#
# A clone of this repo is small; the weights are not. This says exactly what a given model file needs
# before `spark/serve.sh <key>` can work offline, so "reproducible from a clean pull" does not mean
# "download a hundred gigabytes and hope". Point HF_CACHE at a cache you already have and nothing is
# fetched at all:
#
#   HF_CACHE=/mnt/models/huggingface spark/cache.sh qwen3.8-27b-inferact
#
# Without --fetch this script downloads nothing and starts no container. With --fetch it downloads from
# the Hugging Face hub with no token (HF_HUB_DISABLE_IMPLICIT_TOKEN=1) and retries a stalled transfer,
# resuming; `hf download` is resumable, so a retry costs only what it had not finished.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=${1:?usage: spark/cache.sh <model-key> [--fetch]   (one of: $(cd "$HERE/models" && ls *.env | sed 's/\.env$//' | tr '\n' ' '))}
FETCH=0; [ "${2:-}" = --fetch ] && FETCH=1
ENV_FILE=$HERE/models/$KEY.env
[ -f "$ENV_FILE" ] || { echo "no model file $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
TRIES=${CACHE_TRIES:-5}

missing_bytes=0
size_unknown=0     # set when a MISSING row cannot state its size (an unpinned model file)
rows=()   # "state<TAB>what<TAB>size<TAB>detail"
row() { rows+=("$1	$2	$3	$4"); }
# SI, deliberately: a reader of these numbers is sizing a disk, and disks are sold in SI bytes.
# This divided by 1024 and labelled the result "GB" until 2026-09-12, which made every size read
# ~7 % smaller than the space it needs - 126.0 "GB" for a snapshot that is 135.3 GB.
human() { awk -v b="$1" 'BEGIN{ if (b == 0) { print "-"; exit }
  split("B kB MB GB TB", u, " "); i = 1; while (b >= 1000 && i < 5) { b /= 1000; i++ }
  printf "%.1f %s\n", b, u[i] }'; }

# --- the snapshot ---------------------------------------------------------------------------------
# A model file pins SERVE_MODEL and SERVE_REVISION, and the hub cache stores that pair at
# hub/models--<org>--<name>/snapshots/<revision>. serve.sh checks the same path to decide whether it can
# run offline, so the two agree by construction.
snapshot_dir() { echo "$HF_CACHE/hub/models--${1//\//--}/snapshots/$2"; }
# Measured with `du` on this box, 2026-09-12, and used only to say how large a MISSING download would be.
# A present snapshot is measured for real, so a stale hint can never inflate a "present" row.
size_hint() {
  case "$1" in
    RadixArk/Qwen3.8-Flash-Next-NVFP4) echo 135291469824 ;;   # ~126 GiB
    Inferact/Qwen3.8-27B-NVFP4)        echo 26843545600 ;;    # ~25 GiB
    RedHatAI/Qwen3.6-35B-A3B-NVFP4)    echo 25769803776 ;;    # ~24 GiB
    Qwen/Qwen3.8-27B-FP8)              echo 31138512896 ;;    # ~29 GiB
    *)                                 echo 0 ;;
  esac
}

check_snapshot() {
  local repo=$1 rev=$2 dir size want
  # An UNPINNED model file (SERVE_REVISION empty - spark/models/qwen3.8-27b.env ships that way) has no
  # snapshot path to check: a serve would resolve `main` at the hub and fetch whatever is current. Say
  # that, rather than reporting a missing snapshot called "none" at a path nothing will ever write.
  if [ -z "$rev" ]; then
    size_unknown=1
    row MISSING "snapshot $repo" "size unknown" \
        "no pinned revision: a serve would resolve 'main' at the hub and fetch whatever is current, so the size cannot be stated here. Pin SERVE_REVISION."
    return 1
  fi
  dir=$(snapshot_dir "$repo" "$rev")
  if [ -d "$dir" ]; then
    # -L follows the symlinks a snapshot holds into blobs/, which is where the bytes actually are.
    size=$(du -sbL "$dir" 2>/dev/null | cut -f1); size=${size:-0}
    row present "snapshot $repo" "$(human "$size")" "$dir"
    return 0
  fi
  want=$(size_hint "$repo")
  [ "$want" = 0 ] && size_unknown=1
  missing_bytes=$((missing_bytes + want))
  row MISSING "snapshot $repo" "$(human "$want") to fetch" "would go to $dir (revision $rev)"
  return 1
}

fetch_snapshot() {
  local repo=$1 rev=$2 try=1
  command -v hf >/dev/null || { echo "--fetch needs the 'hf' CLI (pip install huggingface_hub[cli])" >&2; return 1; }
  while [ "$try" -le "$TRIES" ]; do
    echo "--fetch: try $try of $TRIES: hf download $repo --revision $rev"
    # T6: the hub only, and no token - an unauthenticated download of a public repo, which is what these
    # pins are. T8: a stalled CDN connection is retried rather than waited on; hf download resumes.
    if env HF_HOME="$HF_CACHE" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=0 \
         hf download "$repo" --revision "$rev" >/dev/null; then
      echo "--fetch: $repo is in the cache after $try try(s)"; return 0
    fi
    try=$((try + 1)); sleep 10
  done
  echo "--fetch: $repo failed after $TRIES tries" >&2; return 1
}

# --- the image ------------------------------------------------------------------------------------
check_image() {
  local ref=$1 id size
  id=$(docker image inspect -f '{{.Id}}' "$ref" 2>/dev/null)
  if [ -n "$id" ]; then
    size=$(docker image inspect -f '{{.Size}}' "$ref" 2>/dev/null); size=${size:-0}
    row present "image $ref" "$(human "$size")" "$id"
    return 0
  fi
  row MISSING "image $ref" "~20-25 GB to pull" "docker pull $ref"
  return 1
}

fetch_image() { echo "--fetch: docker pull $1"; docker pull "$1"; }

# --- what each key needs --------------------------------------------------------------------------
# Narrow on purpose: the one model this repo cannot serve itself names its own extra requirements here,
# rather than every model file growing fields for a case that applies to one of them. Flash-Next's env
# file says the same thing in prose at its top.
BLAZUX_REPO=https://github.com/blazux/qwen3.8-Flash-DGX
BLAZUX_PIN=bd60fcb1b492ca920f74df7462f05da7b6d98f73
BLAZUX_DIR=${BLAZUX_DIR:-$HOME/src/qwen3.8-Flash-DGX}
BLAZUX_IMAGE=qwen38-flash-dgx:bd60fcb

check_blazux() {
  local head
  head=$(git -C "$BLAZUX_DIR" rev-parse HEAD 2>/dev/null)
  if [ "$head" = "$BLAZUX_PIN" ]; then
    row present "clone $BLAZUX_REPO" "$(human "$(du -sb "$BLAZUX_DIR" 2>/dev/null | cut -f1)")" "$BLAZUX_DIR at $BLAZUX_PIN"
  else
    row MISSING "clone $BLAZUX_REPO" "~5 MB to clone" "want $BLAZUX_DIR at $BLAZUX_PIN, found '${head:-nothing}'"
    return 1
  fi
}

echo "cache check: $KEY   (HF_CACHE=$HF_CACHE)"
ok=0
check_snapshot "$SERVE_MODEL" "$SERVE_REVISION" || ok=1
if [ -n "${SERVE_IMAGE:-}" ]; then
  check_image "$SERVE_IMAGE" || ok=1
fi
if [ "$KEY" = qwen3.8-flash-next-nvfp4 ]; then
  # This model is not served by spark/serve.sh: it needs blazux's patched vLLM and the image built from
  # it, which is why its env file has no SERVE_IMAGE. Both are named so the report is complete.
  check_blazux || ok=1
  check_image "$BLAZUX_IMAGE" || ok=1
  row note "built image" "-" "$BLAZUX_IMAGE is built locally: (cd $BLAZUX_DIR && scripts/build.sh), or docker save/load it from another box"
fi

if [ "$FETCH" = 1 ] && [ "$ok" != 0 ]; then
  echo
  fetch_snapshot "$SERVE_MODEL" "$SERVE_REVISION" || true
  [ -n "${SERVE_IMAGE:-}" ] && [ -z "$(docker image inspect -f '{{.Id}}' "$SERVE_IMAGE" 2>/dev/null)" ] \
    && fetch_image "$SERVE_IMAGE"
  echo
  echo "re-checking after --fetch:"
  rows=(); missing_bytes=0; ok=0
  check_snapshot "$SERVE_MODEL" "$SERVE_REVISION" || ok=1
  [ -n "${SERVE_IMAGE:-}" ] && { check_image "$SERVE_IMAGE" || ok=1; }
fi

printf '\n%-8s  %-52s  %-16s  %s\n' STATE WHAT SIZE DETAIL
printf '%s\n' "${rows[@]}" | awk -F'\t' '{ printf "%-8s  %-52s  %-16s  %s\n", $1, $2, $3, $4 }'
echo
if [ "$ok" = 0 ]; then
  echo "nothing to fetch: 0 bytes. spark/serve.sh $KEY can run with no network."
  exit 0
fi
if [ "$size_unknown" = 1 ]; then
  if [ "$missing_bytes" -gt 0 ]; then
    echo "at least $(human "$missing_bytes") would be downloaded, plus items whose size is not known here."
  else
    echo "an unknown amount would be downloaded: nothing missing here can state its size."
  fi
  echo "Re-run with --fetch, or set HF_CACHE to a cache that has it."
else
  echo "$(human "$missing_bytes") would be downloaded. Re-run with --fetch, or set HF_CACHE to a cache that has it."
fi
exit 2
