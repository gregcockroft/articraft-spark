#!/usr/bin/env bash
# spark/cache.sh and spark/serve.sh's offline decision, against a stub `docker` and a temporary
# HF_CACHE. No container is started, no daemon is contacted and nothing is downloaded.
#
#   spark/test_cache.sh
#
# It ships here rather than in tests/ because tests/ is the Python package's suite and these are shell
# scripts; a person who clones this repo to serve a local model can run this before touching the box.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/spark_test_cache.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); printf '  ok  %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  BAD %s\n' "$1"; }
eq()  { if [ "$2" = "$3" ]; then ok "$1: $2"; else bad "$1: got '$2', want '$3'"; fi; }
has() { if grep -qF -- "$2" "$3"; then ok "$1"; else bad "$1 (no '$2' in $3)"; fi; }
hasnt() { if grep -qF -- "$2" "$3"; then bad "$1 (found '$2' in $3)"; else ok "$1"; fi; }
# A check that cannot run must not vanish. Two `hasnt` calls were added to this file before the helper
# was, and bash's command-not-found is not an error under `set -uo pipefail`: the run printed two shell
# errors and still said "40 ok, 0 bad -> PASSED" - the same family as the 0-byte docker stub and the
# piped exit code, a missing check reading as a clean one. An ERR trap is the wrong guard here (this
# file expects non-zero exits everywhere: cache.sh returns 2 by design, and every grep -q may miss), so
# the guard is a floor on the number of checks, raised deliberately when cases are added.
CHECKS_MIN=${CHECKS_MIN:-39}

# A stub docker that records its argv and answers the two queries cache.sh and serve.sh make.
STUB=$TMP/bin; mkdir -p "$STUB"
cat > "$STUB/docker" <<'EOF'
#!/bin/sh
{ printf 'docker'; for a in "$@"; do printf ' [%s]' "$a"; done; printf '\n'; } >> "$STUB_LOG"
case "$1 $2" in
  "image inspect") [ -n "${STUB_IMAGE_ID:-}" ] && echo "${STUB_IMAGE_ID}" ;;
  *) case "$1" in
       ps)     echo "" ;;
       run)    echo stub-container-id ;;
       logs|rm|update|stop) ;;
     esac ;;
esac
exit 0
EOF
printf '#!/bin/sh\nexit 1\n' > "$STUB/curl"     # serve.sh's readiness poll must fail fast, not wait
chmod +x "$STUB/docker" "$STUB/curl"

# A cache with the pinned snapshot of one key in it, and one without.
KEY_PRESENT=qwen3.8-27b-inferact
REPO=$(  (set -a; . "$HERE/models/$KEY_PRESENT.env"; echo "$SERVE_MODEL") )
REV=$(   (set -a; . "$HERE/models/$KEY_PRESENT.env"; echo "$SERVE_REVISION") )
IMAGE=$( (set -a; . "$HERE/models/$KEY_PRESENT.env"; echo "$SERVE_IMAGE") )
FULL=$TMP/cache_full; EMPTY=$TMP/cache_empty
mkdir -p "$FULL/hub/models--${REPO//\//--}/snapshots/$REV" "$EMPTY"
# One real file, so the "present" row measures something rather than reporting 0.
head -c 4096 /dev/urandom > "$FULL/hub/models--${REPO//\//--}/snapshots/$REV/config.json"

echo "=== cache.sh: the snapshot and the image are present -> exit 0, nothing to fetch ==="
out=$TMP/present.txt
STUB_LOG=$TMP/docker_present.log STUB_IMAGE_ID=sha256:stubimage \
  env PATH="$STUB:$PATH" HF_CACHE="$FULL" "$HERE/cache.sh" "$KEY_PRESENT" > "$out" 2>&1
eq "exit code" "$?" 0
has "the snapshot row says present" "present   snapshot $REPO" "$out"
has "the image row says present" "present   image $IMAGE" "$out"
has "and it says so in one line a human can act on" "nothing to fetch: 0 bytes" "$out"
has "naming the key it checked" "spark/serve.sh $KEY_PRESENT can run with no network" "$out"
# The fixture snapshot is a 4 KiB directory. Asserted in SI (4.1 kB) because cache.sh reports SI:
# it divided by 1024 and labelled the result "KB"/"GB" until 2026-09-12, and this line asserted that
# wrong label, so the test agreed with the defect instead of catching it. A size assertion names its
# unit or it is asserting a rendering, not a measurement.
has "the present snapshot's size is MEASURED, not a hint" "4.1 kB" "$out"

echo "=== cache.sh: the cache is empty -> exit 2, the missing snapshot named WITH its size ==="
out=$TMP/absent.txt
before=$(cat /sys/class/net/enP7s7/statistics/rx_bytes 2>/dev/null || echo 0)
STUB_LOG=$TMP/docker_absent.log STUB_IMAGE_ID=sha256:stubimage \
  env PATH="$STUB:$PATH" HF_CACHE="$EMPTY" "$HERE/cache.sh" "$KEY_PRESENT" > "$out" 2>&1
eq "exit code" "$?" 2
has "the snapshot row says MISSING" "MISSING   snapshot $REPO" "$out"
has "with a size, not just a name" "to fetch" "$out"
has "and the path it would go to, with the revision" "would go to $EMPTY/hub/models--${REPO//\//--}/snapshots/$REV" "$out"
has "the total is stated so a caller knows the cost up front" "would be downloaded" "$out"
has "and it says how to avoid it" "set HF_CACHE to a cache that has it" "$out"
# The condition-false case has to be shown to download NOTHING, not merely to report.
remaining=$(find "$EMPTY" -mindepth 1 | wc -l)
eq "the empty cache is still empty" "$remaining" 0
after=$(cat /sys/class/net/enP7s7/statistics/rx_bytes 2>/dev/null || echo 0)
moved=$((after - before))
if [ "$moved" -lt 5000000 ]; then ok "the uplink moved $moved bytes (< 5 MB: nothing was fetched)"
else bad "the uplink moved $moved bytes"; fi
if grep -q 'pull' "$TMP/docker_absent.log" 2>/dev/null; then bad "it tried to pull an image without --fetch"
else ok "no docker pull without --fetch"; fi

echo "=== cache.sh: every model file reports, and Flash-Next names its extra requirements ==="
for key in $(cd "$HERE/models" && ls ./*.env | sed 's|^\./||; s/\.env$//'); do
  out=$TMP/all_$key.txt
  STUB_LOG=$TMP/docker_all.log STUB_IMAGE_ID= \
    env PATH="$STUB:$PATH" HF_CACHE="$EMPTY" "$HERE/cache.sh" "$key" > "$out" 2>&1
  rc=$?
  if [ "$rc" = 2 ] && grep -q 'MISSING' "$out"; then ok "$key: reports, exit 2 on an empty cache"
  else bad "$key: exit $rc, and MISSING $(grep -c MISSING "$out") rows"; fi
done
out=$TMP/all_qwen3.8-flash-next-nvfp4.txt
has "Flash-Next names blazux's clone" "clone https://github.com/blazux/qwen3.8-Flash-DGX" "$out"
has "Flash-Next names the locally built image" "image qwen38-flash-dgx:bd60fcb" "$out"
has "and says how to get that image without a registry" "docker save/load it from another box" "$out"

echo "=== the agreement cache.sh:38 and serve.sh claim 'by construction', CHECKED for every key ==="
# The two scripts derive the same snapshot path from the same fields. That comment was a claim until the
# conductor resolved it against the real cache by hand (2026-09-12 ~11:12); this makes it a check that
# runs for every key on the branch, every time, at no cost to the box.
for key in $(cd "$HERE/models" && ls ./*.env | sed 's|^\./||; s/\.env$//'); do
  repo=$(  (set -a; . "$HERE/models/$key.env"; echo "$SERVE_MODEL") )
  rev=$(   (set -a; . "$HERE/models/$key.env"; echo "${SERVE_REVISION:-}") )
  if [ -z "$rev" ]; then
    # Unpinned (spark/models/qwen3.8-27b.env ships that way): there is no path to agree on, and both
    # scripts must say so rather than inventing one. Anything containing "none" or a bare trailing
    # slash is the bug this case exists to catch.
    out=$TMP/unpinned_$key.txt
    STUB_LOG=$TMP/docker_unpinned.log STUB_IMAGE_ID=sha256:stubimage \
      env PATH="$STUB:$PATH" HF_CACHE="$FULL" "$HERE/cache.sh" "$key" > "$out" 2>&1
    eq "$key is unpinned: cache.sh exits 2" "$?" 2
    has "$key: cache.sh says the size cannot be stated" "size unknown" "$out"
    has "$key: and why" "no pinned revision" "$out"
    hasnt "$key: no invented snapshot path" "snapshots/none" "$out"
    out=$TMP/unpinned_serve_$key.txt
    STUB_LOG=$TMP/docker_unpinned.log STUB_IMAGE_ID=sha256:stubimage \
      env -u HF_HUB_OFFLINE PATH="$STUB:$PATH" HF_CACHE="$FULL" ARTICRAFT_SERVE_LOGS="$TMP/serve" \
      "$HERE/serve.sh" "$key" > "$out" 2>&1
    has "$key: serve.sh names the hub, not a snapshot called 'none'" "pins no SERVE_REVISION" "$out"
    hasnt "$key: serve.sh invents no path" "snapshots/none" "$out"
    continue
  fi
  # Whether THIS box happens to hold the weights is a box fact, not a script fault: on a machine that
  # runs the suite but never serves (ubu24) they are absent, and this file has to pass there too. So it
  # is reported, never scored - what IS scored is that the two scripts agree about it either way.
  path=$HOME/.cache/huggingface/hub/models--$(printf '%s' "$repo" | sed 's|/|--|g')/snapshots/$rev
  if [ -d "$path" ]; then echo "  --  $key: this box holds $repo at $rev"
  else echo "  --  $key: this box does not hold $repo at $rev (cache.sh will report MISSING here)"; fi
  # And the two scripts must agree about that path: cache.sh's own report, against serve.sh's decision.
  out=$TMP/agree_$key.txt
  STUB_LOG=$TMP/docker_agree.log STUB_IMAGE_ID=sha256:stubimage \
    env PATH="$STUB:$PATH" HF_CACHE="$HOME/.cache/huggingface" "$HERE/cache.sh" "$key" > "$out" 2>&1
  serve_out=$TMP/agree_serve_$key.txt
  STUB_LOG=$TMP/docker_agree.log STUB_IMAGE_ID=sha256:stubimage \
    env -u HF_HUB_OFFLINE PATH="$STUB:$PATH" HF_CACHE="$HOME/.cache/huggingface" \
    ARTICRAFT_SERVE_LOGS="$TMP/serve" "$HERE/serve.sh" "$key" > "$serve_out" 2>&1
  cache_says=$(grep -q "present   snapshot $repo" "$out" && echo present || echo missing)
  serve_says=$(grep -q 'HF_HUB_OFFLINE=1' "$serve_out" && echo present || echo missing)
  eq "$key: cache.sh and serve.sh agree on the snapshot" "$cache_says" "$serve_says"
done

echo "=== serve.sh: HF_HUB_OFFLINE=1 when the snapshot is present ==="
log=$TMP/docker_serve_present.log; : > "$log"
STUB_LOG=$log STUB_IMAGE_ID=sha256:stubimage \
  env -u HF_HUB_OFFLINE PATH="$STUB:$PATH" HF_CACHE="$FULL" ARTICRAFT_SERVE_LOGS="$TMP/serve" \
  "$HERE/serve.sh" "$KEY_PRESENT" > "$TMP/serve_present.txt" 2>&1
has "the run passes HF_HUB_OFFLINE=1" "[-e] [HF_HUB_OFFLINE=1]" "$log"
has "and says why, so a caller is not guessing" "is in $FULL, so this serve needs no network" "$TMP/serve_present.txt"

echo "=== serve.sh: HF_HUB_OFFLINE=0 when it is absent, and it says so ==="
log=$TMP/docker_serve_absent.log; : > "$log"
STUB_LOG=$log STUB_IMAGE_ID=sha256:stubimage \
  env -u HF_HUB_OFFLINE PATH="$STUB:$PATH" HF_CACHE="$EMPTY" ARTICRAFT_SERVE_LOGS="$TMP/serve" \
  "$HERE/serve.sh" "$KEY_PRESENT" > "$TMP/serve_absent.txt" 2>&1
has "the run passes HF_HUB_OFFLINE=0" "[-e] [HF_HUB_OFFLINE=0]" "$log"
has "and warns that the weights will be downloaded" "the weights will be downloaded" "$TMP/serve_absent.txt"
has "pointing at cache.sh for the size" "spark/cache.sh $KEY_PRESENT\` says how much" "$TMP/serve_absent.txt"

echo "=== serve.sh: an explicit HF_HUB_OFFLINE from the caller wins, in BOTH directions ==="
# A change that silently overrides what a caller asked for is worse than one that does nothing.
log=$TMP/docker_serve_force0.log; : > "$log"
STUB_LOG=$log STUB_IMAGE_ID=sha256:stubimage \
  env PATH="$STUB:$PATH" HF_CACHE="$FULL" HF_HUB_OFFLINE=0 ARTICRAFT_SERVE_LOGS="$TMP/serve" \
  "$HERE/serve.sh" "$KEY_PRESENT" > "$TMP/serve_force0.txt" 2>&1
has "cached but the caller said 0: it passes 0" "[-e] [HF_HUB_OFFLINE=0]" "$log"
has "and attributes it to the caller" "(from the caller)" "$TMP/serve_force0.txt"
log=$TMP/docker_serve_force1.log; : > "$log"
STUB_LOG=$log STUB_IMAGE_ID=sha256:stubimage \
  env PATH="$STUB:$PATH" HF_CACHE="$EMPTY" HF_HUB_OFFLINE=1 ARTICRAFT_SERVE_LOGS="$TMP/serve" \
  "$HERE/serve.sh" "$KEY_PRESENT" > "$TMP/serve_force1.txt" 2>&1
has "not cached but the caller said 1: it passes 1" "[-e] [HF_HUB_OFFLINE=1]" "$log"

total=$((pass + fail))
if [ "$total" -lt "$CHECKS_MIN" ]; then
  bad "only $total checks ran, expected at least $CHECKS_MIN - a case was skipped or a helper is missing"
fi
printf '\nspark/test_cache.sh: %d ok, %d bad, %d checks (floor %d) -> %s\n' \
  "$pass" "$fail" "$total" "$CHECKS_MIN" "$([ "$fail" = 0 ] && echo PASSED || echo FAILED)"
exit $([ "$fail" = 0 ] && echo 0 || echo 1)
