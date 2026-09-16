#!/usr/bin/env bash
# spark/serve_external.sh against a stub `docker`, a stub `curl` and a synthetic clone. No container
# is started, no daemon is contacted, no image is built and NOTHING IS DOWNLOADED - in particular the
# external project is never cloned from the network: the fixture is a local git repo this script
# makes, with the real project's scripts/serve.sh and Dockerfile copied in when this box has them at
# the pin, so what is asserted is the real argv their launcher builds.
#
#   spark/test_serve_external.sh
#
# It ships beside spark/test_cache.sh rather than in tests/ for the same reason: tests/ is the Python
# package's suite and these are shell scripts.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/spark_test_serve_ext.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); printf '  ok  %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  BAD %s\n' "$1"; }
eq()  { if [ "$2" = "$3" ]; then ok "$1: $2"; else bad "$1: got '$2', want '$3'"; fi; }
has() { if grep -qF -- "$2" "$3"; then ok "$1"; else bad "$1 (no '$2' in $3)"; fi; }
hasnt() { if grep -qF -- "$2" "$3"; then bad "$1 (found '$2' in $3)"; else ok "$1"; fi; }
# The same floor spark/test_cache.sh keeps, and for the same reason: a case that stops running must
# not read as a clean pass. Raise it deliberately when cases are added.
CHECKS_MIN=${CHECKS_MIN:-58}

# --- stubs ----------------------------------------------------------------------------------------
STUB=$TMP/bin; mkdir -p "$STUB"
cat > "$STUB/docker" <<'EOF'
#!/bin/sh
{ printf 'docker'; for a in "$@"; do printf ' [%s]' "$a"; done; printf '\n'; } >> "$STUB_LOG"
case "$1 $2" in
  "image inspect")
    for ref in ${STUB_IMAGES:-}; do [ "$ref" = "$3" ] && { echo "sha256:stub"; exit 0; }; done
    exit 1 ;;
  "inspect -f")
    case "$3" in
      *State.Status*) echo "running exit=0 oom=false" ;;
      *Config.Cmd*)   echo "${STUB_CMD0:-}" ;;
    esac
    exit 0 ;;
esac
case "$1" in
  ps)   echo "${STUB_PS:-}" ;;
  # `docker logs -f` blocks for the container's life. /bin/sleep by absolute path on purpose: the
  # stub `sleep` beside this file is the instant one their script's own `sleep 8` must hit, and a
  # follower that exits at once cannot show whether it was holding a lock.
  logs) exec /bin/sleep "${STUB_LOGS_SLEEP:-0}" ;;
esac
exit 0
EOF
printf '#!/bin/sh\nexit ${STUB_CURL_RC:-1}\n' > "$STUB/curl"
printf '#!/bin/sh\nexit 0\n' > "$STUB/sleep"   # their script sleeps 8 s before reading the state
chmod +x "$STUB/docker" "$STUB/curl" "$STUB/sleep"

# --- the launcher under test, with model files we control -------------------------------------------
# serve_external.sh reads spark/models/<key>.env relative to its own directory, so the copy gets its
# own models/ and the fixture can pin a commit that exists.
FAKE=$TMP/spark; mkdir -p "$FAKE/models"
cp "$HERE/serve_external.sh" "$FAKE/serve_external.sh"

REAL_KEY=qwen3.8-flash-next-nvfp4
REAL_ENV=$HERE/models/$REAL_KEY.env
PIN=$(  (set -a; . "$REAL_ENV"; echo "$SERVE_EXTERNAL_COMMIT") )
UPSTREAM=${SERVE_EXTERNAL_DIR:-$HOME/src/$(basename "$( (set -a; . "$REAL_ENV"; echo "$SERVE_EXTERNAL_REPO") )" .git)}
BASE_IMAGE=$( (set -a; . "$REAL_ENV"; echo "$SERVE_EXTERNAL_BASE_IMAGE") )
TAG=$(        (set -a; . "$REAL_ENV"; echo "$SERVE_EXTERNAL_TAG") )
MODEL=$(      (set -a; . "$REAL_ENV"; echo "$SERVE_MODEL") )
REV=$(        (set -a; . "$REAL_ENV"; echo "$SERVE_REVISION") )

# The fixture clone. Their launcher and Dockerfile are copied from this box's clone when it is at the
# pin; without it the test still runs, against a recorder that stands in for their script, and says so.
CLONE=$TMP/clone; mkdir -p "$CLONE/scripts"
if [ "$(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null)" = "$PIN" ]; then
  cp "$UPSTREAM/scripts/serve.sh" "$CLONE/scripts/serve.sh"
  cp "$UPSTREAM/Dockerfile" "$CLONE/Dockerfile"
  REALITY="the real scripts/serve.sh from $UPSTREAM at $PIN"
else
  printf '#!/usr/bin/env bash\nexec docker run -d --name "$NAME" --restart unless-stopped -p "$PORT:8000" "$IMAGE" recorder\n' \
    > "$CLONE/scripts/serve.sh"
  printf 'FROM %s\n' "$BASE_IMAGE" > "$CLONE/Dockerfile"
  REALITY="a RECORDER standing in for scripts/serve.sh - this box has no clone at $PIN"
fi
chmod +x "$CLONE/scripts/serve.sh"
git -C "$CLONE" init -q
git -C "$CLONE" add -A
git -C "$CLONE" -c user.email=t@example.com -c user.name=t commit -qm fixture
FIX_PIN=$(git -C "$CLONE" rev-parse HEAD)
echo "fixture: $REALITY"

# A cache with the pinned snapshot, and one without.
FULL=$TMP/cache_full; EMPTY=$TMP/cache_empty
mkdir -p "$FULL/hub/models--${MODEL//\//--}/snapshots/$REV" "$EMPTY"
SNAP_IN=/hf/hub/models--${MODEL//\//--}/snapshots/$REV

# The model file under test: the real one, with the fixture's commit and clone path substituted.
mkenv() {  # mkenv <key> <commit> [extra lines...]
  local key=$1 commit=$2; shift 2
  sed -e "s|^SERVE_EXTERNAL_COMMIT=.*|SERVE_EXTERNAL_COMMIT=$commit|" \
      -e "s|^SERVE_EXTERNAL_REPO=.*|SERVE_EXTERNAL_REPO=file://$CLONE|" "$REAL_ENV" > "$FAKE/models/$key.env"
  for line in "$@"; do echo "$line" >> "$FAKE/models/$key.env"; done
}
mkenv flash "$FIX_PIN"
run_ext() {  # run_ext <key> <out-file> <log-file> -- env assignments...
  local key=$1 out=$2 log=$3; shift 3; : > "$log"
  env PATH="$STUB:$PATH" STUB_LOG="$log" HF_CACHE="$FULL" SERVE_EXTERNAL_DIR="$CLONE" \
      ARTICRAFT_SERVE_LOGS="$TMP/logs" STUB_CMD0="$SNAP_IN" "$@" \
      "$FAKE/serve_external.sh" "$key" > "$out" 2>&1
}

echo "=== the settings that reach their launcher come from the model file, not from its defaults ==="
out=$TMP/happy.txt; log=$TMP/happy.log
run_ext flash "$out" "$log" STUB_IMAGES="$TAG" STUB_CURL_RC=0
eq "exit code when the server answers" "$?" 0
has "it says it is ready" " ready" "$out"
RUN=$(grep '^docker \[run\]' "$log" | head -1)
printf '%s\n' "$RUN" > "$TMP/run_argv.txt"
# 1. MTP: their default is 2; the recorded container carries no --speculative-config at all.
hasnt "MTP=0: no --speculative-config on the command line" "--speculative-config" "$TMP/run_argv.txt"
# 4. The capacity flags are the model file's, not their defaults (SEQS=8 there).
has "SEQS comes from SERVE_MAX_SEQS" "[--max-num-seqs] [4]" "$TMP/run_argv.txt"
has "GPU_MEM comes from SERVE_GPU_UTIL" "[--gpu-memory-utilization] [0.80]" "$TMP/run_argv.txt"
has "CTX comes from SERVE_MAX_MODEL_LEN" "[--max-model-len] [262144]" "$TMP/run_argv.txt"
# EXTRA: one argv word, not split by the space a pretty-printed JSON would carry.
has "the image limit is passed through EXTRA" "[--limit-mm-per-prompt] [{\"image\":8}]" "$TMP/run_argv.txt"
has "the port is bound as this repo's :8001 by default" "[-p] [8001:8000]" "$TMP/run_argv.txt"
has "the snapshot the model file pins is what is served" "$SNAP_IN" "$TMP/run_argv.txt"
# 2. The restart policy their script hardcodes is dropped, and read back from docker, not from our log.
has "docker update --restart=no runs" "docker [update] [--restart=no]" "$log"
if [ "$(grep -n 'docker \[run\]' "$log" | cut -d: -f1 | head -1)" -lt "$(grep -n 'docker \[update\] \[--restart=no\]' "$log" | cut -d: -f1 | head -1)" ]; then
  ok "and it runs after the start, not before"
else bad "the restart policy is dropped before the container exists"; fi
has "the container's own snapshot is read back with docker inspect" "docker [inspect] [-f] [{{index .Config.Cmd 0}}]" "$log"
hasnt "no image is built when the tag is already here" "docker [build]" "$log"

echo "=== 1. the negative control: a non-zero MTP reaches the command line, and the test SEES it ==="
mkenv mtp2 "$FIX_PIN" "SERVE_EXTERNAL_MTP=2"
out=$TMP/mtp2.txt; log=$TMP/mtp2.log
run_ext mtp2 "$out" "$log" STUB_IMAGES="$TAG" STUB_CURL_RC=0
grep '^docker \[run\]' "$log" | head -1 > "$TMP/mtp2_argv.txt"
# This is the case that must FAIL the assertion above. If a future edit hardcodes MTP=0, or drops the
# env pass-through entirely, this check goes red and the one above stays green - which is the point.
if grep -qF -- '--speculative-config' "$TMP/mtp2_argv.txt"; then
  ok "SERVE_EXTERNAL_MTP=2 puts --speculative-config on the line, so the MTP=0 check above measures something"
else
  bad "SERVE_EXTERNAL_MTP=2 changed nothing: the MTP=0 check above would pass no matter what"
fi
has "and it is their mtp method, at 2" '[{"method":"mtp","num_speculative_tokens":2}]' "$TMP/mtp2_argv.txt"

echo "=== 5. the clone is driven, never repaired: a wrong commit and a dirty tree both stop it ==="
mkenv wrongpin 0000000000000000000000000000000000000000
out=$TMP/wrongpin.txt; log=$TMP/wrongpin.log
run_ext wrongpin "$out" "$log" STUB_IMAGES="$TAG" STUB_CURL_RC=0
eq "a clone at the wrong commit: exit code" "$?" 4
has "it names what it found and what it wanted" "not the pinned 0000000000000000000000000000000000000000" "$out"
has "and says it will not move a clone it did not make" "does not move a clone it did not make" "$out"
hasnt "nothing was started" "docker [run]" "$log"
echo dirt > "$CLONE/dirty_file"
out=$TMP/dirty.txt; log=$TMP/dirty.log
run_ext flash "$out" "$log" STUB_IMAGES="$TAG" STUB_CURL_RC=0
eq "a clone with local modifications: exit code" "$?" 4
has "it says the image would not be the measured one" "never patched" "$out"
hasnt "nothing was started" "docker [run]" "$log"
rm -f "$CLONE/dirty_file"

echo "=== 6. the ~20 GB base-image pull is disclosed on stderr BEFORE the build starts ==="
out=$TMP/pull.txt; log=$TMP/pull.log
run_ext flash "$out" "$log" STUB_IMAGES="" STUB_CURL_RC=0   # neither the tag nor the base image is here
has "it says the build pulls about 20 GB" "PULL ABOUT 20 GB" "$out"
has "naming the image it will pull" "$BASE_IMAGE" "$out"
has "and it does build" "docker [build] [-t] [$TAG]" "$log"
if [ "$(grep -n 'PULL ABOUT 20 GB' "$out" | cut -d: -f1 | head -1)" -lt "$(grep -n 'serving ' "$out" | cut -d: -f1 | head -1)" ]; then
  ok "the disclosure comes before the container is started, not after"
else bad "the disclosure lands after the start"; fi
out=$TMP/nopull.txt; log=$TMP/nopull.log
run_ext flash "$out" "$log" STUB_IMAGES="$BASE_IMAGE" STUB_CURL_RC=0
hasnt "with the base image present it does not claim a 20 GB pull" "PULL ABOUT 20 GB" "$out"
has "but it does say it is building" "building $TAG" "$out"

echo "=== the Dockerfile must still build FROM the pinned base image ==="
mkenv fromcheck "$FIX_PIN" "SERVE_EXTERNAL_BASE_IMAGE=vllm/vllm-openai@sha256:deadbeef"
out=$TMP/fromcheck.txt; log=$TMP/fromcheck.log
run_ext fromcheck "$out" "$log" STUB_IMAGES="" STUB_CURL_RC=0
eq "a base image the clone does not use: exit code" "$?" 4
has "and it says which pin disagrees" "does not build FROM the pinned base image" "$out"
hasnt "nothing was built" "docker [build]" "$log"

echo "=== the weights are checked here, with the path, and never fetched ==="
out=$TMP/nosnap.txt; log=$TMP/nosnap.log
: > "$log"
env PATH="$STUB:$PATH" STUB_LOG="$log" HF_CACHE="$EMPTY" SERVE_EXTERNAL_DIR="$CLONE" \
    ARTICRAFT_SERVE_LOGS="$TMP/logs" STUB_IMAGES="$TAG" "$FAKE/serve_external.sh" flash > "$out" 2>&1
eq "an empty cache: exit code" "$?" 2
has "it names the snapshot path it wanted" "no snapshot at $EMPTY/hub/models--${MODEL//\//--}/snapshots/$REV" "$out"
has "and points at cache.sh for the cost" "spark/cache.sh flash" "$out"
eq "the empty cache is still empty" "$(find "$EMPTY" -mindepth 1 | wc -l)" 0

echo "=== 3. the readiness contract is the same one spark/serve.sh keeps ==="
out=$TMP/dead.txt; log=$TMP/dead.log
run_ext flash "$out" "$log" STUB_IMAGES="$TAG" STUB_CURL_RC=1 STUB_PS=""
eq "the container is gone and curl never answers: exit code" "$?" 1
has "it reports the exit as an exit" "container exited" "$out"
for f in serve.sh serve_external.sh; do
  eq "$f polls 240 times" "$(grep -c 'seq 1 240' "$HERE/$f")" 1
  eq "$f sleeps 10 s between polls" "$(grep -c 'sleep 10' "$HERE/$f")" 1
  eq "$f states the same 40 min ceiling" "$(grep -c 'not ready after 40 min' "$HERE/$f")" 1
done

echo "=== 7. no fallback: one path, one failure ==="
# A launcher that quietly served something else after the external server failed would produce a run
# whose record says one model and whose numbers came from another.
eq "serve_external.sh never calls the vLLM launcher" "$(grep -c 'HERE/serve\.sh' "$HERE/serve_external.sh")" 0
eq "and retries nothing" "$(grep -cE '\|\| *(docker build|git (pull|checkout|fetch)|exec )' "$HERE/serve_external.sh")" 0
eq "and it starts exactly one container" "$(grep -c 'docker run' "$HERE/serve_external.sh")" 0
eq "serve.sh hands the key over with exec, so there is one process and one exit code" \
   "$(grep -c 'exec "$HERE/serve_external.sh"' "$HERE/serve.sh")" 1

echo "=== spark/serve.sh dispatches the real key here, rather than refusing it with exit 3 ==="
# End to end through the real script and the real model file: what changed on main is that this key
# used to exit 3 with a pointer to a comment block. It now reaches this launcher, which stops on the
# fixture clone (not the pin) - that exit 4 IS the proof the dispatch happened.
out=$TMP/dispatch.txt; log=$TMP/dispatch.log; : > "$log"
env PATH="$STUB:$PATH" STUB_LOG="$log" HF_CACHE="$FULL" SERVE_EXTERNAL_DIR="$CLONE" \
    ARTICRAFT_SERVE_LOGS="$TMP/logs" STUB_IMAGES="$TAG" "$HERE/serve.sh" "$REAL_KEY" > "$out" 2>&1
eq "spark/serve.sh $REAL_KEY: exit code" "$?" 4
hasnt "it no longer refuses the model" "not served by spark/serve.sh" "$out"
has "it reached serve_external.sh's clone check" "not the pinned $PIN" "$out"
has "and it said the weights are already here" "so this serve needs no network" "$out"

echo "=== the lock: neither launcher may leave a flock held by the log follower it backgrounds ==="
# docs/TRACKS.md, 2026-09-12: `setsid nohup docker logs -f` inherits the caller's flock descriptors, so
# `flock <lock> spark/serve.sh <key>` held the lock for the CONTAINER'S life, by a ppid-1 process that
# fuser reports as "docker". It was fixed in the switch scripts and, until this commit, nowhere here.
# The stub follower sleeps for this case, so a leaked descriptor is the difference between a lock that
# can be taken again and one that cannot - which is exactly what the defect looked like on the box.
LOCK=$TMP/lock; : > "$LOCK"
for launcher in serve.sh serve_external.sh; do
  key=$REAL_KEY; [ "$launcher" = serve_external.sh ] && key=$REAL_KEY
  out=$TMP/lock_$launcher.txt; log=$TMP/lock_$launcher.log; : > "$log"
  env PATH="$STUB:$PATH" STUB_LOG="$log" STUB_IMAGES="$TAG" STUB_CURL_RC=0 STUB_LOGS_SLEEP=5423 \
      STUB_CMD0="$SNAP_IN" HF_CACHE="$FULL" SERVE_EXTERNAL_DIR="$UPSTREAM" \
      ARTICRAFT_SERVE_LOGS="$TMP/logs" \
      flock "$LOCK" "$HERE/$launcher" "$key" > "$out" 2>&1
  rc=$?
  # The follower is started detached, so give it a moment to appear before anything is concluded from
  # its absence; a case that races is a case that lies in one direction or the other.
  follower=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    follower=$(pgrep -f "[s]leep 5423" | head -1); [ -n "$follower" ] && break; sleep 0.2
  done
  if flock -n "$LOCK" true; then ok "$launcher returned (exit $rc) and the lock is free"
  else bad "$launcher left the lock held: $(fuser -v "$LOCK" 2>&1 | tail -1)"; fi
  # Without a LIVE follower this case proves nothing: one that had exited would free the lock however
  # its descriptors were handled. And the lock's absence is read from the follower's own fd table,
  # which is where the defect actually lived - not inferred from the lock being takeable.
  if [ -n "$follower" ]; then
    ok "$launcher: the log follower (pid $follower) is still running, so the lock was freed by the fix and not by its exit"
    eq "$launcher: and its own fd table holds no lock" \
       "$(ls -l /proc/$follower/fd 2>/dev/null | grep -c "$LOCK")" 0
  else
    bad "$launcher: no follower is running; this case cannot see the defect"
    sed 's/^/      | /' "$out" | tail -5
  fi
  [ -n "$follower" ] && kill "$follower" 2>/dev/null
done
eq "both launchers close inherited descriptors before backgrounding it" \
   "$(grep -lc 'closefds' "$HERE/serve.sh" "$HERE/serve_external.sh" | wc -l)" 2

total=$((pass + fail))
if [ "$total" -lt "$CHECKS_MIN" ]; then
  bad "only $total checks ran, expected at least $CHECKS_MIN - a case was skipped or a helper is missing"
fi
printf '\nspark/test_serve_external.sh: %d ok, %d bad, %d checks (floor %d) -> %s\n' \
  "$pass" "$fail" "$total" "$CHECKS_MIN" "$([ "$fail" = 0 ] && echo PASSED || echo FAILED)"
exit $([ "$fail" = 0 ] && echo 0 || echo 1)
