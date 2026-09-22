#!/usr/bin/env bash
# spark/demo_dresser.sh's exit code, against a stub model, a stub scorer and a stub renderer. No model
# is called, no container is contacted, nothing is rendered and nothing is downloaded.
#
#   spark/test_demo_dresser.sh
#
# What it is for: the script promises a sheet in its own documentation and in the README's one-liner
# ("build the nine-drawer dresser from one photo, score it, render it"), and it used to call the
# renderer, throw the status away, and exit with the scorer's verdict - so a failed render reported
# success. `pytest` covers no spark/*.sh, so without this file nothing would notice that coming back.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/spark_test_demo.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); printf '  ok  %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  BAD %s\n' "$1"; }
eq()  { if [ "$2" = "$3" ]; then ok "$1: $2"; else bad "$1: got '$2', want '$3'"; fi; }
has() { if grep -qF -- "$2" "$3"; then ok "$1"; else bad "$1 (no '$2' in $3)"; fi; }
hasnt() { if grep -qF -- "$2" "$3"; then bad "$1 (found '$2' in $3)"; else ok "$1"; fi; }
CHECKS_MIN=${CHECKS_MIN:-24}   # 27 on a box with no blender on PATH; the skip case cannot run everywhere

# --- a tree shaped like spark/, with everything the script touches stubbed -------------------------
FAKE=$TMP/spark; mkdir -p "$FAKE/models" "$FAKE/render" "$FAKE/bench/dresser" "$TMP/venv/bin" "$TMP/out"
cp "$HERE/demo_dresser.sh" "$FAKE/demo_dresser.sh"
# A model file with only what this script reads.
cat > "$FAKE/models/stub.env" <<'EOF'
MODEL_STATUS=tested
SERVE_NAME=stub-model
ARTICRAFT_OPENROUTER_MAX_OUTPUT_TOKENS=32768
EOF
# The frozen-input check is real: make real files and a real SHA256SUMS over them.
for f in prompt.txt prompt_demo.txt reference.png ask.json; do echo "stub $f" > "$FAKE/bench/dresser/$f"; done
( cd "$FAKE/bench/dresser" && sha256sum prompt.txt prompt_demo.txt reference.png ask.json > SHA256SUMS )
: > "$FAKE/score.py"
# The stub model writes a run directory, because the script picks the newest one under OUT.
cat > "$TMP/venv/bin/articraft" <<'EOF'
#!/bin/sh
d="$TMP_OUT/20260916-000000-stub"; mkdir -p "$d"; echo '{"status":"success"}' > "$d/record.json"; exit 0
EOF
# The stub python IS the scorer: it exits with the verdict the case asks for.
printf '#!/bin/sh\necho "RESULT stub"\nexit ${STUB_SCORE_RC:-0}\n' > "$TMP/venv/bin/python"
printf '#!/bin/sh\necho "render stub"\nexit ${STUB_RENDER_RC:-0}\n' > "$FAKE/render/render.sh"
printf '#!/bin/sh\necho "gif stub"\nexit ${STUB_GIF_RC:-0}\n' > "$FAKE/render/gif.sh"
printf '#!/bin/sh\nexit 0\n' > "$TMP/venv/bin/curl"     # the "is a server there" probe
chmod +x "$TMP/venv/bin/articraft" "$TMP/venv/bin/python" "$FAKE/render/render.sh" \
  "$FAKE/render/gif.sh" "$TMP/venv/bin/curl"

demo() {  # demo <out-file> -- env assignments...
  local out=$1; shift
  env PATH="$TMP/venv/bin:$PATH" TMP_OUT="$TMP/out" ARTICRAFT_VENV="$TMP/venv" OUT="$TMP/out" \
      BLENDER=/bin/true "$@" "$FAKE/demo_dresser.sh" stub > "$out" 2>&1
}

echo "=== the render succeeds: the exit code is the scorer's verdict, exactly as before ==="
out=$TMP/pass.txt
demo "$out" STUB_SCORE_RC=0 STUB_RENDER_RC=0
eq "score PASS, render ok" "$?" 0
hasnt "and it says nothing about a failed render" "RENDER FAILED" "$out"
out=$TMP/scorefail.txt
demo "$out" STUB_SCORE_RC=1 STUB_RENDER_RC=0
eq "score FAIL, render ok: the verdict is the exit code" "$?" 1
hasnt "still nothing about the render" "RENDER FAILED" "$out"

echo "=== the render fails on a PASSING object: exit 4, and a loud line ==="
# This is the case the script used to report as success.
out=$TMP/renderfail.txt
demo "$out" STUB_SCORE_RC=0 STUB_RENDER_RC=1
eq "score PASS, render FAILED" "$?" 4
has "it says the render failed, with the status" "RENDER FAILED (exit 1)" "$out"
has "and which run has no sheet" "no sheet was written for" "$out"
has "and why the code is 4" "exits 4 to say the sheet is missing" "$out"

echo "=== the render fails on a FAILING object: the verdict keeps the code, the line still prints ==="
out=$TMP/both.txt
demo "$out" STUB_SCORE_RC=3 STUB_RENDER_RC=2
eq "score FAIL, render FAILED" "$?" 3
has "the loud line prints anyway" "RENDER FAILED (exit 2)" "$out"
has "and says why the code is not 4" "the exit code stays 3" "$out"

echo "=== the GIF fails on a PASSING object with a good sheet: exit 5, and a loud line ==="
# The same trap the render used to fall into, one output along: the GIF is what the README shows.
out=$TMP/giffail.txt
demo "$out" STUB_SCORE_RC=0 STUB_RENDER_RC=0 STUB_GIF_RC=1
eq "score PASS, sheet ok, GIF FAILED" "$?" 5
has "it says the GIF failed, with the status" "GIF FAILED (exit 1)" "$out"
has "and why the code is 5" "exits 5 to say the GIF is missing" "$out"

echo "=== a failing object keeps its own code even when the GIF fails too ==="
out=$TMP/gifboth.txt
demo "$out" STUB_SCORE_RC=3 STUB_RENDER_RC=0 STUB_GIF_RC=2
eq "score FAIL, GIF FAILED" "$?" 3
has "the loud line prints anyway" "GIF FAILED (exit 2)" "$out"

echo "=== GIF=0 skips it: the GIF is minutes of Cycles on top of an hour-long run ==="
out=$TMP/gifoff.txt
demo "$out" STUB_SCORE_RC=0 STUB_RENDER_RC=0 GIF=0
eq "GIF=0, everything else fine" "$?" 0
hasnt "and the GIF script was never called" "gif stub" "$out"
out=$TMP/gifon.txt
demo "$out" STUB_SCORE_RC=0 STUB_RENDER_RC=0
has "while the default does call it" "gif stub" "$out"

echo "=== the exit codes are documented in the script, which is why they were missable before ==="
for code in "0   the object was built" "2   this script could not start" "3   the frozen dresser inputs" "4   the object scored PASS" "5   the object scored PASS and the sheet is there"; do
  has "the header states exit ${code%% *}" "$code" "$HERE/demo_dresser.sh"
done

echo "=== with no Blender at all it is not a failure, it is a skip ==="
# BLENDER unset AND no blender on PATH. A box that has one in /usr/bin cannot run this case at all, and
# a case that cannot run must say so rather than quietly asserting something else: the first version of
# this check passed a stub renderer through the Blender branch on this box and called it a skip.
out=$TMP/noblender.txt
if PATH="$TMP/venv/bin:/usr/bin:/bin" command -v blender >/dev/null 2>&1; then
  echo "  --  skipped: this box has blender at $(PATH="$TMP/venv/bin:/usr/bin:/bin" command -v blender), so the no-Blender branch cannot be reached here"
else
  env -u BLENDER PATH="$TMP/venv/bin:/usr/bin:/bin" TMP_OUT="$TMP/out" ARTICRAFT_VENV="$TMP/venv" \
      OUT="$TMP/out" STUB_SCORE_RC=0 "$FAKE/demo_dresser.sh" stub > "$out" 2>&1
  eq "no Blender: exit 0" "$?" 0
  has "and it says the render was skipped" "skipping the render" "$out"
  hasnt "and the renderer was never called" "render stub" "$out"
  hasnt "and neither was the GIF" "gif stub" "$out"
fi

total=$((pass + fail))
if [ "$total" -lt "$CHECKS_MIN" ]; then
  bad "only $total checks ran, expected at least $CHECKS_MIN - a case was skipped or a helper is missing"
fi
printf '\nspark/test_demo_dresser.sh: %d ok, %d bad, %d checks (floor %d) -> %s\n' \
  "$pass" "$fail" "$total" "$CHECKS_MIN" "$([ "$fail" = 0 ] && echo PASSED || echo FAILED)"
exit $([ "$fail" = 0 ] && echo 0 || echo 1)
