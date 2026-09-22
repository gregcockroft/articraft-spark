#!/usr/bin/env bash
# spark/render/gif.sh's exit codes, against a stub Blender, a stub python and a stub encoder. Nothing
# is rendered, no USD is read and no GPU is touched.
#
#   spark/test_gif.sh
#
# What it is for: gif.sh is called by demo_dresser.sh at the end of an hour-long run, and the way it
# fails is the only thing anyone will see of it. `pytest` covers no spark/*.sh.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/spark_test_gif.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); printf '  ok  %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  BAD %s\n' "$1"; }
eq()  { if [ "$2" = "$3" ]; then ok "$1: $2"; else bad "$1: got '$2', want '$3'"; fi; }
has() { if grep -qF -- "$2" "$3"; then ok "$1"; else bad "$1 (no '$2' in $3)"; fi; }
CHECKS_MIN=${CHECKS_MIN:-12}

FAKE=$TMP/render; mkdir -p "$FAKE" "$TMP/venv/bin" "$TMP/run"
cp "$HERE/render/gif.sh" "$FAKE/gif.sh"
: > "$FAKE/usdz2obj.py"; : > "$FAKE/turntable.py"; : > "$FAKE/gif.py"
echo '{"result": "result/usdz/0002.usdz"}' > "$TMP/run/record.json"
mkdir -p "$TMP/run/result/usdz"; echo stub > "$TMP/run/result/usdz/0002.usdz"

# The stub python is every python step at once: it prints the record's result for the -c probe, and
# otherwise does what the script's next step needs (an OBJ, or the GIF file itself).
cat > "$TMP/venv/bin/python" <<'EOF'
#!/bin/sh
case "$*" in
  *-c*)        echo "result/usdz/0002.usdz" ;;
  *usdz2obj*)  [ "${STUB_OBJ_RC:-0}" = 0 ] || exit "$STUB_OBJ_RC"; echo stub > "$3" ;;
  *gif.py*)    [ "${STUB_ENCODE_RC:-0}" = 0 ] || exit "$STUB_ENCODE_RC"; echo stub > "$3"; echo "GIF $3" ;;
esac
exit 0
EOF
# The stub Blender writes STUB_FRAMES frame files, which is what gif.sh counts before encoding.
cat > "$TMP/venv/bin/blender" <<'EOF'
#!/bin/sh
for a in "$@"; do case "$a" in */frames) out=$a ;; esac; done
mkdir -p "$out"
i=0; while [ "$i" -lt "${STUB_FRAMES:-150}" ]; do printf 'x' > "$out/frame_$i.png"; i=$((i + 1)); done
exit "${STUB_RENDER_RC:-0}"
EOF
chmod +x "$TMP/venv/bin/python" "$TMP/venv/bin/blender"

gif() {  # gif <out-file> -- env assignments...
  local out=$1; shift
  env PATH="$TMP/venv/bin:$PATH" ARTICRAFT_VENV="$TMP/venv" BLENDER="$TMP/venv/bin/blender" \
      RENDER_OUT="$TMP/out" "$@" "$FAKE/gif.sh" "$TMP/run" "$TMP/out.gif" > "$out" 2>&1
}

echo "=== the happy path: a run directory in, a GIF out ==="
out=$TMP/ok.txt; gif "$out"; eq "exit" "$?" 0
has "and it names the GIF it wrote" "GIF $TMP/out.gif" "$out"

echo "=== it cannot start: no Blender, no python, no USDZ, no record ==="
out=$TMP/noblender.txt
env -u BLENDER PATH="/usr/bin:/bin" ARTICRAFT_VENV="$TMP/venv" "$FAKE/gif.sh" "$TMP/run" > "$out" 2>&1
eq "no Blender: exit" "$?" 2
has "and says how to point at one" "set BLENDER=" "$out"
out=$TMP/nopython.txt
env PATH="$TMP/venv/bin:$PATH" ARTICRAFT_VENV="$TMP/nowhere" BLENDER="$TMP/venv/bin/blender" \
    "$FAKE/gif.sh" "$TMP/run" > "$out" 2>&1
eq "no python: exit" "$?" 2
has "and says to run install.sh" "spark/install.sh" "$out"
out=$TMP/nousdz.txt
env PATH="$TMP/venv/bin:$PATH" ARTICRAFT_VENV="$TMP/venv" BLENDER="$TMP/venv/bin/blender" \
    "$FAKE/gif.sh" "$TMP/nothing.usdz" > "$out" 2>&1
eq "no USDZ: exit" "$?" 2
out=$TMP/norecord.txt
mkdir -p "$TMP/empty"
env PATH="$TMP/venv/bin:$PATH" ARTICRAFT_VENV="$TMP/venv" BLENDER="$TMP/venv/bin/blender" \
    "$FAKE/gif.sh" "$TMP/empty" > "$out" 2>&1
eq "no record.json: exit" "$?" 2

echo "=== the render fails, or renders too few frames: exit 3, and the log is named ==="
out=$TMP/renderfail.txt; gif "$out" STUB_RENDER_RC=1; eq "Blender exit 1: exit" "$?" 3
has "and it points at the log" "turntable.log" "$out"
# The case that would otherwise pass silently: Blender exits 0 having written almost nothing. A GIF
# needs 60 frames, and an encoder handed 3 must not be allowed to invent the rest.
out=$TMP/fewframes.txt; gif "$out" STUB_FRAMES=3; eq "Blender exit 0 with 3 frames: exit" "$?" 3
has "and it says how many it got" "3 frames" "$out"

echo "=== the encode fails: exit 3, not 0 ==="
out=$TMP/encodefail.txt; gif "$out" STUB_ENCODE_RC=1; eq "encoder exit 1: exit" "$?" 3

echo "=== the exit codes are documented in the script ==="
for code in "0   the GIF was written" "2   it could not start" "3   Blender rendered no frames"; do
  has "the header states exit ${code%% *}" "$code" "$HERE/render/gif.sh"
done

total=$((pass + fail))
if [ "$total" -lt "$CHECKS_MIN" ]; then
  bad "only $total checks ran, expected at least $CHECKS_MIN - a case was skipped or a helper is missing"
fi
printf '\nspark/test_gif.sh: %d ok, %d bad, %d checks (floor %d) -> %s\n' \
  "$pass" "$fail" "$total" "$CHECKS_MIN" "$([ "$fail" = 0 ] && echo PASSED || echo FAILED)"
exit $([ "$fail" = 0 ] && echo 0 || echo 1)
