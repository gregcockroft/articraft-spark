#!/usr/bin/env bash
# Re-derive every claim in spark/results/ from the committed USDZ files.
#
# No GPU, no model, no server, no network - it reads geometry out of files that are in this
# repository and runs the same two scripts the README quotes. If the numbers in a sample's
# README.md are wrong, this says so. About ten seconds.
#
#     spark/verify_results.sh                  # every sample
#     spark/verify_results.sh dresser-flash-next-medium
#
# Exit 0 when every check passes. Needs the project venv (spark/install.sh), because pxr reads USD.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || { echo "no $PY - run spark/install.sh first, or set PYTHON=" >&2; exit 2; }

fail=0
pass=0
note=0

# Two lines of a committed score.txt are provenance rather than geometry: `file`, which names the
# path the file was produced at (not this checkout), and `record`, which is present only when the
# scorer was pointed at a run directory. Both are dropped on both sides; everything else - every
# joint, every axis, every verdict - has to match exactly.
strip_provenance() { grep -vE '^(file|record) +' ; }

for dir in spark/results/*/; do
  name=$(basename "$dir")
  [ $# -eq 0 ] || [ "$name" = "${1:-}" ] || continue
  usdz=$(ls "$dir"result/usdz/*.usdz 2>/dev/null | head -1)
  echo "== $name"

  if [ -f "$dir/SHA256SUMS" ]; then
    if ( cd "$dir" && sha256sum -c SHA256SUMS >/dev/null 2>&1 ); then
      echo "   sha256   every committed file matches  OK"
      pass=$((pass + 1))
    else
      echo "   sha256   MISMATCH:"
      ( cd "$dir" && sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$' | sed 's/^/            /' )
      fail=$((fail + 1))
    fi
  else
    echo "   sha256   no SHA256SUMS in this sample  (not checked)"
    note=$((note + 1))
  fi

  if [ -z "$usdz" ]; then
    echo "   score    no USDZ committed for this sample  (not checked)"
    note=$((note + 1))
    continue
  fi

  for check in score handles; do
    [ -f "$dir/$check.txt" ] || { echo "   $check    no $check.txt  (not checked)"; note=$((note + 1)); continue; }
    if [ "$check" = score ]; then
      got=$("$PY" spark/score.py "$usdz" --ask spark/bench/dresser/ask.json 2>&1)
    else
      got=$("$PY" spark/handles.py "$usdz" 2>&1)
    fi
    if diff -q <(printf '%s\n' "$got" | strip_provenance) <(strip_provenance < "$dir/$check.txt") >/dev/null; then
      echo "   $check    re-run reproduces $check.txt  OK"
      pass=$((pass + 1))
    else
      echo "   $check    re-run does NOT reproduce $check.txt:"
      diff <(strip_provenance < "$dir/$check.txt") <(printf '%s\n' "$got" | strip_provenance) | sed 's/^/            /' | head -20
      fail=$((fail + 1))
    fi
  done

  # Turn counts and token counts, re-derived from the run's own log where one is published.
  if [ -f "$dir/conversation.jsonl" ]; then
    if out=$("$PY" spark/verify_run.py "$dir" 2>&1); then
      echo "   run      every number in stats.json re-derived from conversation.jsonl  OK"
      pass=$((pass + 1))
    else
      echo "   run      conversation.jsonl does NOT agree with stats.json:"
      printf '%s\n' "$out" | sed 's/^/            /'
      fail=$((fail + 1))
    fi
  else
    echo "   run      no conversation.jsonl in this sample  (not checked)"
    note=$((note + 1))
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "$pass checks passed, $note not applicable. Every published number came back the same."
else
  echo "$pass passed, $fail FAILED, $note not applicable."
fi
exit $(( fail > 0 ))
