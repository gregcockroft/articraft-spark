#!/usr/bin/env bash
# Build OpenUSD for aarch64 from source, so `pxr` can live in a normal uv venv with nothing
# installed outside it.
#
#   spark/build_openusd.sh                    # build, then report
#   USD_TAG=v26.05 spark/build_openusd.sh
#
# spark/install.sh calls this for you on aarch64; run it directly only to rebuild or pin a tag.
#
# WHY. `usd-core` publishes NO aarch64 wheel and NO sdist, at any version (checked across every release
# on 2026-09-13: only manylinux x86_64, macOS and win_amd64). That single fact is the whole reason this
# script exists - `pyproject.toml:42` excludes usd-core on aarch64 precisely because uv
# cannot install it. OpenUSD is portable C++, so the answer is to build it rather than to borrow a copy
# from somewhere else.
#
# WHAT IS BUILT, and it is much smaller than a default USD. The fork imports exactly ten pxr modules -
# Gf, Kind, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, UsdValidation - and NOT ONE imaging
# module, so Hydra, Hgi, usdview, OpenSubdiv, Ptex, OpenImageIO, OpenColorIO, MaterialX, Alembic and
# Draco are all off. What remains is USD core plus TBB, which build_usd.py fetches and builds itself.
#
# NO SUDO ANYWHERE. The one thing Ubuntu's system python lacks is headers (`Python.h`), which is the
# usual blocker; uv's managed CPython ships them, so that is the interpreter this builds against - and
# it is the same interpreter a uv venv will use, which is what makes the result importable.
set -uo pipefail

TAG=${USD_TAG:-v26.05}           # what pyproject asks for: usd-core>=26.5,<26.8. ubu24 installs the same.
ROOT=${USD_ROOT:-$HOME/.local/opt/openusd}
SRC=$ROOT/src
BUILD=$ROOT/build
INST=$ROOT/$TAG
JOBS=${USD_JOBS:-16}             # of 20: headroom, so the box stays usable while this runs
LOG=${USD_LOG:-$ROOT/build.log}

mkdir -p "$ROOT"
say() { echo "$(date '+%F %T %Z') [usd] $*" | tee -a "$LOG"; }

say "tag=$TAG jobs=$JOBS install=$INST"
say "cmake $(cmake --version | head -1 | awk '{print $3}')  g++ $(g++ -dumpversion)  cores $(nproc)"

# --- the interpreter: uv's managed CPython, because it has headers and the venv will use the same one --
UVPY=$(uv python find 3.12 2>/dev/null)
if [ -z "$UVPY" ] || ! [ -f "$(dirname "$(dirname "$UVPY")")/include/python3.12/Python.h" ]; then
  say "installing uv's managed CPython 3.12 (the system one has no headers)"
  uv python install 3.12 >>"$LOG" 2>&1
  UVPY=$(uv python find 3.12 2>/dev/null)
fi
PYROOT=$(dirname "$(dirname "$UVPY")")
say "python $UVPY"
[ -f "$PYROOT/include/python3.12/Python.h" ] || { say "ABORT: no Python.h under $PYROOT"; exit 2; }

# --- build tools that are not on the box, from PyPI rather than apt (no sudo) ----------------------
TOOLVENV=$ROOT/toolvenv
[ -x "$TOOLVENV/bin/ninja" ] || {
  say "creating the tool venv (ninja, packaging wheels)"
  uv venv --python "$UVPY" "$TOOLVENV" >>"$LOG" 2>&1
  uv pip install --python "$TOOLVENV/bin/python" ninja wheel setuptools >>"$LOG" 2>&1
}
export PATH=$TOOLVENV/bin:$PATH
say "ninja $("$TOOLVENV/bin/ninja" --version 2>/dev/null || echo MISSING)"

# --- the source ------------------------------------------------------------------------------------
if [ ! -d "$SRC/.git" ]; then
  say "cloning OpenUSD $TAG (shallow)"
  git clone --depth 1 --branch "$TAG" https://github.com/PixarAnimationStudios/OpenUSD "$SRC" >>"$LOG" 2>&1 \
    || { say "ABORT: clone failed"; exit 3; }
fi
say "source at $(git -C "$SRC" describe --tags --always)"

# --- the build -------------------------------------------------------------------------------------
# build_usd.py fetches and builds TBB itself. Everything optional is off; see the header for why each.
say "building - this is the long step; follow it with: tail -f $LOG"
t0=$SECONDS
"$UVPY" "$SRC/build_scripts/build_usd.py" \
    --build "$BUILD" --src "$SRC/deps" \
    --no-imaging --no-usdview --no-examples --no-tutorials --no-tools --no-docs --no-tests \
    --no-materialx --no-alembic --no-draco --no-openvdb --no-embree \
    --build-variant release --python --jobs "$JOBS" \
    "$INST" >>"$LOG" 2>&1
rc=$?
say "build_usd.py exit $rc after $(( (SECONDS - t0) / 60 )) min"
[ "$rc" = 0 ] || { say "FAILED - the last 40 lines of $LOG:"; tail -40 "$LOG"; exit "$rc"; }

# --- does it actually import, from a CLEAN venv with nothing else on the path? ---------------------
say "verifying from a fresh venv, with the environment scrubbed"
VERIFY=$ROOT/verifyvenv
rm -rf "$VERIFY"; uv venv --python "$UVPY" "$VERIFY" >>"$LOG" 2>&1
env -i PATH=/usr/bin:/bin HOME="$HOME" \
    PYTHONPATH="$INST/lib/python" LD_LIBRARY_PATH="$INST/lib" \
    "$VERIFY/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Gf, Sdf, Tf, Kind
print("  import: OK, USD", Usd.GetVersion())
st = Usd.Stage.CreateInMemory()
c = UsdGeom.Cube.Define(st, "/root/c")
UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
j = UsdPhysics.PrismaticJoint.Define(st, "/root/j")
j.CreateAxisAttr("X")
print("  stage:", len(list(st.Traverse())), "prims, rigid body + prismatic joint")
PY
say "done. install tree: $INST"
