#!/usr/bin/env bash
# Install Articraft into a normal virtual environment with uv. Written for an NVIDIA DGX Spark
# (aarch64); works on x86_64 Linux too.
#
#   spark/install.sh                 # .venv in the repo
#   ARTICRAFT_VENV=/path/to/venv spark/install.sh
#
# Everything is installed with uv. On x86_64 everything comes from PyPI, OpenUSD included. On aarch64 there is
# one gap and one only: `usd-core` publishes no aarch64 wheel and no sdist at ANY version, so there is
# nothing for uv to install. OpenUSD is portable C++, so this builds it - about four minutes, because
# Articraft imports ten pxr modules and no imaging module at all, so Hydra, usdview, OpenSubdiv,
# OpenImageIO, MaterialX, Alembic, Draco, OpenVDB and Embree are all off.
#
# No sudo. The usual blocker on Ubuntu is that the system python ships no headers; uv's managed CPython
# does, and it is the same interpreter the venv uses, which is what makes the built module importable.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${ARTICRAFT_VENV:-$ROOT/.venv}
ARCH=$(uname -m)
USD_PREFIX=${USD_PREFIX:-$HOME/.local/opt/openusd}
USD_TAG=${USD_TAG:-v26.05}   # inside the usd-core range pyproject pins, so both arches run one version

for tool in curl gcc g++; do
  command -v "$tool" >/dev/null || { echo "missing $tool (Ubuntu: sudo apt install build-essential curl)" >&2; exit 2; }
done

if command -v uv >/dev/null; then UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then UV=$HOME/.local/bin/uv
else
  echo "installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV=$HOME/.local/bin/uv
fi

# --- the venv and every Python dependency ----------------------------------------------------------
# Running this script twice is a no-op here, not an error: `uv venv` exits 2 on a venv that already
# exists ("A virtual environment already exists at: .venv"), and under `set -e` that aborted the whole
# install before `uv sync` ever ran. Reuse the venv and let the sync below reconcile its dependencies.
if [ -f "$VENV/pyvenv.cfg" ]; then
  echo "reusing the virtual environment at $VENV"
else
  "$UV" venv --python 3.12 "$VENV"
fi
(cd "$ROOT" && VIRTUAL_ENV=$VENV "$UV" sync --group dev --active)

# --- OpenUSD: from PyPI on x86_64, built here on aarch64 -------------------------------------------
if [ "$ARCH" = aarch64 ]; then
  INST=$USD_PREFIX/$USD_TAG
  if [ -d "$INST/lib/python/pxr" ]; then
    echo "OpenUSD $USD_TAG already built at $INST"
  else
    command -v cmake >/dev/null || { echo "missing cmake (Ubuntu: sudo apt install cmake)" >&2; exit 2; }
    echo "building OpenUSD $USD_TAG for aarch64 - no wheel exists for it, so it is built (~4 min)"
    USD_ROOT=$USD_PREFIX USD_TAG=$USD_TAG bash "$ROOT/spark/build_openusd.sh"
  fi
  # ONE path entry to an installed library - the standard venv mechanism, not a copied library. The
  # install tree is self-locating through RUNPATH, so no LD_LIBRARY_PATH is needed.
  SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  echo "$INST/lib/python" > "$SITE/openusd.pth"
  echo "OpenUSD on the venv's path via $SITE/openusd.pth"
fi

"$VENV/bin/python" - <<'PYEOF'
from pxr import Usd
import articraft
print("OpenUSD", ".".join(map(str, Usd.GetVersion())), "| articraft importable")
PYEOF
echo
echo "installed: $VENV/bin/articraft"
command -v docker >/dev/null || echo "note: docker is not installed; spark/serve.sh needs it (with the NVIDIA container toolkit)"
