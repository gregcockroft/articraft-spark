#!/usr/bin/env bash
# Install Articraft into its own micromamba environment. Written for an NVIDIA DGX Spark (aarch64);
# works on x86_64 Linux too.
#
#   spark/install.sh            # env "articraft" under ~/micromamba
#   ARTICRAFT_ENV=other spark/install.sh
#
# Why conda at all: usd-core publishes no aarch64 Linux wheel, so on the Spark OpenUSD (pxr) comes
# from conda-forge's `openusd`. Everything else is a normal pip install of this repository.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_NAME=${ARTICRAFT_ENV:-articraft}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
ARCH=$(uname -m)

for tool in curl gcc g++; do
  command -v "$tool" >/dev/null || { echo "missing $tool (Ubuntu: sudo apt install build-essential curl)" >&2; exit 2; }
done

if command -v micromamba >/dev/null; then MM=$(command -v micromamba)
elif [ -x "$HOME/.local/bin/micromamba" ]; then MM=$HOME/.local/bin/micromamba
else
  case $ARCH in aarch64) PLAT=linux-aarch64 ;; x86_64) PLAT=linux-64 ;; *) echo "unsupported arch $ARCH" >&2; exit 2 ;; esac
  echo "installing micromamba to ~/.local/bin"
  mkdir -p "$HOME/.local"
  curl -Ls "https://micro.mamba.pm/api/micromamba/$PLAT/latest" | tar -xj -C "$HOME/.local" bin/micromamba
  MM=$HOME/.local/bin/micromamba
fi

PKGS=(python=3.12)
[ "$ARCH" = aarch64 ] && PKGS+=(openusd=26.08)   # x86_64 gets usd-core from PyPI instead
if [ -x "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin/python" ]; then
  echo "env $ENV_NAME exists, reusing it"
else
  "$MM" create -y -n "$ENV_NAME" -c conda-forge "${PKGS[@]}"
fi
PY=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin/python
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -e "$ROOT"

"$PY" - <<'PYEOF'
from pxr import Usd
import articraft
print("OpenUSD", ".".join(map(str, Usd.GetVersion())), "| articraft importable")
PYEOF
echo
echo "installed: $MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin/articraft"
command -v docker >/dev/null || echo "note: docker is not installed; spark/serve.sh needs it (with the NVIDIA container toolkit)"
