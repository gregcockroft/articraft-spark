# Security

## Reporting

Use GitHub's private vulnerability reporting on this repository
(**Security → Report a vulnerability**). Please do not open a public issue for something
exploitable.

If the problem is in Articraft itself rather than in this fork's `spark/` scripts, report it to
[articraftresearch/Articraft](https://github.com/articraftresearch/Articraft); this fork carries
their code and will pick up the fix.

## What this repository runs on your machine, and what that exposes

Worth reading before the first run — none of it is a defect, but all of it is a choice you are
making by running the scripts.

- **The model server has no authentication, and it is published on every interface.**
  `spark/serve.sh` runs `docker run -p "$PORT:8000"` (default `8001`), which binds `0.0.0.0`, so
  anyone who can reach the box on that port can send prompts to the model and spend the GPU. On a
  trusted LAN that is convenient and intended — it is how a second machine drives a run. On an
  untrusted network it is an open inference endpoint. To keep it local, set
  `ARTICRAFT_SERVE_PORT` and publish it on the loopback address instead, or put the box behind a
  firewall.

- **The agent executes model-written Python on your machine.** That is what Articraft is: the model
  writes `main.py` against the CAD SDK and the harness runs it, plus shell commands through its
  tools. The workspace is a directory, not a sandbox. Run it on a machine where that is acceptable,
  and read `spark/results/**/main_final_rev*.py` if you want to see the kind of code it produces.

- **Containers run with `--gpus all --ipc=host`.** Standard for vLLM, and weaker isolation than a
  default container.

- **Weights come from Hugging Face and are not pinned by content hash.** `spark/cache.sh` and
  `spark/serve.sh` pin a checkpoint *revision* where the model file sets `SERVE_REVISION`, which is
  the right granularity for reproducibility but is a trust decision about the publisher, not a
  verification of the bytes. The vLLM images are pinned by digest.

- **`spark/clone_and_demo.sh` is meant to be downloaded and run.** Read it first; it is 112 lines.
  It clones this repo, installs into a fresh venv, and refuses to fetch weights that are not already
  in the cache you point it at.

## Keys

There are none in this repository and none are needed for the local path — the server on `:8001`
takes no key. `.env.example` lists the hosted-provider variables for upstream's own use; `.env` is
git-ignored. If you add a hosted key, keep it there.
