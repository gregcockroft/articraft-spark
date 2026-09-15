# Contributing

This is a fork of [Articraft](https://github.com/articraftresearch/Articraft) that runs its agent
loop against an open-weight model served on an NVIDIA DGX Spark. Issues and pull requests are
welcome.

## Where a change belongs

**If it is about Articraft itself** — the CAD SDK, the compiler, the agent loop, a provider — please
open it against [articraftresearch/Articraft](https://github.com/articraftresearch/Articraft). That
is the project; this is a fork with a narrow purpose, and a fix that lands upstream helps more
people than one that lands here. Several of the changes in [NOTICE](NOTICE) are of that kind and are
being offered upstream.

**If it is about running on a Spark** — anything in `spark/`, the install and serve path, the
scoring scripts, a new model file, the README — this is the right repo.

## Reporting a result that disagrees with ours

This is the most useful kind of issue. The numbers in the README come from a small number of runs on
one machine, and we would rather know where they do not hold. Please include:

- the model file you used (`spark/models/<key>.env`) and whether you changed it,
- which prompt (`prompt.txt` or `prompt_demo.txt` — they are not comparable),
- the run's `record.json` and, if you have it, `stats.json`,
- `spark/score.py` and `spark/handles.py` output on the result,
- your box: GPU, unified memory, vLLM image.

`spark/verify_results.sh` re-runs the checks on the samples committed here; if that disagrees with
the committed `score.txt` on your machine, that alone is worth an issue.

## Adding a model

Each model is one file in `spark/models/`. Copy the closest one, edit it, and run it:

```shell
cp spark/models/qwen3.6-35b-a3b-nvfp4.env spark/models/my-model.env
spark/serve.sh my-model && spark/demo_dresser.sh my-model
```

Keep `MODEL_STATUS=untested` until you have driven a run through it end to end, and pin
`SERVE_REVISION` to the checkpoint revision you actually ran. A model file whose `MODEL_STATUS` says
`tested` is a claim that these exact bytes produced an object, so say which run in the comment.

## Checks

CI runs on every push and pull request (`.github/workflows/ci.yml`): repository hygiene,
`ruff format --check`, `ruff check`, `basedpyright`, and `pytest -m 'not live and not volume'` —
all four scoped to `src/articraft` and `tests`.

Two honest limits, so you know what green does and does not mean:

- **CI runs on x86 Linux.** The aarch64 GB10 path — `spark/install.sh`, the OpenUSD source build,
  `spark/serve.sh`, anything touching the GPU — is not exercised by any automated check. It is
  tested by running it.
- **`spark/` is not covered by CI.** If you change a script there, say in the PR what you ran.

Locally, on a Spark:

```shell
spark/install.sh
env -u FORCE_COLOR .venv/bin/python -m pytest tests/ -q -m 'not live'
.venv/bin/python -m ruff check src/articraft tests
spark/verify_results.sh            # no GPU, no model, no network
```

## Style

Follow [AGENTS.md](AGENTS.md), which is upstream's and still applies: narrow changes, no broad
fallback machinery, and a test for anything that changed behaviour. New settings default to
upstream's behaviour so that nothing changes for hosted providers.

`spark/results/` holds recorded runs. The Python in there was written by a model and is kept as
evidence — do not reformat or lint-fix it.
