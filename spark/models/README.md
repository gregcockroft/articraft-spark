# Model files

One file per local model. `spark/serve.sh <key>` starts it; `spark/demo_dresser.sh <key>` runs the
dresser against it. `<key>` is the file name without `.env`.

To try another model, copy the closest file and change it:

| field | what it is |
|---|---|
| `MODEL_STATUS` | `tested` only once a result in `spark/results/` was measured with this exact file |
| `SERVE_IMAGE` | vLLM image, pinned by digest |
| `SERVE_MODEL`, `SERVE_REVISION` | Hugging Face repo id and the commit you served |
| `SERVE_NAME` | the name the server answers to; Articraft sends it as the model |
| `SERVE_MAX_MODEL_LEN`, `SERVE_GPU_UTIL`, `SERVE_MAX_SEQS` | vLLM capacity flags |
| `SERVE_TOOL_PARSER`, `SERVE_REASONING_PARSER` | must match the model's tool-call and thinking format, or tool calls arrive as text |
| `SERVE_SPECULATIVE` | vLLM `--speculative-config` JSON, empty for none |
| `SERVE_LIMIT_MM` | images allowed per request; keep it above `ARTICRAFT_OPENROUTER_MAX_IMAGES` |
| `ARTICRAFT_OPENROUTER_*` | the client side, documented in `docs/providers.md` |

The things that decide whether a model can drive Articraft at all are the tool parser (it has to
emit real tool calls) and the output cap (a reasoning model on a server that bounds nothing can
think for tens of thousands of tokens without acting).

## Bring your own cache

A clone of this repo is a few megabytes; the weights are not. `spark/cache.sh <key>` says exactly what a
given model file needs and whether this machine already has it, and downloads nothing unless you pass
`--fetch`:

```bash
spark/cache.sh qwen3.8-27b-inferact              # present / missing, with sizes; exit 0 if nothing to fetch
HF_CACHE=/mnt/models/huggingface spark/cache.sh qwen3.8-27b-inferact   # check a cache you already have
spark/cache.sh qwen3.8-27b-inferact --fetch      # download the missing pieces (hub only, no token)
```

`HF_CACHE` (default `~/.cache/huggingface`) is where the snapshots live, and `spark/serve.sh` reads the
same path: when the model file's pinned `SERVE_REVISION` is already there, the serve runs with
`HF_HUB_OFFLINE=1` and touches no network. Set `HF_HUB_OFFLINE` yourself and your value wins either way.

What each one costs, measured on a DGX Spark with `du -sbL` over the pinned snapshot. **Sizes are SI
(1 GB = 1,000,000,000 bytes), the same units a disk is sold in** - not GiB:

| key | weights | image |
|---|---|---|
| `qwen3.6-35b-a3b-nvfp4` | ~25 GB | the pinned `vllm/vllm-openai` digest, ~23 GB |
| `qwen3.8-27b-inferact` | ~26 GB | the same pinned digest |
| `qwen3.8-flash-next-nvfp4` | ~135 GB | **not served by `spark/serve.sh`** — it needs blazux/qwen3.8-Flash-DGX at its pin and the image built from it (~21 GB) |

The built image has no registry to pull from: build it in that clone, or move it between machines with
`docker save` and `docker load`.

**`qwen3.8-flash-next-nvfp4` leaves very little headroom on a 121 GiB Spark** — see the memory note at
the top of its `.env` file before running anything else alongside it.
