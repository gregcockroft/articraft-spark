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
