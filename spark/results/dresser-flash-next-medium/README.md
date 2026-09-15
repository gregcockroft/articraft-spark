# Dresser, Qwen3.8-Flash-Next at `reasoning_effort: medium`

Built on one DGX Spark on **2026-09-12** by `RadixArk/Qwen3.8-Flash-Next-NVFP4` (revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`), served by **blazux/qwen3.8-Flash-DGX** at
`bd60fcb1b492ca920f74df7462f05da7b6d98f73` — stock vLLM cannot fit this model on one 121 GB Spark, and that
patched build mmaps the n-gram table from NVMe so ~76 GiB sit on the card. Speculative decoding off. The client
is this fork's code with `spark/models/qwen3.8-flash-next-nvfp4.env` exactly as it ran (sha256
`f440c6d56e9f31eb6ff7c31f038b2f8541ed1bf66fbbf7a2620fee57d08adf8c`): thinking on, an 8,192-token thinking
budget, a 32,768-token output cap, and **`"reasoning_effort": "medium"`**. Input: the reference photo and
`spark/bench/dresser/prompt_demo.txt`, the one-paragraph description `spark/demo_dresser.sh` sends by default.

It ran **64 turns in 1 h 57 m** (7,037 s) and finished on its own final response, on the first clean `compile`
at turn 62. The thinking budget closed its reasoning four times and it acted each time. Effective **15.53 output
tok/s** against a decode median of 16.2, so **98 % of the wall was the model emitting tokens**.

**The sample is revision `0000`, the run's only revision — and `record.json` names that same file as the run's
result.** `result/usdz/0000.usdz`, `score.txt`, `handles.txt`, `sheet.png`, `stats.json`, `main_final_rev0000.py`.

Nine drawers on nine horizontal prismatic slides, all opening out of the front, a bar handle **26.5 mm proud**
of each, and **four stepped bracket feet** under the corners. Checked three ways:

| check | result |
|---|---|
| `spark/score.py` | **PASS** — 9/9 prismatic axes horizontal, 9/9 moving away from their parent |
| `spark/handles.py` | **PASS** — 9/9 drawers have a handle proud of the front, +26.5 mm |
| placement (`pxr`, ±5 mm) | **9/9 drawers inside the carcass**, carcass z `[-0.0, 0.900]`, drawers z `[0.073, 0.858]` |

`score.py` passing is not on its own enough — it will pass a dresser whose drawers hang below the carcass, which
is why the placement read is here beside it. **And none of the three looks at feet:** the other draw of this
pair passed all three identically while building two long side runners instead of four feet, which a person
spots in one glance at the sheet and no check here catches. That is why the last word is a human read.

## Why this draw, and why `medium`

**Why this draw.** It is one of a **registered pair** — two draws at `medium`, on one server instance with no
reload between them, both of which produced a coherent dresser (the bar asked for ≥ 1 of 2). This is the one
the human review rated **"very good"** on 2026-09-13; the other was rated **"good but feet missing at bottom"**, and the
code confirms it: this draw builds `bracket_foot` ×4, the other builds two full-depth side runners while its own
docstring still claims "four short rounded feet".

**Why `medium`.** At `reasoning_effort: low` this model ran **98 turns without once calling `compile`** and
produced nothing — it built the object in the workspace through the shell and never invoked the tool that emits
an artifact. And `low` is not the cheap setting: it spent **51 % more output tokens** (136,463) failing than
this draw spent succeeding (109,283).

## Reproducing it

`spark/cache.sh qwen3.8-flash-next-nvfp4` says what a serve needs and whether this machine has it; the weights
are **~135 GB** and the built image ~21 GB, and there is no registry to pull the image from — build it in the
blazux clone at its pin, or move it with `docker save` / `docker load`. See `spark/models/README.md`.

The prompt here is the detailed one. The short benchmark prompt (`spark/bench/dresser/prompt.txt`) gives
different results; do not compare this run with rows measured from that prompt.
