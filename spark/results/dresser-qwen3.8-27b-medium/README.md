# Dresser, Qwen3.8-27B (Inferact NVFP4) at `reasoning_effort: medium`

Built on one DGX Spark on **2026-09-12** by `Inferact/Qwen3.8-27B-NVFP4` (revision
`6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`), served by the pinned `vllm/vllm-openai` digest
`sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`. Speculative decoding off — MTP-2 was
measured at 1.34× against a registered 1.5× band on this box and rejected. The client is this fork's code with
`spark/models/qwen3.8-27b-inferact.env` exactly as it ran (sha256
`fdb366b11158ddb2ec2468021162c12b2d47877f9dcb2fcb07b13f4d06fd753a`): thinking on, an 8,192-token thinking
budget, a 32,768-token output cap, **`"reasoning_effort": "medium"`**. Input: the reference photo and
`spark/bench/dresser/prompt_demo.txt`.

It ran **39 turns in 1 h 34 m** (5,646 s) and finished on its own final response. It called `compile` three
times — turns 33 and 35 errored, turn 37 was clean. The thinking budget closed its reasoning once. Effective
**8.07 output tok/s** against a decode median of 8.4, so **98 % of the wall was the model emitting tokens**.

**The sample is revision `0002`, and `record.json` names that same file as the run's result.**
`result/usdz/0002.usdz`, `score.txt`, `handles.txt`, `sheet.png`, `stats.json`, `main_final_rev0002.py`.

| check | result |
|---|---|
| `spark/score.py` | **PASS** — 9/9 prismatic axes horizontal, 9/9 moving away from their parent (all three revisions) |
| `spark/handles.py` | **PASS** — 9/9 drawers have a handle proud of the front (all three revisions) |
| placement (`pxr`, ±5 mm) | **9/9 drawers inside the carcass**, carcass z `[0.0, 0.900]`, drawers z `[0.128, 0.872]` |

**The honest caveat, and it is the reason the sheet is here.** The human review of this sheet, 2026-09-13:
**"good but top and sides have underlying geometry slightly pushing through"**. Every automated check above
passes; the interpenetration is not something any of them looks at. **The second draw of this pair was read as
"poor — missing cavity on drawers", and it passed all three checks identically.** So this directory is one
coherent object, not a claim that the model produces them reliably.

## What this model is for, next to the other sample here

It is **half the decode speed** of Qwen3.8-Flash-Next (8.4 against 16.2 tok/s) and **finished in less time**:
45,547 output tokens over 39 turns, against Flash-Next's 109,283 over 64. Run time here is output tokens ÷
decode rate, and this model spends far fewer tokens. It also needs **~26 GB of weights against Flash-Next's
~135 GB**, and it runs on the **pinned upstream vLLM image with no patched build** — which is the practical
difference for anyone reproducing this from a clean pull.

## Reproducing it

`spark/cache.sh qwen3.8-27b-inferact` says what a serve needs and whether this machine has it; the weights are
**~26 GB** and the pinned image ~23 GB. `spark/serve.sh qwen3.8-27b-inferact` runs offline when the snapshot is
already in `HF_CACHE`. See `spark/models/README.md`.

The prompt here is the detailed one. The short benchmark prompt (`spark/bench/dresser/prompt.txt`) gives
different results; do not compare this run with rows measured from that prompt.
