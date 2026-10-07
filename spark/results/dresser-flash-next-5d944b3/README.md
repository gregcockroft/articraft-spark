# Dresser, Qwen3.8-Flash-Next on blazux/qwen3.8-Flash-DGX `5d944b3`

Built on one DGX Spark on **2026-10-04/05** by `RadixArk/Qwen3.8-Flash-Next-NVFP4` (revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`), served by **blazux/qwen3.8-Flash-DGX** at
`5d944b3a91bea0abc0bc3c657cbe137b970f8a44` — their image on the **vLLM v0.30.0 release**
(`vllm/vllm-openai:v0.30.0@sha256:8a69ffad…`), built locally as `qwen38-flash-dgx:5d944b3`, MODE `nvfp4`,
speculative decoding off. It is the measured result for that server pin, the way
`dresser-flash-next-medium/` is for the earlier one (`bd60fcb`, the vLLM preview image).

The client is this fork's code with `spark/models/qwen3.8-flash-next-nvfp4.env` as it ran (sha256
`2954982667c7cabe60e3b8f7fadfb0ae67cf8a397c3cd4cd846031fdc63be48d`; only `MODEL_STATUS` and comments have
changed since): thinking on, an 8,192-token thinking budget, a 32,768-token output cap,
`"reasoning_effort": "medium"`. Input: the reference photo and `spark/bench/dresser/prompt_demo.txt`, which is
`prompt.txt` here.

**It ran the README's own path.** `spark/clone_and_demo.sh` cloned the fork into an empty folder, installed
it, served the model with `spark/serve.sh`, and ran `spark/demo_dresser.sh` — exit 0, sheet and turntable
rendered.

It ran **88 turns in 2 h 50 m** (10,186 s of generation) and finished on its own **final response**. It called
`compile` twice, at turns **81** and **87**; both compiled cleanly, and the result is the second, revision
`0001` — `record.json` names the same file. 156,642 output tokens.

## Checks

| check | result |
|---|---|
| `spark/score.py` | **PASS** — 9/9 prismatic axes horizontal, 9/9 moving away from their parent |
| `spark/handles.py` | **PASS** — 9/9 drawers have a bar handle proud of the front, +27.5 mm |
| placement (`pxr`, ±5 mm) | **9/9 drawers inside the carcass**, carcass z `[0.0, 0.900]`, drawers z `[0.112, 0.862]` |

Nine drawers in three rows of three, fluted fronts with corner brackets, a top, a plinth and feet; opened, each
drawer is a box with a cavity (`sheet.png`, bottom row). **These checks are a floor:** they can show an object
is broken and cannot show it is good.

**A person did review it, and it has a visible defect the checks miss: seams across both sides.** The
carcass has full side panels (`side_panel_left`/`_right`, outer faces at x = ∓0.563), but `mid_rail_1`,
`mid_rail_2` and `bottom_rail` are built to the carcass's outer width, so their end faces are coplanar with
each side's outer face, and `bottom_board` and `back_panel` run 2 mm through it (world bounds read from the
USDZ with `pxr`). Where two faces share a plane the renderer cannot decide which to draw, so the turntable
shows a flickering band across the side at each rail's height — it looks like a drawer poking through, but the
drawers (all at |x| ≤ 0.551) are not involved. Each of those members should end at the side's inner face,
x = ∓0.546. It is the model's habit rather than this server pin: the same flaw appears in 7 of 31 dresser
draws recorded on the machine that built this one, across several models and settings.

## The log, and checking these numbers against it

```shell
python spark/verify_run.py spark/results/dresser-flash-next-5d944b3
```

re-derives turns, output and input tokens, the peak turn, and the compile calls from `conversation.jsonl`, the
turn-by-turn record the harness wrote; all six agree with `stats.json`. `spark/verify_results.sh` re-runs
`score.py` and `handles.py` on the committed USDZ and checks `SHA256SUMS`.

**One thing about the log is not verbatim.** Shell output and tracebacks carried local paths. They are
rewritten — the run directory to `/workspace/run`, the checkout to `/workspace/articraft`, the interpreter to
`/workspace/.venv`, the folders above them to `/workspace`, the username to `user` — and nothing else is
touched, so every token count and tool call is as written.

**What this sample does not have**, unlike `dresser-flash-next-medium/`: no server-side sampler ran, so there is
no decode-rate window, cache-hit figure or instrument check, and `wall_s` is generation time read from
`demo_dresser.sh`'s own log lines.

## Reproducing it

`spark/cache.sh qwen3.8-flash-next-nvfp4` says what a serve needs; the weights are **~135 GB** and the image is
built from blazux's clone at the pin on top of the pinned v0.30.0 base (a ~20 GB pull, once). One draw of a
model that, at `medium`, produced a coherent dresser in 2 of 3 earlier recorded draws: a rerun can differ.
