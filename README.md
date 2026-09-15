# Articraft on a DGX Spark

**A reference photo in, an articulated 3D asset out, on one NVIDIA DGX Spark with an open-weight
model. No API key and no cloud once the weights are downloaded.**

## Latest — 2026-09-12: a second local model built the dresser

**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) drove this loop on one Spark**, from
the same photo and the same one-paragraph prompt as the dresser below. It is served by
**[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)** — about 135 GB of weights in
NVFP4 do not fit on a 121 GB Spark under stock vLLM, and their patched build is what makes it run here.

![The dresser built by Qwen3.8-Flash-Next on a DGX Spark, turning while its nine drawers slide open one after another](spark/results/dresser_flash_next.gif)

*Qwen3.8-Flash-Next, 2026-09-11: **77 turns in 101 minutes, ended on its own final response** — and this is
that run's own last revision, not a good step picked out of one. Nine prismatic joints, every slide
horizontal and opening out of the front (`spark/score.py`: 9/9 and 9/9), every handle proud of its drawer
front (`spark/handles.py`: 9/9). The fronts read better than the dresser below's; the sides of neither are
right, with the drawer rails running out to the outside of the side panels on both.*

On that server the model was ready in 752 s and decoded at 15.95 tok/s, and no source change was needed to
drive it. **One run is not a result:** the draw beside it, at a lower reasoning effort, never called
`compile` in 98 turns and produced nothing at all. Testing continues, and this repo will carry what the
counts say — including the ones that go the other way.

[Articraft](https://github.com/articraftresearch/Articraft) is an agent that turns a prompt or a
reference photo into a posable 3D object. A language model writes Python against Articraft's CAD SDK;
Articraft compiles it, checks it, and exports a USDZ with rigid bodies and joints. Upstream expects a
frontier API behind that loop. This fork runs the same loop against a model served on the Spark itself:
[`RedHatAI/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4), a 35B
mixture-of-experts with 3B active parameters, quantized to NVFP4 for Blackwell and served by vLLM on the
GB10.

![A nine-drawer dresser built by Qwen3.6 on a DGX Spark from one photo, its drawers opening one after another](spark/results/dresser_demo.gif)

*Built on the Spark by the local Qwen3.6 through this fork, from the photo and the one-paragraph prompt in `spark/bench/dresser/` (see Results).*

## Why

This is one piece of **scenewright**, a pipeline on the same Spark that turns a photo or a prompt into
a Blender room (not public yet). Every object it reconstructs is a single scanned mesh, so a dresser
arrives as one frozen shape. Articraft builds the replacement: the same kind of object as rigid
bodies and joints. scenewright can swap one in by name from its Articraft library, which is how the
dresser above ends up in its living room with drawers that open. (In the video the dresser's facing was
set by hand; turning a swapped asset to face the room automatically is still open work in scenewright.)

## Run it

On a DGX Spark with Docker and the NVIDIA container toolkit:

```shell
spark/install.sh                             # uv venv: Python 3.12, this repo, and OpenUSD; re-runnable
spark/serve.sh qwen3.6-35b-a3b-nvfp4         # vLLM in Docker on :8001; the first start downloads 24 GB of weights
spark/demo_dresser.sh qwen3.6-35b-a3b-nvfp4  # build the nine-drawer dresser from one photo, score it, render it
```

The demo sends the one-paragraph prompt in [`spark/bench/dresser/prompt_demo.txt`](spark/bench/dresser/prompt_demo.txt)
and the photo beside it (`PROMPT=frozen` sends the short benchmark prompt instead), waits for the agent
(about an hour on a Spark), then prints a score and, with Blender available (`BLENDER=/path/to/blender`), writes `render/sheet.png` next to the run. Every input is
checked against `SHA256SUMS` first, so a changed prompt cannot pass for a better model.

## What had to change

A local model behind an OpenAI-compatible server behaves in ways a hosted API hides. Five changes,
each a commit on top of Articraft `ac7d688`:

| commit | the problem on the Spark | the change |
|---|---|---|
| Build on aarch64 Linux | `usd-core` publishes no aarch64 wheel and no sdist at any version, so Articraft would not install | `spark/build_openusd.sh` builds OpenUSD v26.05 from source in ~4 min (no imaging; no sudo) |
| Drive a local OpenAI-compatible server | three separate layers assumed openrouter.ai: the endpoint, a required API key, and a text-only CLI | a base URL, a key only where one is needed, and opt-in image input for vision models |
| Survive a local reasoning model | a thinking-only turn was a fatal error; a turn could generate ~49k tokens with no tool call; thinking could not be switched off; images piled up until the server refused the request | empty turns flow into the loop's normal handling; an output cap; chat-template arguments (`enable_thinking`); image-history pruning that always keeps the reference photo |
| Bound the thinking per turn | with thinking on, Qwen reasoned for the full 32k-token cap three turns running without a tool call, and the harness stopped the run; vLLM's own `thinking_token_budget` is accepted and ignored on this server | the client closes the reasoning after N tokens (8,192 here) and asks for the action, and the model acts on the plan it has |
| Keep the last clean compile at the turn limit | the local model keeps revising until the turn ceiling, and a run that had compiled a good revision recorded nothing | the last clean revision is kept and the record says it was salvaged (`max_turns_last_clean`) |

Every new setting is off by default, so nothing changes for hosted providers. The tests are in
`tests/test_openrouter_local.py`.

## Results

Measured on one DGX Spark (GB10, 128 GB unified memory) on 2026-09-10. Warm decode on this server is
about 27 tok/s at a 57k-token prompt.

### The demo run

![The dresser from the demo run, closed and with every drawer open, from four sides](spark/results/dresser-current-harness-thinking-budget/sheet.png)

This fork, thinking on with the 8,192-token budget, the one-paragraph prompt: **67 turns in 53 minutes,
finished on its own final response.** Revision `0002` has nine prismatic joints, every slide horizontal and
opening out of the front (`spark/score.py`: 9/9 and 9/9), and every handle standing proud of its drawer
front (`spark/handles.py`: 9/9). The run's last revision, `0004`, thickened the fronts past the handles, so
`0002` is the sample; [its directory](spark/results/dresser-current-harness-thinking-budget/) has the details.

### The benchmark prompt

The short prompt in `spark/bench/dresser/prompt.txt` is the one every earlier run was given, and the one the
rows below share. Do not compare them with the demo run above: the prompts differ.

**The joint count is not the object.** Articraft's checks and the bench's joint ask both passed on
objects whose parts do not fit together. With thinking switched off, the local model is fast (turns of
10 to 20 seconds) and hits the ask on three of five objects, but none of the five holds together: in the
dresser the drawers slide vertically and a side panel sits in the middle of the carcass. So
[`spark/score.py`](spark/score.py) also checks every drawer in world space: its slide axis has to be
horizontal and opening it has to move it out of the carcass, not through it.

![The same prompt, two harness versions: a coherent chest, and one that passes the joint count with its drawers sliding vertically](spark/results/dresser_review_pinned_vs_current.png)

| dresser run | harness | thinking | turns | wall | joints (ask: 9 prismatic) | score.py geometry |
|---|---|---|---|---|---|---|
| [earlier Articraft](spark/results/dresser-pinned-harness-thinking-on/) | `mattzh72/articraft` `959f1455` | on | 65, finished on its own | 50 min | 9 prismatic, met | **pass**: 9/9 horizontal, 9/9 open outward |
| [current, thinking off](spark/results/dresser-current-harness-thinking-off/) | this fork | off | 100, salvaged | 27 min | 9 prismatic, met | **fail**: 0/9 horizontal |
| current, thinking on | this fork | on | 4, stopped by the harness | 36 min | never compiled | none |
| current, thinking on, budget | this fork | on, 8,192 budget | 60, finished on its own | 22 min | 9 joints, 0 prismatic, not met | none to check |

With thinking on and no budget, turns 2, 3 and 4 each reasoned for the full 32,768-token cap without calling
a tool, and after three empty turns in a row the harness stopped the run. With the budget the runaway is
gone: the run finished on its own, but from the short prompt the drawers came out as removable parts rather
than slides. The demo prompt, which names the slides, the handles and every carcass part, is what produced
the dresser above.

## Try another model

Each model is one file in [`spark/models/`](spark/models/): the vLLM image digest, the checkpoint and
revision, the tool-call and reasoning parsers, and the client settings Articraft sends. To try one:

```shell
cp spark/models/qwen3.6-35b-a3b-nvfp4.env spark/models/my-model.env   # edit it
spark/serve.sh my-model && spark/demo_dresser.sh my-model
```

`qwen3.8-27b.env` is there as the next slot and is marked untested. What decides whether a model can
drive Articraft at all is whether its tool calls parse, and whether it acts before its output cap.

## How it fits together

```mermaid
flowchart LR
  photo[reference photo + prompt] --> agent
  subgraph spark[DGX Spark]
    agent[Articraft agent loop] -- OpenAI chat API --> vllm[vLLM on GB10<br/>Qwen3.6-35B-A3B NVFP4]
    vllm -- tool calls --> agent
    agent -- writes main.py --> sdk[Articraft CAD SDK<br/>build123d]
    sdk --> compile[compile + checks]
    compile -- errors, previews --> agent
    compile --> usdz[USDZ: rigid bodies + joints]
    usdz --> score[spark/score.py<br/>joints + where drawers move]
    usdz --> render[spark/render<br/>Blender sheet and video]
  end
```

## Limits

- **Thinking needs a budget.** Off, the model is quick and the geometry did not hold together in these runs.
  On with no budget, it reasoned to the 32k-token cap three turns running without acting. On with an 8,192-token
  budget, it thinks and then acts. vLLM's own `thinking_token_budget` has no effect on this server (MTP-2), so
  the budget is enforced by the client.
- **The prompt carries a lot.** The same model on the same code built nine sliding drawers from the demo prompt
  and none from the short one. Both prompts are in `spark/bench/dresser/`.
- **The last clean revision is not always the best.** In the demo run the final revision moved the drawer fronts
  past the handles; the checks in `spark/` are how the good revision was picked.
- **The geometry checks are narrow.** `score.py` checks sliding joints and `handles.py` checks handles. Hinges,
  and whether the object resembles the photo, still need a person looking at the render.

## Credits

Articraft is by its authors at [articraftresearch/Articraft](https://github.com/articraftresearch/Articraft),
Apache-2.0; its README is kept as [ARTICRAFT.md](ARTICRAFT.md), and [NOTICE](NOTICE) lists what this
fork changed. Qwen3.6 and Qwen3.8-Flash-Next are by the Qwen team; their NVFP4 checkpoints are by
[Red Hat AI](https://huggingface.co/RedHatAI) and [RadixArk](https://huggingface.co/RadixArk);
serving is [vLLM](https://github.com/vllm-project/vllm), and fitting Flash-Next onto one Spark is
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX).
