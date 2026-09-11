# Dresser, current Articraft, Qwen3.6 with thinking on and a thinking budget

Built on one DGX Spark on 2026-09-10 by `RedHatAI/Qwen3.6-35B-A3B-NVFP4` served by vLLM, through this
fork's code (identical to commit `0b0c903`), with `spark/models/qwen3.6-35b-a3b-nvfp4.env` as it stands:
thinking on, an 8,192-token thinking budget per turn, a 32,768-token output cap. Input: the reference photo
and `spark/bench/dresser/prompt_demo.txt`, the one-paragraph description `spark/demo_dresser.sh` sends by
default. It ran 67 turns in 52 min 56 s and finished on its own final response; the thinking budget closed
the model's reasoning three times, and each time the model then acted.

**The sample is revision `0002`**, the clean compile at turn 49: `result/usdz/0002.usdz`, `score.txt`,
`sheet.png`. Nine drawers on nine horizontal prismatic slides, all opening out of the front, fluted fronts
with a bar handle standing proud of each. `record.json` names revision `0004` as the run's result, the last
clean compile: by then the model had thickened the drawer fronts and left the handles where they were, so
they sit 2.5 mm inside the fronts. That is why `0002` is the one shown.

The prompt here is the detailed one. The short benchmark prompt (`prompt.txt`) gives different results;
see the table in the top-level README, and do not compare this run with that table's rows.
