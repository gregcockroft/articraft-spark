# The dresser bench input

Four files, and their hashes in `SHA256SUMS`. `spark/demo_dresser.sh` checks every one of them
before it starts, so a changed prompt or a swapped photo cannot pass for a better model.

| file | what it is |
|---|---|
| `reference.png` | the photograph the model is shown |
| `prompt_demo.txt` | the one-paragraph description `demo_dresser.sh` sends by default |
| `prompt.txt` | the short benchmark prompt (`PROMPT=frozen`) |
| `ask.json` | the joint count the object is scored against: 9 prismatic |

**The two prompts are not comparable.** `prompt_demo.txt` names the slides, the handles and every
carcass part; `prompt.txt` is two sentences. The same model on the same code built nine sliding
drawers from one and none from the other. Any table that mixes them is wrong.

## `reference.png`

A crop of a photograph from [Pixabay](https://pixabay.com), used under the
[Pixabay Content License](https://pixabay.com/service/license-summary/), which permits use and
modification, including commercially, without attribution. It is credited here anyway, because a
reader who wants to check a result should be able to see where its only input came from.

The crop is taken from a room scene and shows the chest of drawers, the floor in front of it, and
the vases and picture frames standing on top — which the model is asked to ignore. No person appears
in it.
