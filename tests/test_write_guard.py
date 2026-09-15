"""`write` refuses a small change to a file that already exists; `edit` is for that.

The scripted-agent lane drives the refusal through the real loop, so what is
checked is what the model actually receives back from the tool call.
"""

from __future__ import annotations

from pathlib import Path

from harness import GOOD_MAIN_PY, WarmEnvironment, calls, run_scenario, text, tool_call

from articraft.agent.tools.write import MIN_REWRITE_SHARE, changed_line_share

HELPER = "".join(f"VALUE_{index} = {index}\n" for index in range(300))
ONE_LINE_CHANGED = HELPER.replace("VALUE_7 = 7\n", "VALUE_7 = 70\n")
HALF_REWRITTEN = "".join(f"OTHER_{index} = {index}\n" for index in range(150)) + "".join(
    f"VALUE_{index} = {index}\n" for index in range(150, 300)
)


def workspace_of(artifacts) -> Path:
    return artifacts.run_dir / "workspace"


def test_a_one_line_change_to_an_existing_file_is_refused_and_names_edit(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "helper.py", "content": HELPER})),
            calls(tool_call("write", {"path": "helper.py", "content": ONE_LINE_CHANGED})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=4,
    )

    outputs = artifacts.tool_outputs()
    assert "error" not in outputs[0]
    error = outputs[1]["error"]
    assert "helper.py was already written" in error
    assert "changes 0% of its lines (1 of 300)" in error
    assert "Use edit" in error
    assert workspace_of(artifacts).joinpath("helper.py").read_text(encoding="utf-8") == HELPER


def test_a_large_rewrite_of_an_existing_file_is_accepted(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "helper.py", "content": HELPER})),
            calls(tool_call("write", {"path": "helper.py", "content": HALF_REWRITTEN})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=4,
    )

    assert artifacts.tool_outputs()[1]["result"]["path"] == "helper.py"
    written = workspace_of(artifacts).joinpath("helper.py").read_text(encoding="utf-8")
    assert written == HALF_REWRITTEN


def test_a_first_write_of_a_new_path_is_accepted(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "parts/new.py", "content": "VALUE = 1\n"})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=3,
    )

    assert artifacts.tool_outputs()[0]["result"] == {"path": "parts/new.py", "bytes": 10}
    assert workspace_of(artifacts).joinpath("parts/new.py").exists()


def test_an_identical_rewrite_is_refused(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "helper.py", "content": HELPER})),
            calls(tool_call("write", {"path": "helper.py", "content": HELPER})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=4,
    )

    assert "changes 0% of its lines (0 of 300)" in artifacts.tool_outputs()[1]["error"]


def test_the_first_write_of_the_scaffolded_main_py_is_accepted(tmp_path: Path) -> None:
    """The workspace ships a 19-line main.py; replacing it is a first write, not a rewrite."""
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=3,
    )

    assert "error" not in artifacts.tool_outputs()[0]
    assert workspace_of(artifacts).joinpath("main.py").read_text(encoding="utf-8") == GOOD_MAIN_PY


def test_a_refused_write_leaves_the_file_byte_for_byte(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "helper.py", "content": HELPER})),
            calls(tool_call("write", {"path": "helper.py", "content": ONE_LINE_CHANGED})),
            calls(tool_call("read", {"path": "helper.py"})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=5,
    )

    outputs = artifacts.tool_outputs()
    assert "error" in outputs[1]
    assert workspace_of(artifacts).joinpath("helper.py").read_bytes() == HELPER.encode("utf-8")
    assert outputs[2]["result"]["text"].startswith("L1: VALUE_0 = 0\nL2: VALUE_1 = 1\n")


def test_edit_still_makes_the_small_change_the_guard_refused(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "helper.py", "content": HELPER})),
            calls(tool_call("write", {"path": "helper.py", "content": ONE_LINE_CHANGED})),
            calls(
                tool_call(
                    "edit",
                    {
                        "path": "helper.py",
                        "edits": [{"old_text": "VALUE_7 = 7\n", "new_text": "VALUE_7 = 70\n"}],
                    },
                )
            ),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=5,
    )

    assert "error" in artifacts.tool_outputs()[1]
    assert artifacts.tool_outputs()[2]["result"] == {"path": "helper.py", "replaced": 1}
    written = workspace_of(artifacts).joinpath("helper.py").read_text(encoding="utf-8")
    assert written == ONE_LINE_CHANGED


def test_the_threshold_is_a_share_of_the_longer_file() -> None:
    ten = "".join(f"line {index}\n" for index in range(10))
    four_changed = ten.replace("line 0\n", "x\n").replace("line 1\n", "x\n")
    four_changed = four_changed.replace("line 2\n", "x\n").replace("line 3\n", "x\n")
    three_changed = ten.replace("line 0\n", "x\n").replace("line 1\n", "x\n")
    three_changed = three_changed.replace("line 2\n", "x\n")

    assert changed_line_share(ten, four_changed) == (4, 10, 0.4)
    assert changed_line_share(ten, three_changed) == (3, 10, 0.3)
    assert changed_line_share(ten, ten) == (0, 10, 0.0)
    assert changed_line_share("", "one\ntwo\n") == (2, 2, 1.0)
    assert changed_line_share("one\ntwo\n", "") == (2, 2, 1.0)
    assert changed_line_share("", "") == (0, 0, 0.0)
    assert MIN_REWRITE_SHARE == 0.4


def test_the_boundary_is_inclusive(tmp_path: Path) -> None:
    ten = "".join(f"line {index}\n" for index in range(10))
    exactly_four = ten
    for index in range(4):
        exactly_four = exactly_four.replace(f"line {index}\n", f"changed {index}\n")

    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "helper.py", "content": ten})),
            calls(tool_call("write", {"path": "helper.py", "content": exactly_four})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=4,
    )

    assert changed_line_share(ten, exactly_four)[2] == MIN_REWRITE_SHARE
    assert "error" not in artifacts.tool_outputs()[1]


def test_truncating_a_file_to_one_line_is_a_large_change_and_is_written(tmp_path: Path) -> None:
    """The share is of lines touched, so deleting 299 of 300 lines is not a small change."""
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "helper.py", "content": HELPER})),
            calls(tool_call("write", {"path": "helper.py", "content": "VALUE_0 = 0\n"})),
            text("done"),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=4,
    )

    assert artifacts.tool_outputs()[1]["result"] == {"path": "helper.py", "bytes": 12}
    assert changed_line_share(HELPER, "VALUE_0 = 0\n") == (299, 300, 299 / 300)
