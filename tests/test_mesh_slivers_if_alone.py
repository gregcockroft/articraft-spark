"""Slivers stop blocking a compile only when nothing else in that compile's report is wrong.

A sliver triangle is a tessellation artifact -- in the runs that motivated this, a 9 um strip along a
drawer front, invisible in any render -- and on its own it blocked five compiles on this bench. Forgiving
it unconditionally is a different change (``ARTICRAFT_MESH_SLIVERS_NONBLOCKING``, rejected: it let a
half-built draft compile clean). Here the sliver failure is demoted to a diagnostic only when every
failure in the compile's own report is a sliver-only ``mesh_health`` failure and the report carries no
diagnostics, so any other finding -- of any ``FailureKind``, including one added after this was written --
keeps the block.

The predicate cases are unit tests over reports; the lane through the real worker subprocess is covered
by four compiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from articraft.agent.workspace.local import LocalWorkspace
from articraft.compiler.worker import (
    _demote_sliver_failures,
    _sliver_failures_stand_alone,
)
from articraft.sdk import FailureKind, TestReport
from articraft.sdk.testing import TestFailure
from articraft.settings import Settings

SLIVER_DETAILS = (
    "Unhealthy mesh geometry detected:\n"
    "part='base' shape='panel' issue='sliver_faces' count=2 details=triangle quality is below 0.0001"
)
TWO_ISSUE_DETAILS = (
    "Unhealthy mesh geometry detected:\n"
    "part='base' shape='panel' issue='sliver_faces' count=2\n"
    "part='base' shape='panel' issue='boundary_edges' count=4"
)


def sliver_failure(name: str = "fail_if_mesh_unhealthy(model)") -> TestFailure:
    return TestFailure(name, SLIVER_DETAILS, kind=FailureKind.MESH_HEALTH)


def report(
    failures: tuple[TestFailure, ...] = (),
    diagnostics: tuple[TestFailure, ...] = (),
) -> TestReport:
    return TestReport(
        passed=not failures,
        checks_run=1,
        checks=("check",),
        failures=failures,
        diagnostics=diagnostics,
    )


def test_a_sliver_only_failure_stands_alone() -> None:
    assert _sliver_failures_stand_alone((report(), report((sliver_failure(),))))


def test_no_failure_at_all_is_not_a_sliver_to_forgive() -> None:
    """The predicate answers "are the only failures slivers?", so a clean report is False."""
    assert not _sliver_failures_stand_alone((report(), report()))


@pytest.mark.parametrize(
    "kind", [kind for kind in FailureKind if kind is not FailureKind.MESH_HEALTH]
)
def test_every_other_failure_kind_keeps_the_block(kind: FailureKind) -> None:
    """Enumerated over FailureKind: only mesh_health can ever be demoted."""
    other = TestFailure("other_check", "something else is wrong", kind=kind)
    assert not _sliver_failures_stand_alone((report(), report((sliver_failure(), other))))
    assert not _sliver_failures_stand_alone((report((other,)), report((sliver_failure(),))))


def test_a_failure_kind_added_later_keeps_the_block() -> None:
    """The predicate is an allow-list of one kind, so a kind that does not exist yet blocks."""
    future = TestFailure("new_check", "a kind from the future", kind="thermal_budget")  # type: ignore[arg-type]
    assert not _sliver_failures_stand_alone((report(), report((sliver_failure(), future))))


def test_a_diagnostic_keeps_the_block() -> None:
    """r03's shape: its only blocking failure was slivers, but it had overlap diagnostics."""
    overlap = TestFailure("overlap_check", "parts overlap", kind=FailureKind.OVERLAP)
    assert not _sliver_failures_stand_alone(
        (report(), report((sliver_failure(),), diagnostics=(overlap,)))
    )


def test_a_second_mesh_issue_on_the_same_failure_keeps_the_block() -> None:
    two = TestFailure(
        "fail_if_mesh_unhealthy(model)", TWO_ISSUE_DETAILS, kind=FailureKind.MESH_HEALTH
    )
    assert not _sliver_failures_stand_alone((report(), report((two,))))


def test_a_mesh_failure_with_no_issue_lines_keeps_the_block() -> None:
    empty = TestFailure("fail_if_mesh_unhealthy(model)", "", kind=FailureKind.MESH_HEALTH)
    assert not _sliver_failures_stand_alone((report(), report((empty,))))


def test_an_authored_sliver_failure_is_demoted_with_the_compiler_one() -> None:
    authored = report((sliver_failure("authored_mesh_check"),))
    baseline = report((sliver_failure(),))
    assert _sliver_failures_stand_alone((authored, baseline))
    demoted = _demote_sliver_failures(authored)
    assert demoted.failures == ()
    assert demoted.passed
    assert demoted.diagnostics[0].name == "authored_mesh_check"


def test_demotion_keeps_other_failures_untouched() -> None:
    other = TestFailure("other_check", "wrong", kind=FailureKind.MODEL_VALIDITY)
    demoted = _demote_sliver_failures(report((sliver_failure(), other)))
    assert demoted.failures == (other,)
    assert not demoted.passed
    assert [d.name for d in demoted.diagnostics] == ["fail_if_mesh_unhealthy(model)"]


def test_the_setting_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARTICRAFT_MESH_SLIVERS_NONBLOCKING_IF_ALONE", raising=False)
    assert Settings().mesh_slivers_nonblocking_if_alone is False  # pyright: ignore[reportCallIssue]
    monkeypatch.setenv("ARTICRAFT_MESH_SLIVERS_NONBLOCKING_IF_ALONE", "1")
    assert Settings().mesh_slivers_nonblocking_if_alone is True  # pyright: ignore[reportCallIssue]


# --- through the real compile worker ------------------------------------------------------------

SLIVER_HELPER = '''
import numpy as np
from collections import defaultdict
from articraft.sdk import BoxGeometry, MeshGeometry, RigidBodyAssembly, TestContext, TestReport


def sliver_box(eps=1e-7):
    """A watertight unit box whose one split edge leaves two sliver triangles."""
    box = BoxGeometry((1.0, 1.0, 1.0))
    verts = [tuple(float(x) for x in v) for v in np.asarray(box.vertices, dtype=float)]
    faces = [tuple(int(i) for i in f) for f in np.asarray(box.faces)]
    share = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            share[tuple(sorted((u, v)))].append(fi)
    edge, (f1, f2) = next((e, fs) for e, fs in share.items() if len(fs) == 2)
    v0, v1 = edge
    a0, a1 = np.asarray(verts[v0]), np.asarray(verts[v1])
    verts.append(tuple(float(x) for x in a0 + eps * (a1 - a0)))
    pi = len(verts) - 1
    new = []
    for fi, f in enumerate(faces):
        if fi not in (f1, f2):
            new.append(f)
            continue
        i = [k for k in range(3) if {f[k], f[(k + 1) % 3]} == {v0, v1}][0]
        a, b, c = f[i], f[(i + 1) % 3], f[(i + 2) % 3]
        new.append((a, pi, c))
        new.append((pi, b, c))
    return MeshGeometry(verts, new)
'''

SLIVER_ONLY_MAIN = (
    SLIVER_HELPER
    + """
object_model = RigidBodyAssembly("sliver_only")
object_model.rigid_body("base").add(sliver_box(), name="panel")


def run_tests() -> TestReport:
    return TestContext(object_model).report()
"""
)

SLIVER_PLUS_OVERLAP_MAIN = (
    SLIVER_HELPER
    + """
object_model = RigidBodyAssembly("sliver_and_overlap")
object_model.rigid_body("base").add(sliver_box(), name="panel")
object_model.rigid_body("other").add(BoxGeometry((0.5, 0.5, 0.5)), name="block")


def run_tests() -> TestReport:
    return TestContext(object_model).report()
"""
)

SLIVER_PLUS_HOLE_MAIN = (
    SLIVER_HELPER
    + """
object_model = RigidBodyAssembly("sliver_and_hole")
base = object_model.rigid_body("base")
base.add(sliver_box(), name="panel")
box = BoxGeometry((1.0, 1.0, 1.0))
base.add(MeshGeometry(box.vertices, box.faces[:-1]), name="open-sheet")


def run_tests() -> TestReport:
    return TestContext(object_model).report()
"""
)


def compile_source(tmp_path: Path, name: str, source: str, *, if_alone: bool) -> dict[str, Any]:
    env = LocalWorkspace(output_dir=tmp_path, mesh_slivers_nonblocking_if_alone=if_alone)
    run_dir = env.create_run(name)
    (run_dir / "workspace" / "main.py").write_text(source, encoding="utf-8")
    return env.compile_path(run_dir)


def signals(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result["compile_report"]["signal_bundle"]["signals"]


def test_a_sliver_only_workspace_blocks_by_default(tmp_path: Path) -> None:
    result = compile_source(tmp_path, "default", SLIVER_ONLY_MAIN, if_alone=False)

    assert result["status"] == "error"
    failure = result["test_report"]["failures"][0]
    assert failure["kind"] == "mesh_health"
    assert "sliver_faces" in failure["details"]


def test_a_sliver_only_workspace_compiles_with_the_setting_and_still_warns(tmp_path: Path) -> None:
    result = compile_source(tmp_path, "if-alone", SLIVER_ONLY_MAIN, if_alone=True)

    assert result["status"] == "success"
    assert result["test_report"]["failures"] == []
    warned = [
        signal
        for signal in signals(result)
        if signal.get("severity") == "warning" and "sliver_faces" in str(signal)
    ]
    assert warned, "the model must still read the sliver finding"


def test_a_sliver_beside_another_finding_still_blocks(tmp_path: Path) -> None:
    """Two loose parts: the mesh is sliver-only, but the report is not."""
    result = compile_source(tmp_path, "overlap", SLIVER_PLUS_OVERLAP_MAIN, if_alone=True)

    assert result["status"] == "error"


def test_a_second_mesh_issue_still_blocks(tmp_path: Path) -> None:
    result = compile_source(tmp_path, "hole", SLIVER_PLUS_HOLE_MAIN, if_alone=True)

    assert result["status"] == "error"
    kinds = {failure["kind"] for failure in result["test_report"]["failures"]}
    assert kinds == {"mesh_health"}
