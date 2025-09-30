"""Low-latency verifier and reward computation for deletion-focused agents."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


# Ruff codes that map to unused imports/variables
RUFF_UNUSED_RULES = {"F401", "F841"}
MAX_SCORE_STYLE_BONUS = 0.05
SENTINEL_BAD_METRIC = 10**6


@dataclass
class Metrics:
    ruff_unused: int
    pyright_errors: int
    deptry_unused: int
    graph_edges: int
    graph_nodes: int


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    logs: str = ""


RunFunction = Callable[[str, str], Tuple[int, str, str]]
JudgeFunction = Callable[[str], float]


def run_command(cmd: str, cwd: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """Run a shell command, capturing stdout/stderr."""

    process = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
    )
    return process.returncode, process.stdout, process.stderr


def _safe_json_load(payload: str, default: Any) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default


def count_ruff_unused(cwd: str, paths: str = ".") -> int:
    code, out, _ = run_command(f"ruff check --output-format json {paths}", cwd)
    if code not in (0, 1):
        return SENTINEL_BAD_METRIC
    data = _safe_json_load(out or "[]", [])
    return sum(1 for item in data if item.get("code") in RUFF_UNUSED_RULES)


def count_pyright_errors(cwd: str, paths: str = ".") -> int:
    code, out, _ = run_command(f"pyright --outputjson {paths}", cwd)
    if code not in (0, 1):
        return SENTINEL_BAD_METRIC
    data = _safe_json_load(out or "{}", {})
    return int(data.get("summary", {}).get("errorCount", 0))


def count_deptry_unused(cwd: str) -> int:
    code, out, _ = run_command("deptry --json .", cwd)
    if code not in (0, 1):
        return SENTINEL_BAD_METRIC
    data = _safe_json_load(out or "{}", {})
    violations = data.get("violations", [])
    return sum(1 for item in violations if item.get("code") == "DEP002")


def pydeps_graph_stats(cwd: str, path: str = ".", max_depth: int = 2) -> Tuple[int, int]:
    cmd = f"pydeps --show-deps --max-bacon 0 --max-depth {max_depth} --no-output {path}"
    code, out, _ = run_command(cmd, cwd)
    if code != 0:
        return SENTINEL_BAD_METRIC, SENTINEL_BAD_METRIC
    lines = out.splitlines()
    edges = sum(1 for line in lines if "->" in line)
    nodes = sum(1 for line in lines if "[label" in line)
    return edges, nodes


def collect_metrics(cwd: str) -> Metrics:
    edges, nodes = pydeps_graph_stats(cwd)
    return Metrics(
        ruff_unused=count_ruff_unused(cwd),
        pyright_errors=count_pyright_errors(cwd),
        deptry_unused=count_deptry_unused(cwd),
        graph_edges=edges,
        graph_nodes=nodes,
    )


def clone_repo(src: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="verifier-")
    code, _, err = run_command(f"git clone --local --no-hardlinks . {tmp_dir}", src)
    if code != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {err.strip()}")
    return tmp_dir


def apply_patch(cwd: str, unified_diff: str) -> GateResult:
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(unified_diff)
        diff_path = tmp.name
    try:
        code, _, err = run_command(f"git apply --check {diff_path}", cwd)
        if code != 0:
            return GateResult(False, "git-apply-check", err)
        code, _, err = run_command(f"git apply {diff_path}", cwd)
        if code != 0:
            return GateResult(False, "git-apply", err)
    finally:
        os.unlink(diff_path)
    return GateResult(True)


def run_py_compile(cwd: str) -> GateResult:
    script = (
        "python - <<'PY'\n"
        "import os, sys, py_compile\n"
        "for root, _, files in os.walk('.'):\n"
        "    for name in files:\n"
        "        if name.endswith('.py'):\n"
        "            path = os.path.join(root, name)\n"
        "            try:\n"
        "                py_compile.compile(path, doraise=True)\n"
        "            except Exception as exc:\n"
        "                print(path, exc)\n"
        "                sys.exit(2)\n"
        "print('OK')\n"
        "PY"
    )
    code, out, err = run_command(script, cwd)
    if code != 0:
        return GateResult(False, "py-compile", out + err)
    return GateResult(True)


def run_pytest_impacted(cwd: str, maxfail: int = 1, xdist: bool = True) -> GateResult:
    flag = "-x" if maxfail == 1 else f"--maxfail={maxfail}"
    workers = "-n auto" if xdist else ""
    cmd = f"pytest --testmon {workers} {flag} -q"
    code, out, err = run_command(cmd, cwd)
    if code != 0:
        return GateResult(False, "pytest-impacted", out + err)
    return GateResult(True, logs=out)


def run_git_numstat(cwd: str) -> Tuple[int, int]:
    code, out, _ = run_command("git diff --numstat", cwd)
    if code != 0:
        return 0, 0
    added, deleted = 0, 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, d, _ = parts
        if a.isdigit():
            added += int(a)
        if d.isdigit():
            deleted += int(d)
    return added, deleted


def style_judge_score(diff_text: str, judge: Optional[JudgeFunction]) -> float:
    if judge is None:
        return 0.0
    try:
        score = float(judge(diff_text))
    except Exception:
        return 0.0
    return max(0.0, min(MAX_SCORE_STYLE_BONUS, score))


def compute_reward(
    before: Metrics,
    after: Metrics,
    added: int,
    deleted: int,
    mode: str,
    style_bonus: float,
) -> float:
    def safe_delta(prev: int, new: int) -> float:
        if prev <= 0:
            return 0.0 if new >= prev else 1.0
        return (prev - new) / max(1, prev)

    deletions_total = deleted + added
    deletion_ratio = (deleted - added) / max(1, deletions_total) if deletions_total else 0.0
    delta_unused = safe_delta(before.ruff_unused, after.ruff_unused)
    delta_types = safe_delta(before.pyright_errors, after.pyright_errors)
    delta_dep = safe_delta(before.deptry_unused, after.deptry_unused)
    delta_graph = safe_delta(before.graph_edges, after.graph_edges)

    if mode == "refactor":
        loc = max(1, abs(added) + abs(deleted))
        quality = 0.5 * delta_unused + 0.3 * delta_types + 0.2 * delta_dep
        reward = quality / loc + 0.25 * delta_types + 0.15 * delta_graph
    else:
        reward = (
            0.50 * deletion_ratio
            + 0.25 * delta_unused
            + 0.15 * (0.5 * delta_dep + 0.5 * delta_graph)
            + 0.10 * delta_types
        )

    return reward + style_bonus


def score_patch(
    workdir: str,
    unified_diff: str,
    allow_refactor: bool = False,
    judge: Optional[JudgeFunction] = None,
) -> Tuple[float, Dict[str, Any]]:
    tmp_dir = clone_repo(workdir)
    try:
        before_metrics = collect_metrics(tmp_dir)
        gate = apply_patch(tmp_dir, unified_diff)
        if not gate.passed:
            return -1.0, {"gate": gate.reason, "logs": gate.logs}

        compile_gate = run_py_compile(tmp_dir)
        if not compile_gate.passed:
            return -1.0, {"gate": compile_gate.reason, "logs": compile_gate.logs}

        after_pyright = count_pyright_errors(tmp_dir)
        if after_pyright > before_metrics.pyright_errors:
            return -1.0, {"gate": "pyright-regression"}

        tests = run_pytest_impacted(tmp_dir)
        if not tests.passed:
            retry = run_pytest_impacted(tmp_dir)
            if not retry.passed:
                return -1.0, {"gate": tests.reason, "logs": tests.logs}

        added, deleted = run_git_numstat(tmp_dir)
        after_metrics = collect_metrics(tmp_dir)

        diff_code, diff_text, _ = run_command("git diff", tmp_dir)
        diff_output = diff_text if diff_code == 0 else ""
        style_bonus = style_judge_score(diff_output, judge)
        mode = "refactor" if allow_refactor else "delete"
        reward = compute_reward(before_metrics, after_metrics, added, deleted, mode, style_bonus)

        diagnostics = {
            "added": added,
            "deleted": deleted,
            "before": asdict(before_metrics),
            "after": asdict(after_metrics),
            "style_bonus": style_bonus,
            "mode": mode,
        }
        return reward, diagnostics
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


__all__ = [
    "Metrics",
    "GateResult",
    "score_patch",
    "collect_metrics",
    "compute_reward",
]
