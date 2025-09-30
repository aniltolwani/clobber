"""Baseline evaluation harness for claude--.

This script loads tasks (prompt + repo path), runs one or more baseline
agents, and scores each candidate diff using the shared verifier.  It is
intended as a drop-in tool for leaderboard generation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from verifier import score_patch

load_dotenv()


# ---------------------------------------------------------------------------
# Utility data structures
# ---------------------------------------------------------------------------


@dataclass
class Task:
    identifier: str
    prompt: str
    repo_path: Path
    allow_refactor: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineRun:
    baseline: str
    task_id: str
    diff: Optional[str]
    reward: Optional[float]
    diagnostics: Dict[str, Any]
    elapsed_seconds: float
    status: str
    skip_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BaselineOutput:
    diff: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    skip_reason: Optional[str] = None

    def is_skipped(self) -> bool:
        return self.skip_reason is not None

    def has_diff(self) -> bool:
        return bool(self.diff) and not self.is_skipped()


# ---------------------------------------------------------------------------
# Base class for baselines
# ---------------------------------------------------------------------------


class BaselineAgent:
    """Interface for all baseline implementations."""

    name: str

    def run(self, task: Task) -> BaselineOutput:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Heuristic baseline (tools only)
# ---------------------------------------------------------------------------


def run_command(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    import os
    import shutil
    import sys

    # Resolve command to full path if it's a binary
    resolved_cmd = list(cmd)
    if resolved_cmd:
        # First try shutil.which with current PATH
        which_path = shutil.which(resolved_cmd[0])

        # If not found and we're in a venv, check venv bin directory
        if not which_path and hasattr(sys, 'prefix'):
            venv_bin = Path(sys.prefix) / "bin" / resolved_cmd[0]
            if venv_bin.exists():
                which_path = str(venv_bin)

        if which_path:
            resolved_cmd[0] = which_path
        elif resolved_cmd[0] not in ("git",):  # git should always be available
            # Command not found - return a failed process instead of raising
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=127,
                stdout="",
                stderr=f"command not found: {resolved_cmd[0]}",
            )

    return subprocess.run(
        resolved_cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


class HeuristicToolsBaseline(BaselineAgent):
    """Minimal baseline that runs ruff --fix/format in a sandbox clone."""

    def __init__(self, steps: Optional[List[List[str]]] = None) -> None:
        self.name = "heuristic_tools"
        if steps is None:
            steps = [
                ["ruff", "check", "--fix", "--exit-zero", "."],
                ["ruff", "format", "."],
            ]
        self.steps = steps

    def run(self, task: Task) -> BaselineOutput:
        if not task.repo_path.exists():
            return BaselineOutput(diff=None, skip_reason=f"repo missing: {task.repo_path}")

        temp_root = Path(tempfile.mkdtemp(prefix=f"baseline-{self.name}-"))
        temp_dir = temp_root / "repo"
        metadata: Dict[str, Any] = {"commands": [], "return_codes": []}
        try:
            clone_cmd = [
                "git",
                "clone",
                "--local",
                "--no-hardlinks",
                str(task.repo_path),
                str(temp_dir),
            ]
            clone_res = run_command(clone_cmd, cwd=task.repo_path.parent)
            if clone_res.returncode != 0:
                return BaselineOutput(
                    diff=None,
                    skip_reason="clone failed",
                    metadata={
                        "stderr": clone_res.stderr,
                        "stdout": clone_res.stdout,
                        "returncode": clone_res.returncode,
                    },
                )

            # Execute tool steps in order; capture outputs.
            for step in self.steps:
                proc = run_command(step, cwd=temp_dir)
                metadata["commands"].append(step)
                metadata.setdefault("stdout", []).append(proc.stdout)
                metadata.setdefault("stderr", []).append(proc.stderr)
                metadata["return_codes"].append(proc.returncode)

            diff_proc = run_command(["git", "diff"], cwd=temp_dir)
            metadata["diff_returncode"] = diff_proc.returncode
            diff_text = diff_proc.stdout
            if not diff_text.strip():
                return BaselineOutput(diff=None, metadata=metadata, skip_reason="no-op")
            return BaselineOutput(diff=diff_text, metadata=metadata)
        finally:
            if temp_root.exists():
                try:
                    import shutil

                    shutil.rmtree(temp_root, ignore_errors=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# GPT-4 (OpenAI) baseline
# ---------------------------------------------------------------------------


class OpenAIGPT4Baseline(BaselineAgent):
    """Calls OpenAI GPT-4 family models to produce unified diffs."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> None:
        self.name = "gpt4"
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # Client pulls API credentials from environment variables (OPENAI_API_KEY).
        try:
            self._client = OpenAI()
        except Exception:
            self._client = None

    def run(self, task: Task) -> BaselineOutput:
        if self._client is None:
            return BaselineOutput(
                diff=None,
                skip_reason="openai-not-configured",
                metadata={"error": "OPENAI_API_KEY not set"},
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an automated code maintenance agent. Respond with ONLY a unified diff (diff --git ...). "
                    "Avoid explanations, shell commands, or markdown fences."
                ),
            },
            {"role": "user", "content": task.prompt},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )

            diff_text = response.choices[0].message.content or ""
            if not diff_text.strip():
                return BaselineOutput(diff=None, skip_reason="empty-response")

            metadata = {
                "model": self.model,
                "usage": response.usage.model_dump() if response.usage else {},
            }
            return BaselineOutput(diff=diff_text, metadata=metadata)
        except Exception as exc:
            return BaselineOutput(
                diff=None,
                skip_reason="api-error",
                metadata={"error": str(exc)},
            )


# ---------------------------------------------------------------------------
# Placeholder baselines for external agents
# ---------------------------------------------------------------------------


class PlaceholderBaseline(BaselineAgent):
    def __init__(self, name: str, instructions: str) -> None:
        self.name = name
        self.instructions = textwrap.dedent(instructions).strip()

    def run(self, task: Task) -> BaselineOutput:
        return BaselineOutput(
            diff=None,
            skip_reason="not-configured",
            metadata={"instructions": self.instructions},
        )


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_tasks(path: Path, limit: Optional[int]) -> List[Task]:
    tasks: List[Task] = []
    for idx, raw in enumerate(read_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        repo_path = Path(raw["repo_path"]).expanduser().resolve()
        task = Task(
            identifier=raw.get("metadata", {}).get("html_url", f"task-{idx}"),
            prompt=raw["prompt"],
            repo_path=repo_path,
            allow_refactor=bool(raw.get("allow_refactor", False)),
            metadata=raw.get("metadata", {}),
        )
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    agent: BaselineAgent,
    tasks: Iterable[Task],
    dry_run: bool,
) -> Iterator[BaselineRun]:
    for task in tasks:
        start = time.time()
        try:
            output = agent.run(task)
        except Exception as exc:  # pragma: no cover - baseline failure
            yield BaselineRun(
                baseline=agent.name,
                task_id=task.identifier,
                diff=None,
                reward=None,
                diagnostics={},
                elapsed_seconds=time.time() - start,
                status="error",
                error=str(exc),
            )
            continue

        elapsed = time.time() - start
        diagnostics = {"baseline_metadata": output.metadata}

        if output.is_skipped():
            yield BaselineRun(
                baseline=agent.name,
                task_id=task.identifier,
                diff=None,
                reward=None,
                diagnostics=diagnostics,
                elapsed_seconds=elapsed,
                status="skipped",
                skip_reason=output.skip_reason,
            )
            continue

        diff = output.diff or ""
        diff_clean = sanitize_diff(diff)
        if dry_run:
            yield BaselineRun(
                baseline=agent.name,
                task_id=task.identifier,
                diff=diff_clean,
                reward=None,
                diagnostics=diagnostics,
                elapsed_seconds=elapsed,
                status="dry-run",
            )
            continue

        if diff_clean != diff:
            diagnostics.setdefault("baseline_metadata", {})["sanitized"] = True
        diagnostics["raw_diff_preview"] = (diff[:200] + "...") if len(diff) > 200 else diff

        reward, diag = score_patch(
            workdir=str(task.repo_path),
            unified_diff=diff_clean,
            allow_refactor=task.allow_refactor,
        )
        diagnostics.update(diag)
        yield BaselineRun(
            baseline=agent.name,
            task_id=task.identifier,
            diff=diff_clean,
            reward=reward,
            diagnostics=diagnostics,
            elapsed_seconds=elapsed,
            status="ok",
        )


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


BASELINE_REGISTRY: Dict[str, BaselineAgent] = {
    "heuristic": HeuristicToolsBaseline(),
    "gpt4": OpenAIGPT4Baseline(model=os.environ.get("CLOBBER_GPT4_MODEL", "gpt-4o-mini")),
    "gpt4o": OpenAIGPT4Baseline(model="gpt-4o"),
    "aider": PlaceholderBaseline(
        "aider",
        "Invoke the aider CLI with tool schema bindings and return its diff output.",
    ),
    "openhands": PlaceholderBaseline(
        "openhands",
        "Run the OpenHands agent in headless mode with the shared tool schema, then provide the produced diff.",
    ),
    "qwen": PlaceholderBaseline(
        "qwen",
        "Call the Qwen3-Coder instruct model (or tuned variant) with tool schema support and return the resulting diff.",
    ),
}


def resolve_baselines(names: Sequence[str]) -> List[BaselineAgent]:
    agents: List[BaselineAgent] = []
    for name in names:
        key = name.lower()
        if key not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline: {name}")
        agents.append(BASELINE_REGISTRY[key])
    return agents


def write_csv(path: Path, rows: Iterable[BaselineRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        return
    fieldnames = [
        "baseline",
        "task_id",
        "status",
        "skip_reason",
        "error",
        "reward",
        "elapsed_seconds",
        "diff",
        "diagnostics",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(
                {
                    "baseline": row.baseline,
                    "task_id": row.task_id,
                    "status": row.status,
                    "skip_reason": row.skip_reason or "",
                    "error": row.error or "",
                    "reward": row.reward if row.reward is not None else "",
                    "elapsed_seconds": f"{row.elapsed_seconds:.2f}",
                    "diagnostics": json.dumps(row.diagnostics, ensure_ascii=False),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate baselines using the shared verifier")
    parser.add_argument("--dataset", required=True, help="Path to prompt dataset (jsonl)")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["heuristic"],
        help="Baseline names to evaluate (heuristic, gpt4, gpt4o, aider, openhands, qwen)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--output", default="data/baseline_results.csv")
    parser.add_argument("--dry-run", action="store_true", help="Skip verifier scoring")
    parser.add_argument(
        "--print-summaries",
        action="store_true",
        help="Print per-task summaries to stdout",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")

    tasks = load_tasks(dataset_path, args.limit)
    if not tasks:
        parser.error("no tasks loaded; check dataset path or limit")

    agents = resolve_baselines(args.baselines)
    all_rows: List[BaselineRun] = []

    for agent in agents:
        for row in evaluate(agent, tasks, dry_run=args.dry_run):
            all_rows.append(row)
            if args.print_summaries:
                summary = f"[{agent.name}] {row.task_id} -> {row.status}"
                if row.reward is not None:
                    summary += f" reward={row.reward:.3f}"
                if row.skip_reason:
                    summary += f" (skip: {row.skip_reason})"
                if row.error:
                    summary += f" (error: {row.error})"
                print(summary)

    output_path = Path(args.output)
    write_csv(output_path, all_rows)
    print(f"wrote {len(all_rows)} rows to {output_path}")


def sanitize_diff(diff: Optional[str]) -> str:
    if not diff:
        return ""
    text = diff.strip()
    if text.startswith("```diff"):
        text = text[len("```diff"):].lstrip()
    if text.startswith("```patch"):
        text = text[len("```patch"):].lstrip()
    if text.startswith("```"):
        text = text[len("```"):].lstrip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    lines = [line for line in text.splitlines() if line.strip() != "```"]
    return "\n".join(lines).strip()


if __name__ == "__main__":
    main()
