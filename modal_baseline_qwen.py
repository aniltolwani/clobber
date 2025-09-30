"""Modal baseline evaluation using a local Qwen model."""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import modal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import score_patch

MOUNT_PATH = "/root/clobber"
DEFAULT_DATASET = os.environ.get("CLOBBER_BASELINE_DATASET", "/vol/datasets/grpo_prompts.jsonl")
DEFAULT_MODEL = os.environ.get("CLOBBER_BASELINE_MODEL", "Qwen/Qwen1.5-0.5B")
DEFAULT_OUTPUT = os.environ.get("CLOBBER_BASELINE_OUTPUT", "/vol/baselines/qwen_baseline.csv")
MAX_NEW_TOKENS = int(os.environ.get("CLOBBER_BASELINE_MAX_NEW_TOKENS", "512"))
SYSTEM_PROMPT = (
    "You are a code maintenance agent. Respond ONLY with a unified diff (diff --git ...) "
    "that applies the requested change. No explanations or markdown fences."
)

image = (
    modal.Image.debian_slim()
    .add_local_dir(
        Path.cwd(),
        remote_path=MOUNT_PATH,
        ignore=[
            "repos/**",
            "checkpoints/**",
            ".venv/**",
            ".git/**",
            "__pycache__/**",
            "*.pyc",
            ".pytest_cache/**",
            ".ruff_cache/**",
            "data/*.jsonl",
        ],
    )
    .apt_install("git")
    .pip_install(
        "torch",
        "accelerate",
        "transformers",
        "datasets",
    )
)

app = modal.App("clobber-baseline-qwen")
volume = modal.Volume.from_name("clobber-data", create_if_missing=True)

if MOUNT_PATH not in sys.path:
    sys.path.append(MOUNT_PATH)


@dataclass
class BaselineRecord:
    prompt_id: str
    repo_path: str
    allow_refactor: bool
    diff: str
    reward: float
    diagnostics: dict[str, Any]
    status: str


def load_dataset(path: Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_generation_prompt(user_prompt: str) -> str:
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>\n"


def generate_diff(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract assistant segment after the last prompt marker
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    return text.strip()


def score_diff(record: dict[str, Any], diff: str) -> BaselineRecord:
    repo_path = record["repo_path"]
    allow_refactor = bool(record.get("allow_refactor", False))
    reward, diagnostics = score_patch(
        workdir=repo_path,
        unified_diff=diff,
        allow_refactor=allow_refactor,
    )
    return BaselineRecord(
        prompt_id=record.get("metadata", {}).get("html_url", record.get("prompt", "")[:40]),
        repo_path=repo_path,
        allow_refactor=allow_refactor,
        diff=diff,
        reward=reward,
        diagnostics=diagnostics,
        status="ok" if reward != -1.0 else diagnostics.get("gate", "failed"),
    )


def write_csv(path: Path, rows: Iterable[BaselineRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "prompt_id",
            "repo_path",
            "allow_refactor",
            "reward",
            "status",
            "diff",
            "diagnostics",
        ])
        for row in rows:
            writer.writerow([
                row.prompt_id,
                row.repo_path,
                row.allow_refactor,
                f"{row.reward:.4f}",
                row.status,
                row.diff.replace("\n", "\\n"),
                json.dumps(row.diagnostics, ensure_ascii=False),
            ])


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
)
def run_qwen_baseline(
    dataset_path: str = DEFAULT_DATASET,
    output_csv: str = DEFAULT_OUTPUT,
    limit: Optional[int] = None,
) -> None:
    dataset = load_dataset(Path(dataset_path))
    if limit is not None:
        dataset = dataset[:limit]

    print(f"Loaded {len(dataset)} prompts from {dataset_path}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    results: List[BaselineRecord] = []
    for idx, record in enumerate(dataset, 1):
        prompt_text = build_generation_prompt(record["prompt"])
        diff = generate_diff(tokenizer, model, prompt_text)
        try:
            baseline_record = score_diff(record, diff)
        except Exception as exc:  # noqa: BLE001
            baseline_record = BaselineRecord(
                prompt_id=record.get("metadata", {}).get("html_url", f"record-{idx}"),
                repo_path=record["repo_path"],
                allow_refactor=bool(record.get("allow_refactor", False)),
                diff=diff,
                reward=-1.0,
                diagnostics={"error": str(exc)},
                status="error",
            )
        results.append(baseline_record)
        print(
            f"[{idx}/{len(dataset)}] reward={baseline_record.reward:.3f} status={baseline_record.status} repo={baseline_record.repo_path}"
        )

    out_path = Path(output_csv)
    write_csv(out_path, results)
    print(f"Saved baseline results to {out_path}")


@app.local_entrypoint()
def main(dataset: Optional[str] = None, output: Optional[str] = None, limit: Optional[int] = None) -> None:
    run_qwen_baseline.remote(
        dataset_path=dataset or DEFAULT_DATASET,
        output_csv=output or DEFAULT_OUTPUT,
        limit=limit,
    )
