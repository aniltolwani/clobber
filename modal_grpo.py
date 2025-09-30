"""Modal entry point for GRPO training with GPU support."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import modal
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "modal package not installed. Install with `pip install modal` or add to your environment."
    ) from exc

from grpo_trainer import main as grpo_main


DEFAULT_MODEL_ID = os.environ.get("CLOBBER_GRPO_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
DEFAULT_DATASET = os.environ.get("CLOBBER_GRPO_DATASET", "data/grpo_prompts.jsonl")
DEFAULT_TIMEOUT = int(os.environ.get("CLOBBER_GRPO_TIMEOUT", str(60 * 60 * 4)))  # 4 hours


image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "accelerate",
        "datasets",
        "transformers",
        "trl",
        "pyright",
        "deptry",
        "mutmut",
        "pydeps",
        "pytest-testmon",
        "pytest-xdist",
    )
)

stub = modal.Stub("clobber-grpo-trainer", image=image)


@stub.function(gpu="A100", timeout=DEFAULT_TIMEOUT)
def train(
    model_id: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> None:
    """Run GRPO training inside a Modal GPU worker."""

    resolved_model = model_id or DEFAULT_MODEL_ID
    resolved_dataset = dataset_path or DEFAULT_DATASET
    grpo_main(model_id=resolved_model, dataset_path=Path(resolved_dataset))


if __name__ == "__main__":
    with stub.run():
        train.remote()
