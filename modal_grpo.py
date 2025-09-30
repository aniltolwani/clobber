"""Modal entry point for GRPO training with GPU support."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import modal

from grpo_trainer import main as grpo_main


DEFAULT_MODEL_ID = os.environ.get("CLOBBER_GRPO_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
DEFAULT_DATASET = os.environ.get("CLOBBER_GRPO_DATASET", "/vol/datasets/grpo_prompts.jsonl")
DEFAULT_TIMEOUT = int(os.environ.get("CLOBBER_GRPO_TIMEOUT", str(60 * 60 * 4)))  # 4 hours


app = modal.App("clobber-grpo-trainer")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "accelerate",
        "datasets",
        "transformers",
        "trl",
        "ruff",
        "pyright",
        "deptry",
        "mutmut",
        "pydeps",
        "pytest-testmon",
        "pytest-xdist",
    )
)

# Create or reference the shared volume
volume = modal.Volume.from_name("clobber-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # 24GB, ~$1/hr (A100 is $4/hr but has 40GB)
    timeout=DEFAULT_TIMEOUT,
    volumes={"/vol": volume},
)
def train(
    model_id: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> None:
    """Run GRPO training inside a Modal GPU worker.

    The volume is mounted at /vol with:
    - /vol/datasets/grpo_prompts.jsonl (uploaded beforehand)
    - /vol/checkpoints/grpo_model/ (saved here after training)
    """

    resolved_model = model_id or DEFAULT_MODEL_ID
    resolved_dataset = dataset_path or DEFAULT_DATASET
    output_dir = Path("/vol/checkpoints/grpo_model")

    print(f"Starting GRPO training:")
    print(f"  Model: {resolved_model}")
    print(f"  Dataset: {resolved_dataset}")
    print(f"  Output: {output_dir}")

    grpo_main(
        model_id=resolved_model,
        dataset_path=Path(resolved_dataset),
        output_dir=output_dir,
    )

    # Commit changes to persist checkpoints
    print("Committing checkpoints to volume...")
    volume.commit()
    print("✓ Training complete! Checkpoints saved to volume.")


@app.local_entrypoint()
def main(
    model_id: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> None:
    train.remote(model_id=model_id, dataset_path=dataset_path)
