"""Modal entry point for GRPO training with GPU support."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import modal

MOUNT_PATH = "/root/clobber"

# Exclude large directories and files we don't need in the container
# Note: add_local_dir must come LAST (or use copy=True)
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
    .add_local_dir(
        Path.cwd(),
        remote_path=MOUNT_PATH,
        ignore=[
            "repos/**",          # Large cloned repos - verifier re-clones fresh
            "checkpoints/**",    # Model weights - saved to volume instead
            ".venv/**",          # Virtual environment
            ".git/**",           # Git history
            "__pycache__/**",    # Python cache
            "*.pyc",             # Compiled Python
            ".pytest_cache/**",  # Pytest cache
            ".ruff_cache/**",    # Ruff cache
            "data/*.jsonl",      # Large datasets - uploaded to volume separately
        ],
    )
)

app = modal.App("clobber-grpo-trainer")

if MOUNT_PATH not in sys.path:
    sys.path.append(MOUNT_PATH)

from grpo_trainer import main as grpo_main


DEFAULT_MODEL_ID = os.environ.get("CLOBBER_GRPO_MODEL", "Qwen/Qwen1.5-0.5B")
DEFAULT_DATASET = os.environ.get("CLOBBER_GRPO_DATASET", "/vol/datasets/grpo_prompts.jsonl")
DEFAULT_TIMEOUT = int(os.environ.get("CLOBBER_GRPO_TIMEOUT", str(60 * 60 * 4)))  # 4 hours

# Supported models (set via CLOBBER_GRPO_MODEL):
# - Qwen/Qwen1.5-0.5B (fast iteration, fits on A10G)
# - Qwen/Qwen2.5-Coder-7B-Instruct (production, needs A100)
# - Qwen/Qwen3-Coder-30B-A3B-Instruct (SOTA, needs H100 or multi-GPU)

# Create or reference the shared volume
volume = modal.Volume.from_name("clobber-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
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
