"""TRL-based GRPO training loop wired into the verifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from verifier import score_patch

MODEL_ID_SMALL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_ID_TARGET = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DATA_PATH = Path("data/grpo_prompts.jsonl")


def load_model(model_id: str = MODEL_ID_SMALL):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_prompts(dataset_path: Path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prompt dataset missing: {dataset_path}")
    dataset = load_dataset("json", data_files={"train": str(dataset_path)})
    return dataset["train"]


def build_config() -> GRPOConfig:
    return GRPOConfig(
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=6,
        beta=0.02,
        generation_batch_size=6,
        max_prompt_length=4096,
        max_completion_length=768,
        logging_steps=10,
    )


def reward_function(
    prompts: List[str] | None = None,
    completions: List[str] | None = None,
    completion_ids: List | None = None,
    trainer_state=None,
    **kwargs,
) -> List[float]:
    """Reward function for GRPO trainer.

    Must accept keyword arguments as specified by TRL's GRPOTrainer API.
    Additional dataset columns (like repo_path, allow_refactor) are passed via **kwargs.
    """
    if completions is None:
        raise ValueError("completions must be provided")

    rewards: List[float] = []

    # Extract dataset columns from kwargs
    repo_paths = kwargs.get("repo_path", [])
    allow_refactors = kwargs.get("allow_refactor", [False] * len(completions))

    for i, completion in enumerate(completions):
        repo_path = repo_paths[i] if i < len(repo_paths) else None
        allow_refactor = allow_refactors[i] if i < len(allow_refactors) else False

        if repo_path is None:
            # Skip if no repo_path provided
            rewards.append(0.0)
            continue

        diff_text = completion
        score, _ = score_patch(repo_path, diff_text, allow_refactor)
        rewards.append(score)

    return rewards


def main(
    model_id: str = MODEL_ID_SMALL,
    dataset_path: Path = DATA_PATH,
    output_dir: Path | None = None,
):
    dataset_path = Path(dataset_path)
    prompts = load_prompts(dataset_path)
    model, tokenizer = load_model(model_id)
    config = build_config()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=prompts,
        reward_funcs=[reward_function],  # Note: reward_funcs expects a list
    )

    trainer.train()

    # Use provided output_dir or default to local checkpoints
    if output_dir is None:
        output_dir = Path("checkpoints") / "grpo_model"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training harness for claude--")
    parser.add_argument("--model-id", default=MODEL_ID_SMALL, help="HF model id (default: Qwen2.5 7B)")
    parser.add_argument(
        "--dataset",
        default=str(DATA_PATH),
        help="Path to prompt dataset jsonl (default: data/grpo_prompts.jsonl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(model_id=args.model_id, dataset_path=Path(args.dataset))
