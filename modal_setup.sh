#!/bin/bash
# Modal setup and dataset upload for GRPO training
# Run this once to set up Modal infrastructure

set -e

echo "🚀 Modal Setup for Clobber GRPO Training"
echo ""

# Check if modal is authenticated
if ! uv run modal token list &>/dev/null; then
    echo "❌ Modal not authenticated"
    echo ""
    echo "Please run:"
    echo "  uv run modal token new"
    echo ""
    echo "This will open a browser to authenticate with Modal."
    exit 1
fi

echo "✓ Modal authenticated"
echo ""

# Create volume
echo "Creating Modal volume 'clobber-data'..."
if uv run modal volume create clobber-data 2>&1 | grep -q "already exists"; then
    echo "✓ Volume 'clobber-data' already exists"
else
    echo "✓ Volume 'clobber-data' created"
fi
echo ""

# Check if dataset exists
DATASET="data/grpo_prompts.jsonl"
if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset not found: $DATASET"
    echo ""
    echo "Please generate it first:"
    echo "  ./clone_repos.sh data/filtered_prs_large.jsonl"
    echo "  python data_pipeline.py build-prompts --input data/filtered_prs_large.jsonl --repo-root repos --output $DATASET"
    exit 1
fi

echo "✓ Dataset found: $DATASET"
echo ""

# Upload dataset to volume
echo "Uploading dataset to Modal volume..."
uv run python - <<'PY'
import modal
from pathlib import Path

vol = modal.Volume.from_name("clobber-data")
dataset = Path("data/grpo_prompts.jsonl")

print(f"Uploading {dataset} ({dataset.stat().st_size / 1024:.1f} KB)...")
with vol.batch_upload() as batch:
    batch.put_file(str(dataset), "/datasets/grpo_prompts.jsonl")

print("✓ Upload complete")
PY

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start training:"
echo "  uv run modal run modal_grpo.py"
echo ""
echo "To monitor training:"
echo "  # Logs stream to your terminal automatically"
echo ""
echo "To download checkpoints after training:"
echo "  uv run modal volume get clobber-data /checkpoints/grpo_model ./checkpoints/grpo_model"
echo ""
echo "Environment variables (optional):"
echo "  CLOBBER_GRPO_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct  # Use larger model"
echo "  CLOBBER_GRPO_TIMEOUT=14400                            # 4 hours (default)"