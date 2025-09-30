# Modal GPU Training Guide

## Quick Start

### 1. Authenticate with Modal

```bash
uv run modal token new
```

This opens a browser to authenticate. Token is saved to `~/.modal.toml`.

### 2. Create Volume & Upload Dataset

```bash
# Create persistent storage
uv run modal volume create clobber-data

# Upload training data
uv run python - <<'PY'
import modal
from pathlib import Path

vol = modal.Volume.from_name("clobber-data")
dataset = Path("data/grpo_prompts.jsonl")

with vol.batch_upload() as batch:
    batch.put_file(str(dataset), "/datasets/grpo_prompts.jsonl")
print("✓ Dataset uploaded")
PY
```

**Or use the helper script:**
```bash
./modal_setup.sh
```

### 3. Start Training

```bash
uv run modal run modal_grpo.py
```

Logs stream to your terminal. Training runs on an A100 GPU with 4-hour timeout.

### 4. Download Checkpoints

After training completes:

```bash
uv run modal volume get clobber-data \
  /checkpoints/grpo_model \
  ./checkpoints/grpo_model
```

---

## Configuration

### Environment Variables

```bash
# Use larger model (30B instead of 7B)
export CLOBBER_GRPO_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Increase timeout to 8 hours
export CLOBBER_GRPO_TIMEOUT=28800

# Use different dataset path (if uploaded elsewhere)
export CLOBBER_GRPO_DATASET="/vol/datasets/my_custom_prompts.jsonl"

uv run modal run modal_grpo.py
```

### GPU Options

Edit `modal_grpo.py` to change GPU type:

```python
@stub.function(
    gpu="A100-80GB",  # Options: A100, A100-80GB, A10G, T4
    timeout=DEFAULT_TIMEOUT,
    volumes={"/vol": volume},
)
```

---

## Volume Management

### List contents

```bash
uv run modal volume ls clobber-data
uv run modal volume ls clobber-data /datasets
uv run modal volume ls clobber-data /checkpoints
```

### Upload additional files

```bash
uv run modal volume put clobber-data \
  data/extra_prompts.jsonl \
  /datasets/extra_prompts.jsonl
```

### Download files

```bash
uv run modal volume get clobber-data \
  /checkpoints/grpo_model \
  ./local_checkpoints
```

### Delete volume (warning: permanent!)

```bash
uv run modal volume delete clobber-data
```

---

## Cost Estimation

**A100 GPU pricing (approximate):**
- ~$4/hour for standard A100 (40GB)
- ~$6/hour for A100-80GB

**Typical training time:**
- 7B model on 17 prompts: ~30-60 minutes
- 30B model on 17 prompts: ~2-4 hours

**Example costs:**
- Quick test (7B, 1 hour): ~$4
- Full run (30B, 4 hours): ~$24

---

## Monitoring

### View logs in real-time

Logs automatically stream when you run `modal run modal_grpo.py`.

### Check running apps

```bash
uv run modal app list
```

### Stop a running job

```bash
uv run modal app stop clobber-grpo-trainer
```

---

## Architecture

```
Local Machine                  Modal Cloud
─────────────                  ───────────

data/grpo_prompts.jsonl  →  Upload  →  /vol/datasets/grpo_prompts.jsonl
                                              ↓
                                         [A100 GPU]
                                         grpo_trainer.py
                                              ↓
                                      /vol/checkpoints/grpo_model/
                                              ↓
                            Download  ←  volume.commit()
                                              ↓
./checkpoints/grpo_model/
```

**Key points:**
- Volume persists between runs
- Repos are re-cloned fresh by verifier during training
- Only dataset + checkpoints need to be in volume
- Your local `repos/` directory is never uploaded

---

## Troubleshooting

### "Dataset not found"

**Error:** `FileNotFoundError: Prompt dataset missing: /vol/datasets/grpo_prompts.jsonl`

**Fix:** Upload the dataset:
```bash
uv run python - <<'PY'
import modal
from pathlib import Path
vol = modal.Volume.from_name("clobber-data")
with vol.batch_upload() as batch:
    batch.put_file("data/grpo_prompts.jsonl", "/datasets/grpo_prompts.jsonl")
PY
```

### "Token missing"

**Error:** `Token missing. Could not authenticate client.`

**Fix:** Run `uv run modal token new` to authenticate.

### "Out of memory"

**Error:** CUDA OOM during training

**Fixes:**
1. Use 8-bit quantization (edit `grpo_trainer.py` to add `load_in_8bit=True`)
2. Reduce batch size (edit `build_config()` to use smaller `per_device_train_batch_size`)
3. Use A100-80GB instead of A100-40GB

### Timeout before completion

**Error:** Job exceeds 4-hour timeout

**Fix:** Increase timeout:
```bash
export CLOBBER_GRPO_TIMEOUT=28800  # 8 hours
uv run modal run modal_grpo.py
```

---

## Advanced: Custom Training Scripts

You can run any Python script on Modal with the volume:

```python
import modal

stub = modal.Stub("my-custom-script")
volume = modal.Volume.from_name("clobber-data")

@stub.function(
    gpu="A100",
    volumes={"/vol": volume},
)
def my_train():
    from pathlib import Path
    from grpo_trainer import main as grpo_main

    # Custom config
    grpo_main(
        model_id="Qwen/Qwen2.5-Coder-14B",
        dataset_path=Path("/vol/datasets/grpo_prompts.jsonl"),
        output_dir=Path("/vol/checkpoints/custom_model"),
    )
    volume.commit()

if __name__ == "__main__":
    with stub.run():
        my_train.remote()
```

Run with:
```bash
uv run modal run my_custom_script.py
```

---

## Next Steps

1. **Scale up data:** Generate 50-100 training prompts
2. **Monitor metrics:** Add WandB logging to track training curves
3. **Experiment with models:** Try Qwen3-Coder-30B for better performance
4. **Iterate on reward:** Tune weights in `verifier.py` based on results

See `README.md` for full project documentation.