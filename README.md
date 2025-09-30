# claude--

delete-first coding agent research harness

## layout
- `tool_schema.py` – shared tool/action schema for all agents and baselines
- `verifier.py` – low-latency gate + reward calculator (clone/apply/tests/metrics)
- `grpo_trainer.py` – TRL GRPO loop that plugs the verifier
- `data_pipeline.py` – GitHub mining, filtering, and synthetic task generator
- `baseline_runner.py` – baseline harness (heuristic + pluggable LLM agents) scored with `verifier.py`

## quick start
1. install system deps: `ruff`, `pyright`, `deptry`, `pydeps`, `pytest-testmon`, `pytest-xdist`, `mutmut`, `ripgrep`, `sad`, `comby`.
2. fetch + filter PRs: `python data_pipeline.py fetch-prs --query "is:pr language:python label:refactor" --since 2024-01-01` then `python data_pipeline.py filter-prs --input data/raw_prs.jsonl`.
3. build prompts: `python data_pipeline.py build-prompts --input data/filtered_prs.jsonl --repo-root /path/to/repos` -> `data/grpo_prompts.jsonl`.
4. (optional) synthesize hard examples with `python data_pipeline.py generate-synthetic --repo-path repo --match 'match' --rewrite 'rewrite' --language python`.
5. run `python grpo_trainer.py` to fine-tune the policy (defaults to Qwen2.5-Coder-7B) using prompts.
6. benchmark baselines: `python baseline_runner.py --dataset data/grpo_prompts.jsonl --baselines heuristic --print-summaries`.
7. checkpoints land under `checkpoints/grpo_model`.

## verifier contract
- clones the target repo (`repo_path`) into a temp dir
- applies candidate unified diff, runs `py_compile`, `pytest --testmon`, `pyright`
- computes deltas for Ruff (F401/F841), deptry unused deps, pydeps graph edges/nodes
- returns a scalar reward plus diagnostics dictionary; gates failing steps at -1

## data pipeline overview
- `fetch-prs` hits the GitHub Search API (auto-loading `.env` if present for `GITHUB_TOKEN`) and stores enriched PR metadata in JSONL
- `filter-prs` enforces deletion ratio, additions/files thresholds, optional CI success, and test exclusion (defaults tuned for real-world PRs: 10% minimum deletion ratio, 150 max additions)
- `build-prompts` maps PRs to local clones to emit GRPO-ready prompts referencing repo paths
- `generate-synthetic` runs comby in a temp clone to create structural SFT pairs without touching the source repo

## baseline runner
- `python baseline_runner.py --dataset data/grpo_prompts.jsonl --baselines heuristic --print-summaries`
  - clones repos per task into temp dirs, applies the heuristic tool chain, and scores diffs with `verifier.py`
- additional placeholders (`gpt4o`, `aider`, `openhands`, `qwen`) are wired but require you to replace the stub in `baseline_runner.py` with a real implementation that returns a unified diff
- results are written to `data/baseline_results.csv` by default (CSV with reward + diagnostics)

## next steps
- extend diagnostics logging (gate pass rate, delta curves) for training dashboards
- wire optional style judge + sampled `mutmut` runs when infrastructure is ready
