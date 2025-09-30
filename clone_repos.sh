#!/bin/bash
# Clone repositories for training data generation
# Usage: ./clone_repos.sh data/filtered_prs.jsonl

set -e

FILTERED_PRS="${1:-data/filtered_prs_large.jsonl}"

if [ ! -f "$FILTERED_PRS" ]; then
    echo "Error: $FILTERED_PRS not found"
    exit 1
fi

# Extract unique owner/repo pairs from the filtered PRs
echo "Extracting unique repos from $FILTERED_PRS..."
REPOS=$(python3 -c "
import json, sys
repos = set()
with open('$FILTERED_PRS') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            repos.add((data['owner'], data['repo']))
for owner, repo in sorted(repos):
    print(f'{owner}/{repo}')
")

echo "Found $(echo "$REPOS" | wc -l) unique repositories"
echo ""

# Clone each repo
for repo_path in $REPOS; do
    owner=$(echo "$repo_path" | cut -d'/' -f1)
    repo=$(echo "$repo_path" | cut -d'/' -f2)

    target_dir="repos/$owner/$repo"

    if [ -d "$target_dir/.git" ]; then
        echo "✓ $repo_path already cloned, skipping"
    else
        echo "Cloning $repo_path..."
        mkdir -p "repos/$owner"
        git clone "https://github.com/$owner/$repo.git" "$target_dir" || {
            echo "  ⚠️  Failed to clone $repo_path"
        }
    fi
done

echo ""
echo "✅ Clone complete. Repos are in repos/ (gitignored)"
echo ""
echo "Next steps:"
echo "  1. Build prompts: python data_pipeline.py build-prompts --input $FILTERED_PRS --repo-root repos --output data/grpo_prompts.jsonl"
echo "  2. Test baselines: python baseline_runner.py --dataset data/grpo_prompts.jsonl --baselines heuristic --print-summaries"
