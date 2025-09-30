"""Data mining, filtering, and task generation utilities for claude--."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

API_VERSION = "2022-11-28"

if load_dotenv is not None:
    load_dotenv()
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PullFile:
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int


@dataclass
class PullRequestRecord:
    owner: str
    repo: str
    number: int
    merged_at: Optional[str]
    title: str
    body: str
    additions: int
    deletions: int
    changed_files: int
    base_ref: str
    head_ref: str
    merge_commit_sha: Optional[str]
    html_url: str
    patch_url: Optional[str]
    statuses_state: Optional[str]
    status_checks: List[Dict[str, str]] = field(default_factory=list)
    files: List[PullFile] = field(default_factory=list)

    def deletion_ratio(self) -> float:
        total = self.additions + self.deletions
        if total == 0:
            return 0.0
        return (self.deletions - self.additions) / total

    def touches_tests(self) -> bool:
        return any("test" in pf.filename.lower() for pf in self.files)


@dataclass
class PromptRecord:
    prompt: str
    repo_path: str
    allow_refactor: bool
    metadata: Dict[str, str]


def github_session() -> requests.Session:
    token = os.environ.get("GITHUB_TOKEN")
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "claude--/data-pipeline",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


def github_get(session: requests.Session, url: str, params: Optional[Dict[str, str]] = None) -> requests.Response:
    response = session.get(url, params=params)
    if response.status_code == 403:
        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            ts = datetime.utcfromtimestamp(int(reset))
            raise RuntimeError(f"GitHub rate limit hit; resets at {ts} UTC")
        raise RuntimeError(f"GitHub API 403: {response.text}")
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")
    return response


def search_pull_requests(
    session: requests.Session,
    query: str,
    since: Optional[str],
    until: Optional[str],
    max_results: int,
) -> Iterator[Dict[str, Any]]:
    date_filter = ""
    if since and until:
        date_filter = f" merged:{since}..{until}"
    elif since:
        date_filter = f" merged:>={since}"
    elif until:
        date_filter = f" merged:<={until}"
    full_query = f"{query} {date_filter}".strip()
    url = "https://api.github.com/search/issues"
    page = 1
    fetched = 0
    while fetched < max_results:
        params = {
            "q": full_query,
            "sort": "updated",
            "order": "desc",
            "per_page": "100",
            "page": str(page),
        }
        resp = github_get(session, url, params=params)
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        for item in items:
            if fetched >= max_results:
                break
            if "pull_request" not in item:
                continue
            fetched += 1
            yield item
        page += 1
        if page > 10:
            break


def fetch_pull_detail(session: requests.Session, issue_item: Dict[str, Any]) -> PullRequestRecord:
    pull_url = issue_item["pull_request"]["url"]
    pull_data = github_get(session, pull_url).json()
    owner_repo = issue_item["repository_url"].rsplit("/", 2)[-2:]
    owner, repo = owner_repo
    files = list(fetch_pull_files(session, pull_data["url"]))
    status_checks = fetch_status_checks(session, owner, repo, pull_data.get("merge_commit_sha"))
    return PullRequestRecord(
        owner=owner,
        repo=repo,
        number=pull_data["number"],
        merged_at=pull_data.get("merged_at"),
        title=pull_data.get("title", ""),
        body=pull_data.get("body") or "",
        additions=pull_data.get("additions", 0),
        deletions=pull_data.get("deletions", 0),
        changed_files=pull_data.get("changed_files", 0),
        base_ref=pull_data.get("base", {}).get("ref", ""),
        head_ref=pull_data.get("head", {}).get("ref", ""),
        merge_commit_sha=pull_data.get("merge_commit_sha"),
        html_url=pull_data.get("html_url", ""),
        patch_url=pull_data.get("patch_url"),
        statuses_state=status_checks.get("state"),
        status_checks=status_checks.get("contexts", []),
        files=files,
    )


def fetch_pull_files(session: requests.Session, base_url: str) -> Iterator[PullFile]:
    url = f"{base_url}/files"
    page = 1
    while True:
        params = {"per_page": "100", "page": str(page)}
        resp = github_get(session, url, params=params)
        data = resp.json()
        if not data:
            break
        for item in data:
            yield PullFile(
                filename=item.get("filename", ""),
                status=item.get("status", ""),
                additions=item.get("additions", 0),
                deletions=item.get("deletions", 0),
                changes=item.get("changes", 0),
            )
        if "next" not in resp.links:
            break
        page += 1


def fetch_status_checks(session: requests.Session, owner: str, repo: str, sha: Optional[str]) -> Dict[str, Any]:
    if not sha:
        return {"state": None, "contexts": []}
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/status"
    resp = github_get(session, url)
    data = resp.json()
    contexts = [
        {
            "context": ctx.get("context", ""),
            "state": ctx.get("state", ""),
            "description": ctx.get("description", ""),
        }
        for ctx in data.get("statuses", [])
    ]
    return {"state": data.get("state"), "contexts": contexts}


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def cmd_fetch_prs(args: argparse.Namespace) -> None:
    session = github_session()
    raw_records: List[Dict[str, Any]] = []
    for item in search_pull_requests(session, args.query, args.since, args.until, args.max_results):
        record = fetch_pull_detail(session, item)
        raw_records.append(asdict(record))
        print(f"fetched {record.owner}/{record.repo}#{record.number} (deletions={record.deletions}, additions={record.additions})", file=sys.stderr)
    write_jsonl(Path(args.output), raw_records)
    print(f"wrote {len(raw_records)} records to {args.output}")


def passes_filters(record: Dict[str, Any], args: argparse.Namespace) -> bool:
    deletions = int(record.get("deletions", 0))
    additions = int(record.get("additions", 0))
    changed_files = int(record.get("changed_files", 0))
    total = deletions + additions
    ratio = (deletions - additions) / total if total else 0.0
    if deletions < args.min_deletions:
        return False
    if additions > args.max_additions:
        return False
    if changed_files > args.max_files:
        return False
    if ratio < args.min_deletion_ratio:
        return False
    if args.require_status_success and record.get("statuses_state") not in ("success", "neutral"):
        return False
    files = record.get("files", [])
    if args.only_python and any(not f.get("filename", "").endswith(".py") for f in files):
        return False
    if args.exclude_tests:
        for f in files:
            name = f.get("filename", "").lower()
            if name.startswith("tests/") or "/tests/" in name or name.endswith("_test.py"):
                return False
    return True


def cmd_filter_prs(args: argparse.Namespace) -> None:
    records = list(read_jsonl(Path(args.input)))
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if passes_filters(record, args):
            record["deletion_ratio"] = PullRequestRecord(
                owner=record.get("owner", ""),
                repo=record.get("repo", ""),
                number=record.get("number", 0),
                merged_at=record.get("merged_at"),
                title=record.get("title", ""),
                body=record.get("body", ""),
                additions=record.get("additions", 0),
                deletions=record.get("deletions", 0),
                changed_files=record.get("changed_files", 0),
                base_ref=record.get("base_ref", ""),
                head_ref=record.get("head_ref", ""),
                merge_commit_sha=record.get("merge_commit_sha"),
                html_url=record.get("html_url", ""),
                patch_url=record.get("patch_url"),
                statuses_state=record.get("statuses_state"),
                status_checks=record.get("status_checks", []),
                files=[PullFile(**f) for f in record.get("files", [])],
            ).deletion_ratio()
            filtered.append(record)
    write_jsonl(Path(args.output), filtered)
    print(f"filtered {len(filtered)} of {len(records)} records -> {args.output}")


def build_prompt(record: Dict[str, Any], repo_root: Path, allow_refactor_when_tests: bool) -> Optional[PromptRecord]:
    owner = record.get("owner")
    repo = record.get("repo")
    if not owner or not repo:
        return None
    repo_path = repo_root / owner / repo
    if not repo_path.exists():
        return None
    files = record.get("files", [])
    touches_tests = any("test" in f.get("filename", "").lower() for f in files)
    allow_refactor = touches_tests and allow_refactor_when_tests
    prompt_lines = [
        f"Repository: {owner}/{repo}",
        "Task: remove dead code or redundant logic without regressing behavior.",
        f"Base ref: {record.get('base_ref', '')}",
        f"PR title: {record.get('title', '').strip()}",
    ]
    body = record.get("body", "").strip()
    if body:
        prompt_lines.append("PR body:\n" + body)
    prompt_lines.append(
        "Respond with ONLY a unified diff (diff --git ...). Do not add explanations."
    )
    metadata = {
        "owner": owner,
        "repo": repo,
        "number": str(record.get("number", "")),
        "html_url": record.get("html_url", ""),
        "merge_commit_sha": record.get("merge_commit_sha", ""),
        "deletion_ratio": f"{record.get('deletion_ratio', 0):.3f}",
    }
    return PromptRecord(
        prompt="\n\n".join(prompt_lines),
        repo_path=str(repo_path.resolve()),
        allow_refactor=allow_refactor,
        metadata=metadata,
    )


def cmd_build_prompts(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root)
    records = list(read_jsonl(Path(args.input)))
    prompts: List[Dict[str, Any]] = []
    skipped = 0
    for record in records:
        prompt_record = build_prompt(record, repo_root, args.allow_refactor_when_tests)
        if prompt_record is None:
            skipped += 1
            continue
        prompts.append({
            "prompt": prompt_record.prompt,
            "repo_path": prompt_record.repo_path,
            "allow_refactor": prompt_record.allow_refactor,
            "metadata": prompt_record.metadata,
        })
    write_jsonl(Path(args.output), prompts)
    print(f"wrote {len(prompts)} prompts to {args.output} (skipped {skipped})")


def clone_repo(source: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="synthetic-"))
    cmd = ["git", "clone", "--local", "--no-hardlinks", str(source), str(temp_dir)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return temp_dir


def run_comby(temp_repo: Path, match: str, rewrite: str, files: Sequence[str], language: Optional[str]) -> None:
    targets = list(files) if files else ["."]
    cmd = ["comby", match, rewrite, *targets, "-in-place"]
    if language:
        cmd.extend(["-matcher", language])
    subprocess.run(cmd, cwd=temp_repo, check=True)


def git_diff(temp_repo: Path) -> str:
    cmd = ["git", "diff"]
    result = subprocess.run(cmd, cwd=temp_repo, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.stdout


def cmd_generate_synthetic(args: argparse.Namespace) -> None:
    source = Path(args.repo_path)
    if not source.exists():
        raise FileNotFoundError(f"repo_path not found: {source}")
    temp_repo = clone_repo(source)
    try:
        run_comby(temp_repo, args.match, args.rewrite, args.files, args.language)
        diff = git_diff(temp_repo)
        if not diff.strip():
            print("no diff produced; nothing written", file=sys.stderr)
            return
        prompt = args.prompt or (
            "Repository: {repo}\nTask: apply the described structural cleanup and respond with diff only."
        ).format(repo=source.name)
        record = {
            "prompt": prompt,
            "repo_path": str(source.resolve()),
            "allow_refactor": args.allow_refactor,
            "gold_diff": diff,
            "metadata": {
                "synthetic": "comby",
                "match": args.match,
                "rewrite": args.rewrite,
                "files": args.files,
                "language": args.language,
            },
        }
        output = Path(args.output)
        if output.exists() and not args.append:
            raise FileExistsError(f"{output} exists; use --append to add")
        mode = "a" if args.append else "w"
        with output.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote synthetic example to {output}")
    finally:
        shutil.rmtree(temp_repo, ignore_errors=True)


def add_fetch_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fetch-prs", help="Fetch merged PR metadata from GitHub")
    parser.add_argument("--query", required=True, help="GitHub search query, e.g. 'language:python label:refactor'")
    parser.add_argument("--since", help="ISO date lower bound for merged PRs")
    parser.add_argument("--until", help="ISO date upper bound for merged PRs")
    parser.add_argument("--max-results", type=int, default=200, help="Maximum PRs to fetch (<=1000)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "raw_prs.jsonl"))
    parser.set_defaults(func=cmd_fetch_prs)


def add_filter_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("filter-prs", help="Filter raw PR metadata")
    parser.add_argument("--input", required=True, help="Path to raw PR jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "filtered_prs.jsonl"))
    parser.add_argument("--min-deletions", type=int, default=10)
    parser.add_argument("--max-additions", type=int, default=150)
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--min-deletion-ratio", type=float, default=0.1)
    parser.add_argument("--require-status-success", action="store_true")
    parser.add_argument("--only-python", action="store_true")
    parser.add_argument("--exclude-tests", action="store_true")
    parser.set_defaults(func=cmd_filter_prs)


def add_build_prompts_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("build-prompts", help="Construct GRPO prompt dataset")
    parser.add_argument("--input", required=True, help="Filtered PR jsonl path")
    parser.add_argument("--repo-root", required=True, help="Directory containing local repo clones")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "grpo_prompts.jsonl"))
    parser.add_argument("--allow-refactor-when-tests", action="store_true")
    parser.set_defaults(func=cmd_build_prompts)


def add_generate_synthetic_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate-synthetic", help="Apply comby pattern to build synthetic SFT pairs")
    parser.add_argument("--repo-path", required=True, help="Path to local git repo")
    parser.add_argument("--match", required=True, help="Comby match pattern")
    parser.add_argument("--rewrite", required=True, help="Comby rewrite template")
    parser.add_argument("--files", nargs="*", default=["."], help="File globs or paths for comby")
    parser.add_argument("--language", help="Comby matcher language, e.g. 'python'")
    parser.add_argument("--prompt", help="Override prompt text")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "synthetic.jsonl"))
    parser.add_argument("--allow-refactor", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.set_defaults(func=cmd_generate_synthetic)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data pipeline utilities for claude--")
    subparsers = parser.add_subparsers(dest="command")
    add_fetch_parser(subparsers)
    add_filter_parser(subparsers)
    add_build_prompts_parser(subparsers)
    add_generate_synthetic_parser(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        parser.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
