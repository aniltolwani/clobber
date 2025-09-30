"""Shared tool action schema for all agents and baselines.

This module defines the dataclasses passed between the supervisor and
coding agents plus a registry describing every action for LLM tool
calling APIs (OpenAI/Qwen style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ------------------------------
# Input payload dataclasses
# ------------------------------


@dataclass
class Shell:
    """Run an arbitrary shell command inside the repo sandbox."""

    cmd: str
    timeout_seconds: Optional[int] = None


@dataclass
class RGSearch:
    """ripgrep search"""

    pattern: str
    globs: Optional[List[str]] = None
    flags: Optional[List[str]] = None


@dataclass
class EditSad:
    """Structural-aware text edit via `sad`."""

    path_glob: str
    find: str
    replace: str
    preview: bool = True


@dataclass
class EditComby:
    """Structural edit with comby templates."""

    match: str
    rewrite: str
    files: List[str]
    language: Optional[str] = None
    timeout_seconds: Optional[int] = None


@dataclass
class ApplyPatch:
    """Apply a unified diff using `git apply`."""

    unified_diff: str


@dataclass
class RuffCheck:
    paths: Optional[List[str]] = None
    select: Optional[List[str]] = None
    fix: bool = False
    format: bool = False


@dataclass
class PyrightCheck:
    paths: Optional[List[str]] = None


@dataclass
class DeptryCheck:
    path: str = "."


@dataclass
class PydepsMetrics:
    path: str = "."
    max_depth: int = 2


@dataclass
class PytestImpacted:
    args: Optional[List[str]] = None
    xdist: bool = True
    maxfail: int = 1


@dataclass
class PytestFull:
    args: Optional[List[str]] = None
    xdist: bool = True
    maxfail: int = 1


@dataclass
class MutmutSample:
    paths: Optional[List[str]] = None
    timeout_seconds: int = 60
    max_mutants: int = 10


# ------------------------------
# Standardized tool result payload
# ------------------------------


@dataclass
class ToolResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    meta: Optional[Dict[str, Any]] = None


# ------------------------------
# Tool specifications for LLM APIs
# ------------------------------


def _string_array_schema(description: str) -> Dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "string"},
    }


def _boolean_schema(description: str) -> Dict[str, Any]:
    return {
        "type": "boolean",
        "description": description,
    }


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "shell": {
        "description": "Execute a shell command inside the repository sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "required": ["cmd"],
        },
    },
    "rg_search": {
        "description": "Search code using ripgrep; respects .gitignore.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "globs": _string_array_schema("File globs to include/exclude."),
                "flags": _string_array_schema("ripgrep CLI flags, e.g. ['-n', '-i']"),
            },
            "required": ["pattern"],
        },
    },
    "edit_sad": {
        "description": "Batch edit files using sad.",
        "parameters": {
            "type": "object",
            "properties": {
                "path_glob": {"type": "string"},
                "find": {"type": "string"},
                "replace": {"type": "string"},
                "preview": _boolean_schema("Preview diff without applying."),
            },
            "required": ["path_glob", "find", "replace"],
        },
    },
    "edit_comby": {
        "description": "Structural code edit with comby templates.",
        "parameters": {
            "type": "object",
            "properties": {
                "match": {"type": "string"},
                "rewrite": {"type": "string"},
                "files": _string_array_schema("File glob patterns"),
                "language": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "required": ["match", "rewrite", "files"],
        },
    },
    "apply_patch": {
        "description": "Apply a unified diff to the repo using git apply.",
        "parameters": {
            "type": "object",
            "properties": {
                "unified_diff": {"type": "string"},
            },
            "required": ["unified_diff"],
        },
    },
    "ruff_check": {
        "description": "Run ruff lint/format with optional autofix.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": _string_array_schema("Paths to lint"),
                "select": _string_array_schema("Explicit ruff rule list"),
                "fix": _boolean_schema("Enable autofix"),
                "format": _boolean_schema("Run ruff format after lint"),
            },
        },
    },
    "pyright_check": {
        "description": "Run Pyright type checker and report errors.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": _string_array_schema("Paths to check"),
            },
        },
    },
    "deptry_check": {
        "description": "Check dependency usage with deptry.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
        },
    },
    "pydeps_metrics": {
        "description": "Compute import graph metrics via pydeps.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_depth": {"type": "integer", "minimum": 1},
            },
        },
    },
    "pytest_impacted": {
        "description": "Run impacted tests via pytest-testmon.",
        "parameters": {
            "type": "object",
            "properties": {
                "args": _string_array_schema("Extra pytest args"),
                "xdist": _boolean_schema("Use pytest-xdist"),
                "maxfail": {"type": "integer", "minimum": 1},
            },
        },
    },
    "pytest_full": {
        "description": "Run full pytest suite.",
        "parameters": {
            "type": "object",
            "properties": {
                "args": _string_array_schema("Extra pytest args"),
                "xdist": _boolean_schema("Use pytest-xdist"),
                "maxfail": {"type": "integer", "minimum": 1},
            },
        },
    },
    "mutmut_sample": {
        "description": "Run a sampled mutation test batch on changed files.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": _string_array_schema("Specific file paths"),
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "max_mutants": {"type": "integer", "minimum": 1},
            },
        },
    },
}


def tool_names() -> List[str]:
    """Return the canonical tool list."""

    return list(TOOL_SPECS.keys())


def tool_spec(tool: str) -> Dict[str, Any]:
    """Return the JSON schema block for a tool."""

    if tool not in TOOL_SPECS:
        raise KeyError(f"Unknown tool: {tool}")
    return TOOL_SPECS[tool]


__all__ = [
    "Shell",
    "RGSearch",
    "EditSad",
    "EditComby",
    "ApplyPatch",
    "RuffCheck",
    "PyrightCheck",
    "DeptryCheck",
    "PydepsMetrics",
    "PytestImpacted",
    "PytestFull",
    "MutmutSample",
    "ToolResult",
    "TOOL_SPECS",
    "tool_names",
    "tool_spec",
]
