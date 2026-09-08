#!/usr/bin/env python3
"""Fail on common release-tree leaks and packaging omissions."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".infra-team", "build", "dist", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".html", ".css", ".js", ".sh"}
FORBIDDEN_SUFFIXES = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt"}
PATTERNS = {
    "absolute macOS user path": re.compile(r"/Users/[^/\s]+"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    failures: list[str] = []
    required = [
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "src/infra_team/defaults/autonomy-local.yaml",
        "src/infra_team/registry/qwen38-dflash2-v1.yaml",
        "examples/qwen38-dflash2-m5pro/benchmark_samples.json",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required release file: {relative}")

    for path in ROOT.rglob("*"):
        if path.is_symlink() and ".git" not in path.parts:
            failures.append(f"symbolic links are not allowed in the release tree: {path.relative_to(ROOT)}")

    for path in iter_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"model or weight file must not be released: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            failures.append(f"unexpected file larger than 5 MB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if relative.parts[0] == "tests" or relative == Path("scripts/check-release.py"):
                continue
            for name, pattern in PATTERNS.items():
                if name == "absolute macOS user path" and (
                    "re.compile" in line or "re.sub" in line
                ):
                    continue
                if pattern.search(line):
                    failures.append(f"{name} in {relative}:{line_number}")

    if failures:
        print("release check failed:")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
