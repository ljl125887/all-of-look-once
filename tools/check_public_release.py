"""Fail closed on unexpected files, common secrets, media and private paths."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".json", ".yml", ".yaml"}
ALLOWED_SPECIAL = {"LICENSE", ".gitignore", ".gitkeep"}
PATTERNS = {
    "credential URL": re.compile(r"[a-z]+://[^\s/]+:[^\s/@]+@", re.I),
    "private IPv4": re.compile(r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
    "local absolute drive": re.compile(r"\b[A-Za-z]:[\\/]"),
    "token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"),
}


def violations(path: Path, text: str) -> list[str]:
    found = [label for label, pattern in PATTERNS.items() if pattern.search(text)]
    if path.suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_SPECIAL:
        found.append("unexpected file type")
    if path.name == "config.json" or any(part in {"data", "results", "logs"} for part in path.parts):
        found.append("runtime/private data")
    return found


def scan() -> list[str]:
    failures = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in {".git", "__pycache__", ".venv"} for part in relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"{relative}: symlink")
            continue
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            failures.append(f"{relative}: non-text or unreadable")
            continue
        failures.extend(f"{relative}: {label}" for label in violations(relative, content))
    return failures


if __name__ == "__main__":
    failures = scan()
    if failures:
        print("Release scan failed:\n" + "\n".join(failures))
        sys.exit(1)
    print("Public release scan passed.")
