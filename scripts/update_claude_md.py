#!/usr/bin/env python3
"""
Update the "Recent Development Log" section of CLAUDE.md after each git commit.

Called automatically by .git/hooks/post-commit.
Can also be run manually: python scripts/update_claude_md.py
"""

import os
import re
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAUDE_MD = os.path.join(REPO_ROOT, "CLAUDE.md")

SECTION_HEADER = "## Recent Development Log"
AUTO_TAG = "<!-- AUTO-UPDATED by post-commit hook - DO NOT EDIT MANUALLY -->"

# Number of recent commits to include
N_COMMITS = 7


def run_cmd(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO_ROOT)
    return result.stdout.strip()


def get_recent_commits(n: int = N_COMMITS) -> list[dict]:
    """Return list of {hash, date, subject, body} for the last n commits."""
    # Use a separator that won't appear in commit messages
    sep = "<<<COMMIT_SEP>>>"
    fmt = f"{sep}%H|%ad|%s%n%b"
    raw = run_cmd(f'git log -{n} --format="{fmt}" --date=short')

    commits = []
    for block in raw.split(sep):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if not lines:
            continue
        header = lines[0]
        parts = header.split("|", 2)
        if len(parts) < 3:
            continue
        commit_hash, date, subject = parts[0].strip(), parts[1].strip(), parts[2].strip()
        body = "\n".join(lines[1:]).strip()
        commits.append({"hash": commit_hash[:8], "date": date, "subject": subject, "body": body})

    return commits


def format_log_section(commits: list[dict]) -> str:
    """Format commits into a readable CLAUDE.md section."""
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        SECTION_HEADER,
        AUTO_TAG,
        f"_Last updated: {updated_at}_",
        "",
    ]

    for i, c in enumerate(commits):
        lines.append(f"### [{c['hash']}] {c['date']} — {c['subject']}")
        if c["body"]:
            # Indent body for readability; keep structured sections (## Changes etc.)
            for line in c["body"].splitlines():
                lines.append(line)
        lines.append("")

    return "\n".join(lines)


def update_claude_md(section_text: str) -> None:
    with open(CLAUDE_MD, "r") as f:
        content = f.read()

    if SECTION_HEADER in content:
        # Replace existing section (from header to end of file or next top-level ##)
        pattern = re.compile(
            rf"{re.escape(SECTION_HEADER)}.*?(?=\n## (?!#)|\Z)",
            re.DOTALL,
        )
        new_content = pattern.sub(section_text.rstrip(), content)
        if new_content == content:
            # Fallback: just replace from header to end
            idx = content.index(SECTION_HEADER)
            new_content = content[:idx] + section_text
    else:
        # Append new section
        new_content = content.rstrip() + "\n\n" + section_text + "\n"

    with open(CLAUDE_MD, "w") as f:
        f.write(new_content)


def main():
    commits = get_recent_commits(N_COMMITS)
    if not commits:
        print("No commits found.", file=sys.stderr)
        sys.exit(0)

    section = format_log_section(commits)
    update_claude_md(section)
    print(f"CLAUDE.md updated with {len(commits)} recent commits.")


if __name__ == "__main__":
    main()
