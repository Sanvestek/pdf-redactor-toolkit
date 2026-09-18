#!/usr/bin/env python3
"""Claude Code PreToolUse hook (see .claude/settings.json): before Claude's Bash tool
runs a git commit or git push, re-run scripts/pre-commit's real-PII checks against it.

This exists alongside the actual git hook (scripts/pre-commit, installed via
`ln -sf ../../scripts/pre-commit .git/hooks/pre-commit`) as a second, independent gate:
it still catches a bad commit/push even when that git hook was never installed (a fresh
clone/environment) or was bypassed with --no-verify. Deliberately avoids depending on
jq - it isn't guaranteed to be installed - and parses the hook's stdin JSON directly.
"""
import json
import subprocess
import sys


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if "git commit" in command:
        mode = "commit"
    elif "git push" in command:
        mode = "push"
    else:
        return 0

    return subprocess.call([sys.executable, "scripts/pre-commit", mode])


if __name__ == "__main__":
    sys.exit(main())
