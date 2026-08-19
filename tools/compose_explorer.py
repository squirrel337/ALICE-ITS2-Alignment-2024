#!/usr/bin/env python3
"""Compose docs/alignment-explorer.html from the shell and the extracted payload.

    python3 tools/compose_explorer.py

The shell carries a single __DATA__ placeholder inside <script type="application/json">.
Substitution is a literal str.replace: the payload holds backslashes and megabytes of
digits, so a regex or a shell heredoc mangles it.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHELL = ROOT / "docs" / "_explorer_shell.html"
DATA = ROOT / "docs" / "explorer-data.json"
OUT = ROOT / "docs" / "alignment-explorer.html"

shell = SHELL.read_text(encoding="utf-8")
if shell.count("__DATA__") != 1:
    sys.exit(f"{SHELL}: expected exactly one __DATA__ placeholder")
if not DATA.exists():
    sys.exit(f"{DATA}: missing. Run tools/build_explorer_data.py first.")

OUT.write_text(shell.replace("__DATA__", DATA.read_text(encoding="utf-8")), encoding="utf-8")
print(f"[compose] {OUT.relative_to(ROOT)}  {OUT.stat().st_size/1e6:.2f} MB")
