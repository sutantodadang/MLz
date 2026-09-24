#!/usr/bin/env python3
"""Search vendored llama.cpp sources for an architecture token."""
import os
import sys

root = sys.argv[1]
token = sys.argv[2]
hits = 0
for base, _dirs, files in os.walk(root):
    for name in files:
        if not name.endswith((".h", ".cpp", ".c", ".cu")):
            continue
        path = os.path.join(base, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for lineno, line in enumerate(handle, 1):
                    if token in line:
                        print(f"{path}:{lineno}: {line.rstrip()}")
                        hits += 1
                        if hits > 80:
                            print("... (truncated)")
                            raise SystemExit(0)
        except OSError:
            continue
print(f"-- {hits} hits")
