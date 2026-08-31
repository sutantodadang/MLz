#!/usr/bin/env python3
"""Search text files for lines matching a literal token.

usage: python tools/grep_zig.py <token> <file> [context]
"""
import sys

token = sys.argv[1]
path = sys.argv[2]
context = int(sys.argv[3]) if len(sys.argv) > 3 else 0
with open(path, "r", encoding="utf-8") as handle:
    lines = handle.readlines()
hits = 0
for i, line in enumerate(lines):
    if token in line:
        start = max(0, i - context)
        end = min(len(lines), i + context + 1)
        for j in range(start, end):
            marker = "->" if j == i else "  "
            print(f"{marker}{j+1}: {lines[j].rstrip()}")
        print("--")
        hits += 1
        if hits > 20:
            print("(truncated)")
            break
print(f"-- {hits} hit(s)")
