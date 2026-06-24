#!/usr/bin/env python3
"""SIMD micro-benchmark regression gate.

Runs `zig build bench -Dsimd-backend=true -- --json`, which emits one NDJSON
line per kernel ({"kernel": "...", "metric": <GFLOPS|GigaOps/s>}), and either:

  --update   record the current numbers as the baseline (bench/baseline.json)
  (default)  compare against the baseline; FAIL if any kernel regresses by more
             than --tol (default 0.20 = 20%), or a baseline kernel is missing
  --smoke    no baseline; just assert every kernel produced a finite metric > 0
             (hardware-independent liveness check — use this in CI, where the
             absolute GFLOPS vary per runner)

Absolute GFLOPS are machine-specific, so the regression gate (--update/default)
is meant for a consistent local/self-hosted machine; CI uses --smoke.
"""
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "bench", "baseline.json")
DEFAULT_CMD = ["zig", "build", "bench", "-Dsimd-backend=true", "--", "--json"]


def run_bench(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    results = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
            results[o["kernel"]] = float(o["metric"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    if not results:
        sys.stderr.write("bench produced no kernel results. Output:\n" + out + "\n")
        sys.exit(2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="write current numbers as baseline")
    ap.add_argument("--smoke", action="store_true", help="liveness only: every metric finite > 0")
    ap.add_argument("--tol", type=float, default=0.20, help="allowed regression fraction (default 0.20)")
    ap.add_argument("--cmd", nargs=argparse.REMAINDER, help="override bench command")
    args = ap.parse_args()

    cmd = args.cmd if args.cmd else DEFAULT_CMD
    cur = run_bench(cmd)

    if args.update:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as f:
            json.dump(cur, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote baseline: {len(cur)} kernels -> {BASELINE}")
        return

    if args.smoke:
        bad = [k for k, v in cur.items() if not math.isfinite(v) or v <= 0]
        for k in sorted(cur):
            print(f"  {k:<32} {cur[k]:>8.1f}")
        if bad:
            print(f"SMOKE FAIL: {len(bad)} kernel(s) with non-finite/zero metric: {bad}")
            sys.exit(1)
        print(f"SMOKE OK: {len(cur)} kernels all live")
        return

    # regression check
    if not os.path.exists(BASELINE):
        print(f"no baseline at {BASELINE}; run with --update first")
        sys.exit(2)
    with open(BASELINE) as f:
        base = json.load(f)

    regressions, missing = [], []
    for k, b in base.items():
        if k not in cur:
            missing.append(k)
            continue
        c = cur[k]
        if b > 0 and c < b * (1.0 - args.tol):
            regressions.append((k, b, c, (c - b) / b * 100.0))

    for k in sorted(cur):
        b = base.get(k)
        tag = "NEW" if b is None else f"{(cur[k]-b)/b*100:+.1f}%"
        print(f"  {k:<32} {cur[k]:>8.1f}  ({tag})")

    if missing:
        print(f"FAIL: {len(missing)} baseline kernel(s) missing from run: {missing}")
    if regressions:
        print(f"FAIL: {len(regressions)} kernel(s) regressed > {args.tol*100:.0f}%:")
        for k, b, c, pct in regressions:
            print(f"  {k}: {b:.1f} -> {c:.1f} ({pct:.1f}%)")
    if missing or regressions:
        sys.exit(1)
    print(f"OK: {len(base)} kernels within {args.tol*100:.0f}% of baseline")


if __name__ == "__main__":
    main()
