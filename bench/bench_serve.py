#!/usr/bin/env python3
"""Throughput / latency benchmark for the MLz OpenAI server.

Stdlib only (urllib + threads) so it runs anywhere Python 3 does. Measures:
  - sequential baseline tok/s
  - concurrent aggregate tok/s + per-request latency percentiles
  - optional shared-prefix run to show the prefix-cache prefill win

Usage:
  python bench/bench_serve.py --url http://127.0.0.1:8080 \
      --requests 8 --concurrency 4 --max-tokens 64
  python bench/bench_serve.py --prefix-test   # measures prefix-cache benefit
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post_chat(url, model, messages, max_tokens, timeout=300):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = time.perf_counter() - t0
    usage = data.get("usage") or {}
    return {
        "latency": dt,
        "completion_tokens": usage.get("completion_tokens", 0),
        "content": (data.get("choices") or [{}])[0].get("message", {}).get("content", ""),
    }


def run_batch(url, model, prompts, max_tokens, concurrency):
    """Fire len(prompts) requests through a pool of `concurrency` workers."""
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(
            lambda m: post_chat(url, model, m, max_tokens), msgs))
    wall = time.perf_counter() - t0
    toks = sum(r["completion_tokens"] for r in results)
    lats = sorted(r["latency"] for r in results)
    return {
        "wall": wall,
        "tokens": toks,
        "tok_per_s": toks / wall if wall else 0,
        "p50": statistics.median(lats) if lats else 0,
        "p99": lats[int(len(lats) * 0.99)] if lats else 0,
        "n": len(results),
        "results": results,
    }


def fmt(label, r):
    return (f"{label:<22} {r['n']:>3} req  {r['tokens']:>5} tok  "
            f"{r['wall']:>6.1f}s  {r['tok_per_s']:>6.1f} tok/s  "
            f"p50={r['p50']*1000:>6.0f}ms p99={r['p99']*1000:>6.0f}ms")


def warmup(url, model):
    try:
        post_chat(url, model, [{"role": "user", "content": "hi"}], 4, timeout=600)
    except Exception as e:
        print(f"warmup failed (server not ready?): {e}")
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="mlz")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prefix-test", action="store_true",
                    help="measure prefix-cache benefit with a shared long system prompt")
    args = ap.parse_args()

    print(f"Benchmarking {args.url} (model={args.model})")
    warmup(args.url, args.model)

    if args.prefix_test:
        # A long shared prefix dominates prefill cost. With prefix caching the
        # second wave should prefill only the short suffix → lower latency.
        prefix = ("You are a meticulous assistant. " * 80).strip()
        prompts = [f"{prefix}\n\nQuestion {i}: name one color." for i in range(args.requests)]
        print("\n-- prefix-cache test (identical long prefix, varied suffix) --")
        cold = run_batch(args.url, args.model, prompts, args.max_tokens, args.concurrency)
        print(fmt("wave 1 (cold cache)", cold))
        warm = run_batch(args.url, args.model, prompts, args.max_tokens, args.concurrency)
        print(fmt("wave 2 (warm cache)", warm))
        if cold["p50"] and warm["p50"]:
            print(f"prefix-cache p50 latency: {cold['p50']*1000:.0f}ms -> "
                  f"{warm['p50']*1000:.0f}ms  ({cold['p50']/warm['p50']:.2f}x faster)")
        return

    topics = ["the ocean", "mountains", "rivers", "forests", "deserts",
              "glaciers", "volcanoes", "coral reefs", "the tundra", "rainforests"]
    prompts = [f"Tell me about {topics[i % len(topics)]}." for i in range(args.requests)]

    print("\n-- sequential baseline (concurrency=1) --")
    seq = run_batch(args.url, args.model, prompts, args.max_tokens, 1)
    print(fmt("sequential", seq))

    print(f"\n-- concurrent (concurrency={args.concurrency}) --")
    con = run_batch(args.url, args.model, prompts, args.max_tokens, args.concurrency)
    print(fmt("concurrent", con))

    if seq["tok_per_s"]:
        print(f"\nthroughput speedup: {con['tok_per_s'] / seq['tok_per_s']:.2f}x "
              f"({seq['tok_per_s']:.1f} -> {con['tok_per_s']:.1f} tok/s)")


if __name__ == "__main__":
    main()
