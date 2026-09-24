#!/usr/bin/env python3
"""List tensor names/types/sizes for selected GGUF blocks (metadata only)."""
import sys

from gguf import GGUFReader

def main() -> int:
    if len(sys.argv) < 2:
        print("usage: dump_qwen_tensors.py <model.gguf> [block ...] [prefix ...]")
        return 2
    path = sys.argv[1]
    blocks = set(sys.argv[2:])
    reader = GGUFReader(path)
    total = 0
    for tensor in reader.tensors:
        name = tensor.name
        parts = name.split(".")
        keep = False
        if parts[0] == "blk" and len(parts) >= 2 and parts[1] in blocks:
            keep = True
        elif any(name.startswith(prefix) for prefix in blocks):
            keep = True
        if not keep:
            continue
        dims = "x".join(str(d) for d in reversed(tensor.shape))
        print(f"{name}  [{tensor.tensor_type.name}]  {dims}  ({tensor.n_bytes} bytes)")
        total += tensor.n_bytes
    print(f"-- selected bytes total: {total}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
