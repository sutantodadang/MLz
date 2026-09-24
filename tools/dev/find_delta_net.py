import os
root = "deps"
hits = []
for dirpath, dirs, files in os.walk(root):
    d = dirpath.replace(os.sep, "/").lower()
    if any(s in d for s in [".git", "build", "zig-cache", "zig-out", "node_modules"]):
        continue
    for f in files:
        if f.endswith((".cpp", ".hpp", ".c", ".h")):
            p = os.path.join(dirpath, f).replace(os.sep, "/")
            try:
                t = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if "build_delta_net" in t:
                hits.append(p)
print("\n".join(hits))
