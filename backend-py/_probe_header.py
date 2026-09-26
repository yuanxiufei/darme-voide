"""临时：打印真权重的张量清单（跑完即删）。"""
import json
import struct
import sys

for path in sys.argv[1:]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    meta = header.get("__metadata__") or {}
    print("=" * 20, path)
    print("meta:", {k: (v[:60] if isinstance(v, str) else v) for k, v in meta.items()})
    rows = [(k, tuple(v["shape"]), v["dtype"]) for k, v in header.items() if k != "__metadata__"]
    print("tensors:", len(rows))
    for key, shape, dtype in rows:
        print(f"  {key}  {shape}  {dtype}")
