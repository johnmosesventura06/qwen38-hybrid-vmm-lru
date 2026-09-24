# SPDX-License-Identifier: Apache-2.0
"""Assemble bf16 PLE table from official shards. Verified or aborts; never partial-writes the target dir."""
import json
import os
import re
import struct
import sys

LANE_HOME = os.environ.get("LANE_HOME", os.path.expanduser("~"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(LANE_HOME, "models"))
SRC = os.environ.get("PLE_BF16_SRC", os.path.join(MODELS_DIR, "ple-bf16-src"))
OUT_DIR = os.environ.get("PLE_BF16_OUT", os.path.join(MODELS_DIR, "ple-disk8-bf16"))
TMP = os.environ.get("PLE_BF16_TMP", os.path.join(MODELS_DIR, "ple-bf16-assembling.bin"))
TENSOR = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{n}.weight"
ROWS_PER_SHARD = 2_500_012
WIDTH = 160
TOTAL_BYTES = 320_001_536 * WIDTH * 2

def header_of(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    data_start = 8 + n
    return hdr, data_start

# 1) build shard->(file, offset, nbytes, shape, dtype) map from ALL source headers
map_shards = {}
files = sorted(f for f in os.listdir(SRC) if f.endswith(".safetensors"))
for fn in files:
    path = os.path.join(SRC, fn)
    hdr, dstart = header_of(path)
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        m = re.fullmatch(r"model\.language_model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight", name)
        if not m:
            continue
        n = int(m.group(1))
        s, e = meta["data_offsets"]
        assert meta["dtype"] == "BF16" and meta["shape"] == [ROWS_PER_SHARD, WIDTH], (name, meta)
        assert n not in map_shards
        map_shards[n] = (path, dstart + s, e - s)
assert len(map_shards) == 128, f"only {len(map_shards)} shards found"
assert all(map_shards[n][2] == ROWS_PER_SHARD * WIDTH * 2 for n in map_shards)
print("128 shards mapped, shapes+dtypes verified")

# 2) assemble in numeric shard order
with open(TMP, "wb") as out:
    out.truncate(TOTAL_BYTES)
    with open(TMP, "r+b") as outm:
        for n in range(128):
            path, off, nb = map_shards[n]
            with open(path, "rb") as src:
                src.seek(off)
                rem = nb
                while rem:
                    buf = src.read(min(rem, 1 << 26))
                    outm.write(buf)
                    rem -= len(buf)
            outm.flush()
print("assembly written, size:", os.path.getsize(TMP), "expected:", TOTAL_BYTES)
assert os.path.getsize(TMP) == TOTAL_BYTES

# 3) spot-verify 10 random rows byte-for-byte vs source
import random
random.seed()
for n in random.sample(range(128), 10):
    path, off, nb = map_shards[n]
    row = random.randrange(ROWS_PER_SHARD)
    rb = WIDTH * 2
    with open(path, "rb") as f:
        f.seek(off + row * rb)
        src_bytes = f.read(rb)
    with open(TMP, "rb") as o:
        o.seek((n * ROWS_PER_SHARD + row) * rb)
        out_bytes = o.read(rb)
    assert src_bytes == out_bytes, f"row mismatch shard {n} row {row}"
print("10/10 random rows byte-identical")

# 4) cross-check vs live fp8 table: bf16 row -> e4m3*scale ~ fp8 row
import numpy as np
import torch
fp8_path = os.path.join(
    os.environ.get("PLE_FP8_DIR", os.path.join(MODELS_DIR, "ple-disk8")),
    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight.bin",
)
if os.path.exists(fp8_path):
    rel_errs = []
    for n in random.sample(range(128), 4):
        rows = random.sample(range(ROWS_PER_SHARD), 25)
        for row in rows:
            with open(TMP, "rb") as o:
                o.seek((n * ROWS_PER_SHARD + row) * WIDTH * 2)
                bf = np.frombuffer(o.read(WIDTH * 2), dtype=np.uint16)
            with open(fp8_path, "rb") as f8:
                f8.seek((n * ROWS_PER_SHARD + row) * WIDTH)
                fp = np.frombuffer(f8.read(WIDTH), dtype=np.uint8)
            bf_t = torch.from_numpy(bf.copy()).view(torch.bfloat16).float()
            fp_t = torch.from_numpy(fp.copy()).view(torch.float8_e4m3fn).float()
            denom = bf_t.abs().clamp_min(1.0)
            rel_errs.append(((bf_t - fp_t).abs() / denom).max().item())
    worst = max(rel_errs)
    print(f"fp8 cross-check worst row-rel-err over 100 rows: {worst:.4f} (expect <0.13 e4m3 eps)")
    assert worst < 0.2, "cross-check FAILED — row order or content mismatch; NOT deploying table"
else:
    print("WARNING: fp8 table absent, cross-check skipped")

# 5) NaN/Inf census (padding rows may be zero; NaN is unusual but not auto-fatal)
import numpy as np
u16 = np.memmap(TMP, dtype=np.uint16, mode="r")
exp = (u16 >> 7) & 0xFF
nans = int(np.count_nonzero((exp == 0xFF) & ((u16 & 0x7F) != 0)))
infs = int(np.count_nonzero((exp == 0xFF) & ((u16 & 0x7F) == 0)))
del u16
print(f"NaN rows-elements: {nans}, Inf: {infs}")

# 6) publish atomically + manifest
os.makedirs(OUT_DIR, exist_ok=True)
target = os.path.join(OUT_DIR, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight.bin")
assert not os.path.exists(target), "target exists — refusing; move it away first"
os.rename(TMP, target)
with open(target.replace(".bin", ".done.json"), "w") as dj:
    json.dump({"shape": [320_001_536, 160], "dtype": "torch.bfloat16",
               "source": "Qwen/Qwen3.8-Flash-Next", "assembled_utc": "local",
               "assembled_by": "shard-map v1"}, dj)
print("TABLE DEPLOYED:", target)
