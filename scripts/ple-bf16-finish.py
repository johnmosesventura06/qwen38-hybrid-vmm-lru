# SPDX-License-Identifier: Apache-2.0
"""finish gates v2: scale-aware cross-check + NaN census + atomic deploy."""
import json
import os
import random

import numpy as np
import torch

LANE_HOME = os.environ.get("LANE_HOME", os.path.expanduser("~"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(LANE_HOME, "models"))
TMP = os.environ.get("PLE_BF16_TMP", os.path.join(MODELS_DIR, "ple-bf16-assembling.bin"))
OUT_DIR = os.environ.get("PLE_BF16_OUT", os.path.join(MODELS_DIR, "ple-disk8-bf16"))
TOTAL = 102_400_491_520
SCALE = 0.00019931793212890625
RPS, W = 2_500_012, 160
assert os.path.getsize(TMP) == TOTAL

fp8_path = os.path.join(
    os.environ.get("PLE_FP8_DIR", os.path.join(MODELS_DIR, "ple-disk8")),
    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight.bin",
)
random.seed(3)
errs = []
for n in random.sample(range(128), 8):
    for row in random.sample(range(RPS), 15):
        with open(TMP, "rb") as o:
            o.seek((n * RPS + row) * W * 2)
            bf = torch.from_numpy(np.frombuffer(o.read(W * 2), dtype=np.uint16).copy()).view(torch.bfloat16).float()
        with open(fp8_path, "rb") as f:
            f.seek((n * RPS + row) * W)
            fp = torch.from_numpy(np.frombuffer(f.read(W), dtype=np.uint8).copy()).view(torch.float8_e4m3fn).float() * SCALE
        errs.append(((bf - fp).abs() / bf.abs().clamp_min(fp.abs().max() * 0.02 + 1e-6)).median().item())
mx = max(errs)
print(f"scale-aware cross-check: max median-rel-err {mx:.5f} over {len(errs)} rows")
assert mx < 0.25, "cross-check FAILED — not deploying"

def census(path):
    n = 0
    i = 0
    CH = 1 << 26
    with open(path, "rb") as f:
        while i < TOTAL // 2:
            cnt = min(CH, TOTAL // 2 - i)
            f.seek(i * 2)
            u = np.frombuffer(f.read(cnt * 2), dtype=np.uint16)
            e = (u >> 7) & 0xFF
            n += int(np.count_nonzero((e == 0xFF) & ((u & 0x7F) != 0)))
            i += cnt
    return n

nan = census(TMP)
print(f"NaN elements: {nan}")
assert nan == 0, "table contains NaNs — not deploying"

os.makedirs(OUT_DIR, exist_ok=True)
target = os.path.join(OUT_DIR, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight.bin")
assert not os.path.exists(target), "target exists, refusing"
os.rename(TMP, target)
with open(target.replace(".bin", ".done.json"), "w") as dj:
    json.dump({"shape": [320_001_536, 160], "dtype": "torch.bfloat16",
               "source": "Qwen/Qwen3.8-Flash-Next", "assembled_by": "shard-map v2 + scale-aware fp8 cross-check"}, dj)
print("TABLE DEPLOYED", target)
