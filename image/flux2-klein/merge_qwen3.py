"""Merge diffusers-format Qwen3 text encoder shards into a single
qwen_3_4b.safetensors for ComfyUI CLIPLoader (type 'flux2').
Runs ON the Colab VM. Peak RAM ~8GB (two 4GB shards, sequential)."""
import json
import os

CACHE = "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B"
snap = os.path.join(CACHE, "snapshots", os.listdir(os.path.join(CACHE, "snapshots"))[0])
te = os.path.join(snap, "text_encoder")

from safetensors.torch import load_file, save_file

idx = json.load(open(os.path.join(te, "model.safetensors.index.json")))
shards = {}
for name, shard in idx["weight_map"].items():
    shards.setdefault(shard, []).append(name)

tensors = {}
for shard in sorted(shards):
    print(f"loading {shard}...", flush=True)
    sd = load_file(os.path.join(te, shard))
    for n in shards[shard]:
        tensors[n] = sd[n]
    del sd

out = "/content/ComfyUI/models/text_encoders/qwen_3_4b.safetensors"
os.makedirs(os.path.dirname(out), exist_ok=True)
save_file(tensors, out)
print(f"saved {len(tensors)} tensors -> {out} "
      f"({os.path.getsize(out) / 1e9:.2f} GB)", flush=True)
