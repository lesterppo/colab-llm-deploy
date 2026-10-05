# Audio generation on Colab

One-command text-to-speech and lyrics-to-song on free Colab GPUs (T4, 15GB
VRAM). Each model dir is self-contained: `setup.py` (VM-side, stage-gated)
+ `gen.py` (VM-side, reads `/content/gen_config.json`).

## One command

```bash
# From the repo root:
python3 deploy_media.py generate --kind audio --model chatterbox \
    --text1 "The lighthouse stands above the waves." \
    --text2 "Shine on through the night." \
    --out ./outputs

python3 deploy_media.py generate --kind audio --model yue2 \
    --style "cinematic folk, warm male vocal, 90 bpm" \
    --lyrics "[Verse]
Golden light on the water
[Chorus]
Shine on" \
    --max-seconds 40 \
    --out ./outputs

python3 deploy_media.py generate --kind audio --model minimax-music3 \
    --caption "cinematic folk, warm male vocal" \
    --lyrics "[verse] Golden light
[chorus] Shine on" \
    --duration 30 \
    --out ./outputs

# Speech-to-text:
python3 deploy_media.py generate --kind audio --model whisper \
    --audio-file ./speech.wav --out ./outputs
```

`deploy_media.py` does everything: session → upload → detached setup →
poll → detached gen → poll → download → stop session.

## Models

### chatterbox — Chatterbox TTS 0.5B (MIT)
- Text-to-speech + zero-shot voice cloning. `pip install chatterbox-tts`,
  no ComfyUI. gen_config: `text1` (default voice) + optional `text2`
  (cloned from first 10s of wav 1).
- Proven (2026-10-05, T4): 5.1s + 2.9s wavs, peak VRAM 3.6GB.
- Pitfalls: pip can pull torchvision newer than Colab's torch 2.6.0
  (`torchvision::nms` missing) → pin `torchvision==0.21.0`; uninstall
  torchao (peft ImportError). No HF token needed (ungated).

### yue2 — YuE2-3B lyrics-to-song (CC BY-NC 4.0, test use only)
- ComfyUI + `pytraveler/YuE2-ComfyUI` + tiktoken. Weights `m-a-p/YuE2-3B` +
  `m-a-p/YuE2-Vae` via `hf` CLI. gen_config: `style`, `lyrics` (with
  `[Verse]`/`[Chorus]` tags), `seed`, `max_seconds`, `out_name`.
- Workflow: `YuE2Options(cot=full, max_seconds, download=off,
  attention_backend=sdpa)` → `YuE2GenerateSong` → `SaveAudio`.
- Proven (2026-10-05, T4): 40s 48kHz song in 250s, peak VRAM 6.4GB.
  T4 has no hardware BF16 — the acoustic stage is slower than on modern
  cards, but 90-min gen cap is plenty.

### minimax-music3 — MiniMax Music 3, INT8 repack (verify license)
- 8B Qwen3-init LLM + 0.6B local LLM + 2.4B DiT + Flow-VAE; the INT8
  `convrot` repack squeezes 22GB → 11.9GB (from
  `lesterppo/minimax-music3-colab`). ComfyUI, gen_config: `caption`,
  `lyrics`, `seed`, `duration`, `out_name`.
- Proven (2026-10-05, T4): 30s song in 301s, peak VRAM 11.9GB.
- Pitfall: upload the config as a file — multi-line lyrics break when
  passed through `exec --code` (shell newline handling).

### whisper — Whisper large-v3-turbo STT (verify license for non-test use)
- 809M params, fp16, ~1.9GB VRAM — the lightest deployment in the series.
  gen_config: `audio_path` (uploaded via `--audio-file`), `out_name`.
- Proven (2026-10-05, T4): 18/18 words correct transcribing Chatterbox
  output, 3.7s for 5.1s of audio.
