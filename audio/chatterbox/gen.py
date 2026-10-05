"""Chatterbox TTS generation on Colab T4.
Run via exec_detach AFTER audio/chatterbox/setup.py completes.
Reads /content/gen_config.json:
  {"text1": str, "text2": str (opt, cloned), "out_name": str (opt)}
Writes /content/tts_gen_status.json.
Outputs: /content/<out_name>.wav (default voice) and
/content/<out_name>_clone.wav (zero-shot clone from first 10s of wav 1).
"""
import json
import os
import subprocess
import sys
import time

STATUS = "/content/tts_gen_status.json"
CONFIG = "/content/gen_config.json"


def st(stage, **kw):
    d = {"stage": stage, "t": time.time(), **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return -1


try:
    cfg = json.load(open(CONFIG))
    text1 = cfg.get("text1")
    if not text1:
        raise RuntimeError("gen_config.json missing 'text1'")
    text2 = cfg.get("text2")
    out_name = cfg.get("out_name", "chatterbox_out")
    OUT1 = f"/content/{out_name}.wav"
    OUT2 = f"/content/{out_name}_clone.wav"
    REF_TRIM = "/content/clone_ref.wav"

    st("load", msg="ChatterboxTTS.from_pretrained(cuda)", vram_mb=vram_mb())
    from chatterbox.tts import ChatterboxTTS
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cuda")
    load_s = round(time.time() - t0, 1)
    sr = model.sr
    st("loaded", load_s=load_s, sr=sr, vram_mb=vram_mb())

    st("synth_default", vram_mb=vram_mb())
    t0 = time.time()
    wav1 = model.generate(text1)
    gen1_s = round(time.time() - t0, 1)
    import torchaudio
    torchaudio.save(OUT1, wav1, sr)
    dur1 = round(wav1.shape[-1] / sr, 2)
    st("synth1_done", gen_s=gen1_s, wav=OUT1, dur_s=dur1,
       size_b=os.path.getsize(OUT1), vram_mb=vram_mb())

    result = {"wav1": OUT1, "dur1_s": dur1, "gen1_s": gen1_s}
    if text2:
        st("trim_ref", msg="trimming first 10s for clone reference")
        torchaudio.save(REF_TRIM, wav1[..., : int(sr * 10)], sr)
        st("clone", msg="zero-shot voice clone", vram_mb=vram_mb())
        t0 = time.time()
        wav2 = model.generate(text2, audio_prompt_path=REF_TRIM)
        gen2_s = round(time.time() - t0, 1)
        torchaudio.save(OUT2, wav2, sr)
        dur2 = round(wav2.shape[-1] / sr, 2)
        result.update({"wav2": OUT2, "dur2_s": dur2, "gen2_s": gen2_s,
                       "size2_b": os.path.getsize(OUT2)})

    st("done", ready=True, size1_b=os.path.getsize(OUT1),
       vram_mb=vram_mb(), **result)
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
