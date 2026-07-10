"""Load OmniVoice 1 lần, render script theo speaker (batch cùng giọng), giữ idx."""
import os
from collections import defaultdict

import torch
import yaml
from omnivoice import OmniVoice

MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
BATCH = int(os.environ.get("OMNIVOICE_BATCH", "4"))  # 8GB VRAM: 4 an toàn


def load_voices(path="voices.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("default", {"instruct": "neutral, calm"}), cfg.get("speakers", {})


def _voice_kwargs(voice):
    """voice dict -> kwargs cho generate. cloning (ref_audio/ref_text) hoặc instruct."""
    if voice.get("ref_audio"):
        kw = {"ref_audio": voice["ref_audio"]}
        if voice.get("ref_text"):
            kw["ref_text"] = voice["ref_text"]
        return kw
    return {"instruct": voice.get("instruct", "neutral, calm")}


def _load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA không khả dụng — bạn đang chạy nhầm Python (global, CPU-only) "
            "thay vì venv. Dùng gui.bat hoặc .venv\\Scripts\\python.exe app.py"
        )
    return OmniVoice.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.float16)


_MODEL = None


def get_model():
    """Load 1 lần, cache module-global (GUI gọi nhiều lần)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def render(script, voices_path="voices.yaml", model=None, voices=None):
    """script: [{idx,text,speaker}] -> list[np.ndarray] theo đúng thứ tự idx.
    voices: (default_voice, speakers) override; nếu None thì đọc voices_path."""
    default_voice, speakers = voices if voices else load_voices(voices_path)
    model = model or get_model()

    # Gom theo speaker để batch các câu cùng giọng.
    by_speaker = defaultdict(list)
    for seg in script:
        by_speaker[seg["speaker"]].append(seg)

    audio_by_idx = {}
    for speaker, segs in by_speaker.items():
        voice = speakers.get(speaker, default_voice)
        kw = _voice_kwargs(voice)
        # ref_audio/instruct broadcast theo list text -> nhân bản cho từng câu.
        for i in range(0, len(segs), BATCH):
            chunk = segs[i:i + BATCH]
            texts = [s["text"] for s in chunk]
            call = {k: [v] * len(texts) for k, v in kw.items()}
            try:
                audios = model.generate(text=texts, **call)
                for s, a in zip(chunk, audios):
                    audio_by_idx[s["idx"]] = a
            except ValueError:
                # ponytail: 1 câu ra audio rỗng làm OmniVoice nổ .max() cả lô ->
                # fallback render từng câu để cô lập câu hỏng, bỏ qua nó.
                for s, text in zip(chunk, texts):
                    try:
                        audio_by_idx[s["idx"]] = model.generate(
                            text=[text], **{k: [v[0]] for k, v in call.items()})[0]
                    except ValueError:
                        print(f"[skip] segment {s['idx']} render rỗng: {text[:40]!r}")

    return [audio_by_idx[i] for i in sorted(audio_by_idx)]
