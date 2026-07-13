"""Load OmniVoice 1 lần, render script theo speaker (batch cùng giọng), giữ idx."""
import os
import re
from collections import defaultdict

import torch
import yaml
from omnivoice import OmniVoice

MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
BATCH = int(os.environ.get("OMNIVOICE_BATCH", "4"))  # 8GB VRAM: 4 an toàn


def load_voices(path="voices.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("default", {"instruct": "female, moderate pitch"}), cfg.get("speakers", {})


_LEXICON = None


def load_lexicon(path="lexicon.yaml"):
    """{từ: cách đọc}. Thay từ AI đọc sai bằng cách viết đọc đúng, trước khi generate.
    OmniVoice không có phoneme control cho tiếng Việt -> text-replace là đường duy nhất."""
    global _LEXICON
    if _LEXICON is None:
        try:
            with open(path, encoding="utf-8") as f:
                _LEXICON = yaml.safe_load(f) or {}
        except FileNotFoundError:
            _LEXICON = {}
    return _LEXICON


def apply_lexicon(text, lex=None):
    lex = load_lexicon() if lex is None else lex
    for word, say in lex.items():
        # word-boundary lỏng: hợp cả từ CJK/Việt (\b không tin cậy với unicode).
        text = re.sub(rf"(?<![\w]){re.escape(word)}(?![\w])", str(say), text)
    return text


def _voice_kwargs(voice):
    """voice dict -> kwargs cho generate. cloning (ref_audio/ref_text) hoặc instruct."""
    if voice.get("ref_audio"):
        kw = {"ref_audio": voice["ref_audio"]}
        if voice.get("ref_text"):
            kw["ref_text"] = voice["ref_text"]
        return kw
    return {"instruct": voice.get("instruct", "female, moderate pitch")}


def _load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA không khả dụng — bạn đang chạy nhầm Python (global, CPU-only) "
            "thay vì venv. Dùng gui.bat hoặc .venv\\Scripts\\python.exe app.py"
        )
    return OmniVoice.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.float16)


_MODEL = None

# Cache audio per-câu: key=(text,speaker,lang,speed) -> np.ndarray.
# ponytail: dict phẳng cap 512 câu (FIFO), đủ cho tinh chỉnh lặp; LRU khi RAM thành vấn đề.
_AUDIO_CACHE = {}
_CACHE_CAP = 512


def _cache_get(key):
    return _AUDIO_CACHE.get(key)


def _cache_put(key, audio):
    if len(_AUDIO_CACHE) >= _CACHE_CAP:
        _AUDIO_CACHE.pop(next(iter(_AUDIO_CACHE)))
    _AUDIO_CACHE[key] = audio


def get_model():
    """Load 1 lần, cache module-global (GUI gọi nhiều lần)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def render(script, voices_path="voices.yaml", model=None, voices=None, speed=1.0):
    """script: [{idx,text,speaker}] -> list[np.ndarray] theo đúng thứ tự idx.
    voices: (default_voice, speakers) override; nếu None thì đọc voices_path.
    speed: tốc độ đọc chung (>1 nhanh, <1 chậm); segment có 'speed' riêng thì ưu tiên."""
    default_voice, speakers = voices if voices else load_voices(voices_path)
    model = model or get_model()

    # Gom theo speaker để batch các câu cùng giọng.
    by_speaker = defaultdict(list)
    for seg in script:
        by_speaker[seg["speaker"]].append(seg)

    lex = load_lexicon()
    audio_by_idx = {}
    for speaker, segs in by_speaker.items():
        voice = speakers.get(speaker, default_voice)
        kw = _voice_kwargs(voice)
        # ref_audio/instruct broadcast theo list text -> nhân bản cho từng câu.
        for i in range(0, len(segs), BATCH):
            chunk = segs[i:i + BATCH]
            texts = [apply_lexicon(s["text"], lex) for s in chunk]
            spds = [s.get("speed") or speed for s in chunk]
            keys = [(t, speaker, s.get("language"), sp)
                    for t, s, sp in zip(texts, chunk, spds)]
            # Cache hit -> lấy sẵn; miss -> gom lại render.
            todo = [(j, s, t) for j, (s, t, k) in enumerate(zip(chunk, texts, keys))
                    if _cache_get(k) is None]
            for s, k in zip(chunk, keys):
                cached = _cache_get(k)
                if cached is not None:
                    audio_by_idx[s["idx"]] = cached
            if not todo:
                continue
            js, tsegs, ttexts = zip(*todo)
            call = {k2: [v] * len(ttexts) for k2, v in kw.items()}
            tlangs = [s.get("language") for s in tsegs]
            if any(tlangs):
                call["language"] = tlangs  # per-câu; None -> model tự đoán câu đó
            call["speed"] = [spds[j] for j in js]  # per-câu > global
            try:
                audios = model.generate(text=list(ttexts), **call)
                for s, t, k, a in zip(tsegs, ttexts, [keys[j] for j in js], audios):
                    audio_by_idx[s["idx"]] = a
                    _cache_put(k, a)
            except ValueError:
                # ponytail: 1 câu ra audio rỗng làm OmniVoice nổ .max() cả lô ->
                # fallback render từng câu để cô lập câu hỏng, bỏ qua nó.
                for n, (s, text) in enumerate(zip(tsegs, ttexts)):
                    try:
                        a = model.generate(
                            text=[text], **{k2: [v[n]] for k2, v in call.items()})[0]
                        audio_by_idx[s["idx"]] = a
                        _cache_put(keys[js[n]], a)
                    except ValueError:
                        print(f"[skip] segment {s['idx']} render rỗng: {text[:40]!r}")

    return [audio_by_idx[i] for i in sorted(audio_by_idx)]


if __name__ == "__main__":
    # lexicon: thay đúng từ, không dính substring.
    lex = {"API": "ây pi ai", "AWS": "ây đắp liu ét"}
    assert apply_lexicon("Gọi API rồi lên AWS.", lex) == "Gọi ây pi ai rồi lên ây đắp liu ét."
    assert apply_lexicon("APIx không đổi", lex) == "APIx không đổi"  # substring không dính
    # cache: put/get + cap FIFO.
    _AUDIO_CACHE.clear()
    _cache_put(("a",), 1); assert _cache_get(("a",)) == 1
    assert _cache_get(("z",)) is None
    print("tts selfcheck ok")
