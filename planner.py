"""Tách đoạn văn dài -> script.json = [{idx, text, speaker}].

--planner rule : tách câu bằng regex; lời thoại trong "" 「」 “” -> speaker,
                 còn lại -> narrator. Không tốn API.
--planner llm  : gọi API (OpenAI-compatible) 1 lần, trả cùng JSON.
"""
import json
import os
import re

# Kết câu: . ! ? … 。 ！ ？ (giữ dấu). Bao gồm ký tự CJK.
_SENT = re.compile(r'[^.!?…。！？\n]+[.!?…。！？]+|\S[^.!?…。！？\n]*$', re.U)
# Lời thoại nằm trong cặp ngoặc kép các kiểu.
_QUOTE = re.compile(r'["“”「」『』](.+?)["“”「」『』]', re.U)


_QCHARS = '"“”「」『』'


def _unbalanced(s):
    # Ngoặc kép chưa đóng: tổng số ký tự quote lẻ.
    return sum(s.count(c) for c in _QCHARS) % 2 == 1


def split_sentences(text):
    # Tách theo DÒNG trước (thơ: mỗi dòng 1 segment, dù kết bằng phẩy hay không),
    # rồi trong mỗi dòng tách tiếp theo dấu câu.
    raw = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = _SENT.findall(line)
        raw.extend(p.strip() for p in parts if p.strip()) if parts else raw.append(line)
    # Gộp các mảnh bị cắt giữa lời thoại (?! nằm trong ngoặc kép).
    out, buf = [], ""
    for s in raw:
        buf = f"{buf} {s}".strip() if buf else s
        if not _unbalanced(buf):
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def detect_lang(text):
    """ISO code cho OmniVoice (vi/en/zh/ja...). None nếu không chắc -> model tự đoán."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # ponytail: langdetect non-deterministic -> seed cho ổn định
        code = detect(text)
    except Exception:
        return None
    # ponytail: langdetect kém với câu NGẮN (<~15 ký tự hay đoán bậy). Sửa tay cột
    # language trong GUI nếu sai; nâng lên fasttext-lid nếu cần độ chính xác cao.
    return code.split("-")[0]  # zh-cn/zh-tw -> zh (OmniVoice chỉ có 'zh')


def plan_rule(text):
    """Câu có lời thoại (trong ngoặc kép) -> speaker 'speaker1'; còn lại narrator.
    Rule không đoán được TÊN nhân vật -> gán generic, user sửa ở GUI/voices.yaml."""
    out = []
    for i, s in enumerate(split_sentences(text)):
        speaker = "speaker1" if _QUOTE.search(s) else "narrator"
        out.append({"idx": i, "text": s, "speaker": speaker, "language": detect_lang(s)})
    return out


def plan_llm(text, model=None, base_url=None, api_key=None):
    """Gọi 1 lần LLM để nhận diện ai nói câu nào. OpenAI-compatible."""
    from openai import OpenAI  # ponytail: lazy import, chỉ cần khi dùng llm

    client = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )
    model = model or os.environ.get("PLANNER_MODEL", "gpt-4o-mini")
    prompt = (
        "Tách đoạn văn sau thành từng câu. Với mỗi câu:\n"
        "- speaker: tên nhân vật đang nói, hoặc 'narrator' cho lời dẫn.\n"
        "- language: mã ISO (vi/en/zh/ja/ko/fr...).\n"
        "- text: câu gốc. Nếu câu có cảm xúc rõ, CHÈN tag cảm xúc phù hợp vào "
        "đầu/giữa câu (chỉ dùng các tag: [laughter] [sigh] [surprise-ah] "
        "[surprise-oh] [question-en] [dissatisfaction-hnn]). Không có cảm xúc "
        "rõ thì để nguyên, đừng chèn bừa.\n"
        'Trả về JSON array [{"idx":int,"speaker":str,"language":str,"text":str}], '
        "không giải thích.\n\n" + text
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    segs = data if isinstance(data, list) else data.get("segments", data.get("script", []))
    for s in segs:  # LLM untrusted -> bỏ tag ngoài whitelist OmniVoice
        s["text"] = _strip_bad_tags(s.get("text", ""))
    return segs


# Tag non-verbal OmniVoice chấp nhận (khớp _NONVERBAL_PATTERN trong model).
_VALID_TAGS = {"laughter", "sigh", "confirmation-en", "question-en", "question-ah",
               "question-oh", "question-ei", "question-yi", "surprise-ah",
               "surprise-oh", "surprise-wa", "surprise-yo", "dissatisfaction-hnn"}
_TAG = re.compile(r"\[([a-z-]+)\]")


def _strip_bad_tags(text):
    return _TAG.sub(lambda m: m.group(0) if m.group(1) in _VALID_TAGS else "", text).strip()


def plan(text, mode="rule", **kw):
    return plan_rule(text) if mode == "rule" else plan_llm(text, **kw)


if __name__ == "__main__":
    t = 'Trời đã tối. "Cậu đi đâu đấy?" cô hỏi. Anh im lặng.'
    r = plan_rule(t)
    assert len(r) == 3, r
    assert r[1]["speaker"] == "speaker1", r  # câu có ngoặc kép
    assert r[0]["speaker"] == "narrator", r
    # Thơ: mỗi dòng 1 segment, kể cả dòng kết bằng dấu phẩy (không được nuốt).
    poem = "Dòng một, chưa hết.\nDòng hai kết phẩy,\nDòng ba hết câu."
    sp = split_sentences(poem)
    assert len(sp) == 3, sp
    assert sp[1] == "Dòng hai kết phẩy,", sp
    # Detect ngôn ngữ per-câu (câu đủ dài mới đáng tin).
    assert detect_lang("Tôi yêu em rất nhiều lắm luôn.") == "vi"
    assert detect_lang("I really love you so much today.") == "en"
    # Sanitize tag LLM: giữ tag hợp lệ, bỏ tag bịa.
    assert _strip_bad_tags("[laughter] vui [evil] quá [sigh]") == "[laughter] vui  quá [sigh]"
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print("poem ok:", sp)
