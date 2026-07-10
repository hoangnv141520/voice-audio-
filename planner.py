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
    raw = [s.strip() for s in _SENT.findall(text) if s.strip()]
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


def plan_rule(text):
    """Câu có lời thoại (trong ngoặc kép) -> speaker 'speaker1'; còn lại narrator.
    Rule không đoán được TÊN nhân vật -> gán generic, user sửa ở GUI/voices.yaml."""
    out = []
    for i, s in enumerate(split_sentences(text)):
        speaker = "speaker1" if _QUOTE.search(s) else "narrator"
        out.append({"idx": i, "text": s, "speaker": speaker})
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
        "Tách đoạn văn sau thành từng câu. Với mỗi câu, xác định speaker "
        "(tên nhân vật đang nói, hoặc 'narrator' cho lời dẫn). Trả về JSON "
        'array [{"idx":int,"text":str,"speaker":str}], không giải thích.\n\n'
        + text
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    return data if isinstance(data, list) else data.get("segments", data.get("script", []))


def plan(text, mode="rule", **kw):
    return plan_rule(text) if mode == "rule" else plan_llm(text, **kw)


if __name__ == "__main__":
    t = 'Trời đã tối. "Cậu đi đâu đấy?" cô hỏi. Anh im lặng.'
    r = plan_rule(t)
    assert len(r) == 3, r
    assert r[1]["speaker"] == "speaker1", r  # câu có ngoặc kép
    assert r[0]["speaker"] == "narrator", r
    print(json.dumps(r, ensure_ascii=False, indent=2))
