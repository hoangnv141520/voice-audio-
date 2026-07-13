"""GUI Gradio: văn bản dài -> script (sửa tay) -> render -> nghe/tải out.wav.

Flow 2 bước để bạn kiểm soát trước khi tốn GPU:
  1. Tách câu (planner) -> bảng script sửa được (speaker + text).
  2. Khai giọng từng speaker (upload file mẫu = cloning, hoặc gõ instruct = design).
  3. Render -> 1 file wav.
"""
import json
import os
import tempfile

import gradio as gr

import tts
from merge import merge
from planner import plan
from tts import render

SESSION = "session.json"  # autosave state UI, giữ qua reload


def _tolist(df):
    return df.values.tolist() if hasattr(df, "values") else df


def _save(text, mode, script, voices, lex, dinstr, speed, gap):
    state = dict(text=text, mode=mode, script=_tolist(script), voices=_tolist(voices),
                 lex=_tolist(lex), dinstr=dinstr, speed=speed, gap=gap)
    with open(SESSION, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def _load():
    try:
        with open(SESSION, encoding="utf-8") as f:
            s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return (gr.skip(),) * 8
    return (s.get("text", ""), s.get("mode", "rule"), s.get("script"),
            s.get("voices"), s.get("lex"), s.get("dinstr", "female, moderate pitch"),
            s.get("speed", 1.0), s.get("gap", 0.25))


def do_plan(text, mode):
    script = plan(text, mode=mode)
    rows = [[s["idx"], s["speaker"], s.get("language") or "", s["text"]] for s in script]
    speakers = sorted({s["speaker"] for s in script})
    hint = "Speakers: " + ", ".join(speakers) + "\nKhai giọng cho từng speaker bên dưới."
    return rows, hint


def _parse_voices(voice_rows, default_instruct):
    """voice_rows: [[speaker, instruct, ref_audio_path], ...] -> (default, speakers)."""
    speakers = {}
    rows = voice_rows.values.tolist() if hasattr(voice_rows, "values") else voice_rows
    for row in rows:
        if len(row) < 3:
            continue
        name = (str(row[0]) if row[0] else "").strip()
        if name.lower() == "speaker":  # header row leak
            continue
        if not name:
            continue
        instruct = (str(row[1]) if row[1] else "").strip()
        ref = (str(row[2]) if row[2] else "").strip().strip('"').strip("'")
        if ref:
            speakers[name] = {"ref_audio": ref}
        elif instruct:
            speakers[name] = {"instruct": instruct}
    return {"instruct": default_instruct or "female, moderate pitch"}, speakers


def _parse_lexicon(lex_rows):
    """lex_rows: [[từ, cách đọc], ...] -> dict, bỏ dòng header/rỗng."""
    lex = {}
    rows = lex_rows.values.tolist() if hasattr(lex_rows, "values") else (lex_rows or [])
    for row in rows:
        if len(row) < 2:
            continue
        word = (str(row[0]) if row[0] else "").strip()
        say = (str(row[1]) if row[1] else "").strip()
        if word and say and word.lower() != "từ":
            lex[word] = say
    return lex


def do_render(script_rows, voice_rows, lex_rows, default_instruct, speed, gap):
    rows = script_rows.values.tolist() if hasattr(script_rows, "values") else script_rows
    script = [{"idx": i, "speaker": str(r[1]).strip(),
               "language": str(r[2]).strip() or None, "text": str(r[3]).strip()}
              for i, r in enumerate(rows)
              if len(r) >= 4 and str(r[3]).strip() and str(r[3]).strip().lower() != "text"]
    if not script:
        raise gr.Error("Script rỗng — bấm 'Tách câu' trước.")
    # Validate speaker match
    voice_rows_parsed = voice_rows.values.tolist() if hasattr(voice_rows, "values") else voice_rows
    vnames = set()
    for r in voice_rows_parsed:
        if len(r) >= 1 and str(r[0]).strip() and str(r[0]).strip().lower() != "speaker":
            vnames.add(str(r[0]).strip())
    unseen = {s["speaker"] for s in script} - vnames - {"narrator", "speaker1"}
    if unseen:
        gr.Warning(f"Speaker không có trong bảng Voices (dùng giọng chung): {', '.join(sorted(unseen))}")
    # Validate ref_audio tồn tại
    for r in voice_rows_parsed:
        if len(r) >= 3 and str(r[2]).strip():
            ref = str(r[2]).strip().strip('"').strip("'")
            if not os.path.exists(ref):
                raise gr.Error(f"File không tồn tại: {ref}")
    tts._LEXICON = _parse_lexicon(lex_rows)
    voices = _parse_voices(voice_rows, default_instruct)
    segments = render(script, voices=voices, speed=float(speed))
    out = tempfile.mktemp(suffix=".wav")
    path, dur = merge(segments, out, gap_s=gap)
    return path, f"Xong: {dur:.1f}s, {len(script)} câu."


# ponytail: import/export = _save + _load ra file user chọn. reuse format.
def _voice_preview(voice_rows, default_instruct, speed):
    from tts import get_model, _voice_kwargs
    rows = _tolist(voice_rows)
    for r in rows:
        if len(r) >= 2 and str(r[0]).strip() and str(r[0]).strip().lower() != "speaker":
            kw = _voice_kwargs({"instruct": str(r[1]) or default_instruct,
                               "ref_audio": str(r[2]).strip('"').strip("'") if len(r) > 2 else ""})
            a = get_model().generate(text=["Đây là giọng nói của speaker."],
                                     speed=[float(speed)], **kw)[0]
            p = tempfile.mktemp(suffix=".wav")
            import soundfile as sf
            from merge import SR
            sf.write(p, a, SR)
            return p
    raise gr.Error("Bảng Voices trống — thêm ít nhất 1 speaker.")


# ponytail: import text = đọc file đơn giản. .docx thêm dep sau.
def _import_text(file):
    try:
        with open(file.name, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file.name, encoding="utf-8-sig") as f:
            return f.read()


def _export_project(text, mode, script, voices, lex, dinstr, speed, gap, file):
    state = dict(text=text, mode=mode, script=_tolist(script), voices=_tolist(voices),
                 lex=_tolist(lex), dinstr=dinstr, speed=speed, gap=gap)
    path = file.name if hasattr(file, "name") else SESSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    return f"Xuất: {path}"


def _import_project(file):
    try:
        with open(file.name, encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        raise gr.Error("Không đọc được file project.")
    return (s.get("text", ""), s.get("mode", "rule"), s.get("script"),
            s.get("voices"), s.get("lex"), s.get("dinstr", "female, moderate pitch"),
            s.get("speed", 1.0), s.get("gap", 0.25))


with gr.Blocks(title="voice-audio") as demo:
    gr.Markdown("# voice-audio — multi-speaker TTS (OmniVoice)")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Đoạn văn", lines=10,
                              placeholder='Trời tối. "Cậu đi đâu?" cô hỏi.')
            planner_mode = gr.Radio(["rule", "llm"], value="rule", label="Planner")
            plan_btn = gr.Button("1. Tách câu", variant="secondary")
            hint = gr.Textbox(label="Speakers", interactive=False)
        with gr.Column():
            script_tbl = gr.Dataframe(
                headers=["idx", "speaker", "language", "text"],
                datatype=["number", "str", "str", "str"],
                label="Script (sửa speaker/language/text; language trống = tự đoán)",
                interactive=True, wrap=True)

    gr.Markdown("### Giọng — mỗi speaker: ref_audio (cloning) HOẶC instruct (design). "
                "Dán đường dẫn file mẫu vào cột ref_audio.")
    voice_tbl = gr.Dataframe(
        headers=["speaker", "instruct", "ref_audio (đường dẫn)"],
        datatype=["str", "str", "str"],
        label="Voices", interactive=True, row_count=(2, "dynamic"))
    gr.Markdown("### Từ điển phát âm — từ AI đọc sai -> gõ cách đọc đúng (vd: API -> ây pi ai)")
    lex_tbl = gr.Dataframe(
        headers=["từ", "cách đọc"], datatype=["str", "str"],
        label="Lexicon", interactive=True, row_count=(2, "dynamic"))

    with gr.Row():
        default_instruct = gr.Textbox(value="female, moderate pitch",
                                      label="Giọng chung (default)")
        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                          label="Tốc độ đọc (>1 nhanh, <1 chậm)")
        gap = gr.Slider(0, 1, value=0.25, step=0.05, label="Khoảng lặng giữa câu (s)")

    render_btn = gr.Button("2. Render", variant="primary")
    out_audio = gr.Audio(label="Kết quả", type="filepath")
    status = gr.Textbox(label="Trạng thái", interactive=False)

    plan_btn.click(do_plan, [text, planner_mode], [script_tbl, hint])
    render_btn.click(do_render,
                     [script_tbl, voice_tbl, lex_tbl, default_instruct, speed, gap],
                     [out_audio, status])

    # Voice preview: render 1 câu kiểm tra từ speaker + instruct/ref dòng đầu tiên của Voices
    preview_btn = gr.Button("3. Preview giọng", variant="secondary", size="sm")
    preview_out = gr.Audio(label="Preview", type="filepath")
    preview_btn.click(_voice_preview,
                      [voice_tbl, default_instruct, speed],
                      preview_out)

    # _fields dùng chung cho autosave + import/export (định nghĩa trước khi wiring).
    _fields = [text, planner_mode, script_tbl, voice_tbl, lex_tbl,
               default_instruct, speed, gap]

    # Import/Export project
    with gr.Row():
        import_btn = gr.UploadButton("📂 Nhập project", file_types=[".json"])
        export_btn = gr.DownloadButton("💾 Xuất project", variant="secondary", size="sm")
    import_btn.upload(_import_project, import_btn, _fields)
    export_btn.click(_export_project,
                     [text, planner_mode, script_tbl, voice_tbl, lex_tbl,
                      default_instruct, speed, gap, export_btn],
                     status)

    # Import text từ .txt
    txt_import = gr.UploadButton("📄 Import .txt", file_types=[".txt"], size="sm")
    txt_import.upload(_import_text, txt_import, text)

    # Autosave: mọi thay đổi -> ghi session.json; mở trang -> nạp lại.
    for c in _fields:
        c.change(_save, _fields, None)
    demo.load(_load, None, _fields)


def _selfcheck():
    import sys
    # ponytail: selfcheck không load GPU — chỉ check parse logic.
    # ponytail: Dataframe có thể trả header row / placeholder -> parse phải bỏ qua, không crash.
    rows = [["idx", "speaker", "language", "text"],
            [0, "narrator", "vi", "Câu một."], [1, "alice", "", ""]]
    parsed = [{"idx": i, "speaker": str(r[1]).strip(),
               "language": str(r[2]).strip() or None, "text": str(r[3]).strip()}
              for i, r in enumerate(rows)
              if len(r) >= 4 and str(r[3]).strip() and str(r[3]).strip().lower() != "text"]
    assert parsed == [{"idx": 1, "speaker": "narrator", "language": "vi",
                       "text": "Câu một."}], parsed
    d, s = _parse_voices([["speaker", "instruct", "ref"], ["alice", "female", ""]], "male")
    assert s == {"alice": {"instruct": "female"}}, s
    lx = _parse_lexicon([["từ", "cách đọc"], ["API", "ây pi ai"], ["", ""]])
    assert lx == {"API": "ây pi ai"}, lx
    # Autosave roundtrip: save rồi load phải khớp.
    global SESSION
    SESSION = tempfile.mktemp(suffix=".json")
    _save("xin chào", "llm", [[0, "narrator", "vi", "Câu."]], [["a", "female", ""]],
          [["API", "ây pi ai"]], "male", 1.5, 0.3)
    r = _load()
    assert r[0] == "xin chào" and r[1] == "llm" and r[6] == 1.5, r
    assert r[2] == [[0, "narrator", "vi", "Câu."]], r[2]
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        # ponytail: port=None -> Gradio tự tìm cổng trống, khỏi sập khi 7860 bị chiếm
        demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 0)) or None, inbrowser=True)
