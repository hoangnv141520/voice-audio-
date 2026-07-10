"""GUI Gradio: văn bản dài -> script (sửa tay) -> render -> nghe/tải out.wav.

Flow 2 bước để bạn kiểm soát trước khi tốn GPU:
  1. Tách câu (planner) -> bảng script sửa được (speaker + text).
  2. Khai giọng từng speaker (upload file mẫu = cloning, hoặc gõ instruct = design).
  3. Render -> 1 file wav.
"""
import tempfile

import gradio as gr

from merge import merge
from planner import plan
from tts import render


def do_plan(text, mode):
    script = plan(text, mode=mode)
    rows = [[s["idx"], s["speaker"], s["text"]] for s in script]
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
        ref = (str(row[2]) if row[2] else "").strip()
        if ref:
            speakers[name] = {"ref_audio": ref}
        elif instruct:
            speakers[name] = {"instruct": instruct}
    return {"instruct": default_instruct or "female, moderate pitch"}, speakers


def _fill_voices_from_files(files):
    """Upload -> điền bảng Voices: mỗi file = 1 dòng [tên(stem), "", path].
    Tên nhân vật = tên file (vd alice.wav -> speaker 'alice'); sửa lại cho khớp Script."""
    import os
    if not files:
        return []
    return [[os.path.splitext(os.path.basename(f.name))[0], "", f.name] for f in files]


def do_render(script_rows, voice_rows, default_instruct, gap):
    # ponytail: idx từ enumerate, không tin cột idx của Dataframe (leak header/placeholder).
    rows = script_rows.values.tolist() if hasattr(script_rows, "values") else script_rows
    script = [{"idx": i, "speaker": str(r[1]).strip(), "text": str(r[2]).strip()}
              for i, r in enumerate(rows)
              if len(r) >= 3 and str(r[2]).strip() and str(r[2]).strip().lower() != "text"]
    if not script:
        raise gr.Error("Script rỗng — bấm 'Tách câu' trước.")
    voices = _parse_voices(voice_rows, default_instruct)
    segments = render(script, voices=voices)
    out = tempfile.mktemp(suffix=".wav")
    path, dur = merge(segments, out, gap_s=gap)
    return path, f"Xong: {dur:.1f}s, {len(script)} câu."


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
                headers=["idx", "speaker", "text"],
                datatype=["number", "str", "str"],
                label="Script (sửa speaker/text tuỳ ý)", interactive=True, wrap=True)

    gr.Markdown("### Giọng — mỗi speaker: upload file mẫu (cloning) HOẶC gõ instruct (design)")
    voice_tbl = gr.Dataframe(
        headers=["speaker", "instruct", "ref_audio (đường dẫn)"],
        datatype=["str", "str", "str"],
        label="Voices", interactive=True, row_count=(2, "dynamic"))
    ref_upload = gr.File(label="Upload file mẫu (copy path vào cột ref_audio)",
                         file_types=["audio"], file_count="multiple")
    with gr.Row():
        default_instruct = gr.Textbox(value="female, moderate pitch",
                                      label="Giọng chung (default)")
        gap = gr.Slider(0, 1, value=0.25, step=0.05, label="Khoảng lặng giữa câu (s)")

    render_btn = gr.Button("2. Render", variant="primary")
    out_audio = gr.Audio(label="Kết quả", type="filepath")
    status = gr.Textbox(label="Trạng thái", interactive=False)

    plan_btn.click(do_plan, [text, planner_mode], [script_tbl, hint])
    ref_upload.upload(_fill_voices_from_files, ref_upload, voice_tbl)
    render_btn.click(do_render, [script_tbl, voice_tbl, default_instruct, gap],
                     [out_audio, status])


def _selfcheck():
    # Dataframe có thể trả header row / placeholder -> parse phải bỏ qua, không crash.
    rows = [["idx", "speaker", "text"], [0, "narrator", "Câu một."], [1, "alice", ""]]
    parsed = [{"idx": i, "speaker": str(r[1]).strip(), "text": str(r[2]).strip()}
              for i, r in enumerate(rows)
              if len(r) >= 3 and str(r[2]).strip() and str(r[2]).strip().lower() != "text"]
    assert parsed == [{"idx": 1, "speaker": "narrator", "text": "Câu một."}], parsed
    d, s = _parse_voices([["speaker", "instruct", "ref"], ["alice", "female", ""]], "male")
    assert s == {"alice": {"instruct": "female"}}, s
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
