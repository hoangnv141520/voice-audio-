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
    for row in voice_rows:
        name = (row[0] or "").strip()
        if not name:
            continue
        instruct = (row[1] or "").strip()
        ref = (row[2] or "").strip()
        if ref:
            speakers[name] = {"ref_audio": ref}
        elif instruct:
            speakers[name] = {"instruct": instruct}
    return {"instruct": default_instruct or "female, moderate pitch"}, speakers


def do_render(script_rows, voice_rows, default_instruct, gap):
    script = [{"idx": int(r[0]), "speaker": str(r[1]), "text": str(r[2])}
              for r in script_rows if str(r[2]).strip()]
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
    ref_upload.upload(lambda fs: "\n".join(f.name for f in fs) if fs else "",
                      ref_upload, hint)
    render_btn.click(do_render, [script_tbl, voice_tbl, default_instruct, gap],
                     [out_audio, status])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
