# voice-audio-

Multi-speaker TTS bằng [OmniVoice](https://github.com/k2-fsa/OmniVoice): nhập
đoạn văn dài → tách câu + gán giọng từng nhân vật → render → gộp thành 1 file
audio. Có GUI.

## Cài
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv "torch>=2.4,<2.7" torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv -r requirements.txt
```
GPU: RTX 2070 8GB, fp16, VRAM peak ~2GB. Model tự tải từ HuggingFace lần đầu.

## Dùng (CLI)
```bash
python run.py --text 'Trời đã tối. "Cậu đi đâu?" cô hỏi.' --out out.wav
python run.py --file truyen.txt --planner rule --out out.wav
```

## Giọng — `voices.yaml`
- **Cloning**: `ref_audio` (+`ref_text`) → clone từ file mẫu trong `refs/`.
- **Design**: `instruct` (whitelist: male/female, low/high pitch, british accent,
  whisper... xem file). Nhân vật không khai → dùng `default`.

## Planner
- `rule` (mặc định): tách câu bằng regex, lời thoại trong ngoặc kép → speaker.
- `llm`: gọi API OpenAI-compatible nhận diện speaker (cần `OPENAI_API_KEY`).

## File
`planner.py` tách câu · `tts.py` render (batch theo speaker) · `merge.py` gộp ·
`run.py` CLI · `app.py` GUI.
