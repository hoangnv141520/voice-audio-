# voice-audio — multi-speaker TTS bằng OmniVoice

Nhập 1 đoạn văn dài → agent tách câu + gán giọng từng nhân vật → OmniVoice
render từng câu → gộp thành 1 file audio duy nhất. Có GUI.

## Môi trường (đã chốt)
- OmniVoice cài ở `D:\voice` (venv riêng), project ở `D:\voice\voice-audio`.
- GPU: RTX 2070 8GB → `device_map="cuda:0"`, `dtype=float16`. VRAM trống ~7.4GB.
- Model trả `list[np.ndarray]` shape (T,) @ **24 kHz**.
- Gán giọng: **voice cloning** (file mẫu/nhân vật) + fallback **giọng chung**
  cho nhân vật không có mẫu. Planner: **rule** hoặc **llm** (chọn được).

## Kiến trúc
```
text ─▶ [1 planner] ─▶ script.json ─▶ [2 tts: model load 1 lần] ─▶ [3 merge] ─▶ out.wav
                                                    ▲
                                            voices.yaml (map speaker→giọng)
```

## File (tối thiểu)
```
voice-audio/
├── voices.yaml     # map nhân vật → ref_audio/ref_text hoặc instruct; có "default"
├── planner.py      # tách câu + gán speaker → script.json   (--planner rule|llm)
├── tts.py          # load model 1 lần, render theo speaker, giữ idx
├── merge.py        # concat np.ndarray + gap 0.25s → sf.write(out.wav, 24000)
├── app.py          # Gradio GUI (reuse dep sẵn có, không thêm mới)
├── run.py          # CLI: text → out.wav (ráp 3 bước)
└── refs/           # file audio mẫu do user cấp
```

## 3 bước
1. **planner.py** → `script.json = [{idx, text, speaker}]`
   - `rule`: split câu theo `. ! ? … 。 ！ ？`; lời thoại trong `"" 「」 “”` gán
     speaker; còn lại = `narrator`. Không tốn API.
   - `llm`: gọi API 1 lần, trả cùng JSON — nhận diện ai nói câu nào.
2. **tts.py** → load model 1 lần; với mỗi segment lấy giọng của `speaker` từ
   voices.yaml (không có → `default`); `model.generate(text, ref_audio, ref_text
   | instruct)`; lưu `audio` theo `idx`.
   - Kiểm tra lúc build: `generate` có nhận `text=[list]` để batch GPU thật không.
     Có → batch theo nhóm speaker (size 2–4, hợp 8GB). Không → loop tuần tự trên
     model đã load (vẫn nhanh, không reload). `# ponytail: seq loop, batch khi API hỗ trợ`.
3. **merge.py** → nối theo `idx`, chèn `np.zeros(int(0.25*24000))` giữa câu →
   1 file `out.wav`.

## GUI (Gradio)
- Ô nhập văn bản dài + upload nhiều file mẫu (đặt tên = tên nhân vật).
- Chọn planner (rule/llm). Nút Generate. Nghe/tải `out.wav`.
- Bảng script.json để sửa tay speaker trước khi render.

## Git (mỗi lần xong 1 chức năng)
```
cd D:\voice\voice-audio
git init && git add -A && git commit -m "..."
git branch -M main
git remote add origin https://github.com/hoangnv141520/voice-audio-.git
git push -u origin main
```

## Điểm cần verify (không đoán)
- [ ] Chữ ký `generate()`: có batch list không, tham số ref_audio/ref_text/instruct.
- [ ] Model chạy được fp16 trên RTX 2070 8GB (test 1 câu trước khi viết đủ).
- [ ] Tốc độ thực tế (RTF) để quyết batch size.

## Thứ tự làm
1. Cài OmniVoice ở D:\voice + test 1 câu (xác nhận GPU + signature generate).
2. voices.yaml + tts.py + merge.py → chạy CLI ra 1 file. → **git push**
3. planner.py (rule trước, llm sau). → **git push**
4. app.py Gradio GUI. → **git push**
