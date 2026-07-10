"""CLI: đoạn văn dài -> 1 file audio. Ráp planner + tts + merge."""
import argparse
import json
import sys

from merge import merge
from planner import plan
from tts import render


def main():
    ap = argparse.ArgumentParser(description="Multi-speaker TTS bằng OmniVoice")
    ap.add_argument("--text", help="Đoạn văn (hoặc dùng --file)")
    ap.add_argument("--file", help="Đường dẫn file .txt")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--planner", choices=["rule", "llm"], default="rule")
    ap.add_argument("--voices", default="voices.yaml")
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--dump-script", help="Ghi script.json ra đây (tuỳ chọn)")
    args = ap.parse_args()

    text = args.text or (open(args.file, encoding="utf-8").read() if args.file else None)
    if not text:
        ap.error("cần --text hoặc --file")

    script = plan(text, mode=args.planner)
    if args.dump_script:
        with open(args.dump_script, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
    print(f"[planner] {len(script)} câu", file=sys.stderr)

    segments = render(script, voices_path=args.voices)
    path, dur = merge(segments, args.out, gap_s=args.gap)
    print(f"[done] {path} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
