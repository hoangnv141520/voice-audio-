"""Nối các segment np.ndarray @24kHz thành 1 file, chèn khoảng lặng giữa câu."""
import numpy as np
import soundfile as sf

SR = 24000


def merge(segments, out_path, gap_s=0.25):
    """segments: list[np.ndarray] đã sắp đúng thứ tự."""
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    parts = []
    for a in segments:
        parts.append(np.asarray(a, dtype=np.float32))
        parts.append(gap)
    final = np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)
    sf.write(out_path, final, SR)
    return out_path, len(final) / SR


if __name__ == "__main__":
    a = np.ones(SR, np.float32) * 0.1
    p, dur = merge([a, a], "_merge_test.wav", gap_s=0.5)
    assert abs(dur - 2.5) < 0.01, dur  # 1s + 0.5 gap + 1s
    print("ok", p, dur)
