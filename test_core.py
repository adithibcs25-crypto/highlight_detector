"""Sanity tests on synthetic audio (no video needed).  Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import highlight_detector as hd  # noqa: E402


def make_wav(path, burst_times, seconds=300, burst_len=6):
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.02, hd.SR * seconds)           # steady crowd noise
    for t in burst_times:                              # loud roars
        a, b = int(t * hd.SR), int((t + burst_len) * hd.SR)
        y[a:b] += rng.normal(0, 0.3, b - a)
    sf.write(path, np.clip(y, -1, 1), hd.SR)


def test_detects_bursts(tmp_path):
    bursts = [60, 150, 240]
    wav = tmp_path / "a.wav"
    make_wav(wav, bursts)
    times, _, _, score = hd.loudness_score(str(wav))
    peaks, _, _ = hd.detect_peaks(score, threshold_std=2.0, min_gap_s=30)
    found = times[peaks]
    assert len(found) == 3
    for b, f in zip(bursts, found):
        assert b - 2 <= f <= b + 8        # peak lands inside / right after the roar


def test_windows_merge_and_clamp():
    w = hd.build_windows([5, 10, 100], pre=12, post=8, duration=105)
    assert w == [[0.0, 18.0], [88.0, 105.0]]


def test_cap_strongest_keeps_order():
    score = np.array([0, 5, 0, 9, 0, 7, 0.0])
    peaks = np.array([1, 3, 5])
    assert list(hd.cap_strongest(peaks, score, 2)) == [3, 5]


def test_scene_filter():
    times = np.arange(0, 100, 0.1)
    peaks = np.array([100, 500])                       # t = 10 s and 50 s
    cuts = np.array([14.0])                            # cut 4 s after first peak
    assert list(hd.filter_by_scene_cut(peaks, times, cuts, 10)) == [100]
