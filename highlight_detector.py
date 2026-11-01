#!/usr/bin/env python3
"""
Automatic football highlight detector (audio-spike based).

Pipeline:
  1. Extract audio from the match video            (MoviePy)
  2. Compute RMS loudness                          (librosa)
  3. Score loudness against a rolling baseline     (numpy / scipy)
  4. Detect peaks                                  (scipy.signal.find_peaks)
  5. Optional: keep peaks near a scene change      (PySceneDetect)
  6. Cut clips around each peak + build a reel     (MoviePy)

Only use video you have permission to use.
"""
import argparse
import json
import tempfile
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

# MoviePy 2.x and 1.x differ in imports and method names
try:
    from moviepy import VideoFileClip, concatenate_videoclips  # 2.x
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips  # 1.x
    MOVIEPY_V2 = False

SR = 22050            # audio sample rate
HOP = SR // 10        # 0.1 s per RMS frame -> 10 frames per second
FRAME = HOP * 2       # 0.2 s analysis window
FPS_FEAT = SR / HOP   # feature frames per second (10)


def subclip(clip, start, end):
    return clip.subclipped(start, end) if MOVIEPY_V2 else clip.subclip(start, end)


# --------------------------------------------------------------------------- #
# Audio extraction
# --------------------------------------------------------------------------- #
def extract_audio(video_path: Path, wav_path: str) -> float:
    """Write a mono WAV from the video's audio track. Returns duration (s)."""
    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            raise SystemExit("This video has no audio track.")
        video.audio.write_audiofile(
            wav_path, fps=SR, nbytes=2, ffmpeg_params=["-ac", "1"], logger=None
        )
        return float(video.duration)


# --------------------------------------------------------------------------- #
# Loudness + score
# --------------------------------------------------------------------------- #
def loudness_score(wav_path: str, smooth_s: float = 1.5, baseline_s: float = 60.0):
    """
    Returns (times, smoothed_db, baseline_db, score).

    score = dB the smoothed loudness sits above a rolling-median baseline.
    The baseline stops constant crowd noise or loud commentary from swamping
    genuine spikes.
    """
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=1.0)

    smooth = uniform_filter1d(rms_db, size=max(1, int(smooth_s * FPS_FEAT)))
    baseline = median_filter(
        smooth, size=max(3, int(baseline_s * FPS_FEAT)), mode="nearest"
    )
    score = smooth - baseline
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    return times, smooth, baseline, score


# --------------------------------------------------------------------------- #
# Peak detection
# --------------------------------------------------------------------------- #
def detect_peaks(score, threshold_std=2.0, min_gap_s=30.0, prominence_std=0.5):
    mu, sigma = float(np.mean(score)), float(np.std(score))
    peaks, _ = find_peaks(
        score,
        height=mu + threshold_std * sigma,
        distance=max(1, int(min_gap_s * FPS_FEAT)),
        prominence=prominence_std * sigma,
    )
    return peaks, mu, sigma


def cap_strongest(peaks, score, max_clips):
    """Keep only the N strongest peaks (chronological order preserved)."""
    if max_clips and len(peaks) > max_clips:
        peaks = peaks[np.argsort(score[peaks])[::-1][:max_clips]]
    return np.sort(peaks)


# --------------------------------------------------------------------------- #
# Optional: scene-change filter (PySceneDetect)
# --------------------------------------------------------------------------- #
def scene_cut_times(video_path: Path, threshold: float = 27.0):
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        raise SystemExit("Scene filter needs: pip install 'scenedetect[opencv]'")
    scenes = detect(str(video_path), ContentDetector(threshold=threshold))
    return np.array([s[0].get_seconds() for s in scenes[1:]])


def filter_by_scene_cut(peaks, times, cuts, window_s):
    """Keep peaks that have a camera cut within `window_s` seconds after them."""
    if len(cuts) == 0:
        return peaks
    keep = [i for i in peaks if np.any((cuts > times[i]) & (cuts < times[i] + window_s))]
    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------- #
# Windows + cutting
# --------------------------------------------------------------------------- #
def build_windows(peak_times, pre, post, duration):
    """Turn peak times into [start, end] windows, merging overlaps."""
    windows = []
    for t in sorted(peak_times):
        s, e = max(0.0, t - pre), min(duration, t + post)
        if windows and s <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], e)
        else:
            windows.append([s, e])
    return windows


def fmt(t):
    t = int(t)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def cut_clips(video_path: Path, windows, out_dir: Path, make_reel: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_kw = dict(codec="libx264", audio_codec="aac", logger=None)
    with VideoFileClip(str(video_path)) as video:
        clips = []
        for i, (s, e) in enumerate(windows, 1):
            clip = subclip(video, s, e)
            path = out_dir / f"clip_{i:02d}_{fmt(s).replace(':', '-')}.mp4"
            print(f"  writing {path.name}  ({fmt(s)} -> {fmt(e)})")
            clip.write_videofile(str(path), **write_kw)
            clips.append(clip)
        if make_reel and clips:
            print("  building highlight reel ...")
            reel = concatenate_videoclips(clips)
            reel.write_videofile(str(out_dir / "highlights_reel.mp4"), **write_kw)
            reel.close()


# --------------------------------------------------------------------------- #
# Tuning aid
# --------------------------------------------------------------------------- #
def plot_debug(times, smooth, baseline, score, peaks, mu, sigma, thr, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot. pip install matplotlib)")
        return
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(times / 60, smooth, lw=0.6, label="loudness (dB)")
    ax[0].plot(times / 60, baseline, lw=1, label="rolling baseline")
    ax[0].legend(loc="upper right")
    ax[1].plot(times / 60, score, lw=0.6, label="score (dB above baseline)")
    ax[1].axhline(mu + thr * sigma, color="r", ls="--", label=f"threshold ({thr} std)")
    ax[1].plot(times[peaks] / 60, score[peaks], "rx", label="peaks")
    ax[1].set_xlabel("minutes")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Saved tuning plot: {out_png}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Football highlight detector (audio spikes)")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--out-dir", type=Path, default=Path("highlights"))
    p.add_argument("--threshold", type=float, default=2.0,
                   help="peak height in std-devs above mean score (higher = fewer clips)")
    p.add_argument("--prominence", type=float, default=0.5,
                   help="min peak prominence in std-devs (filters weak bumps)")
    p.add_argument("--min-gap", type=float, default=30.0, help="min seconds between peaks")
    p.add_argument("--smooth", type=float, default=1.5, help="loudness smoothing window (s)")
    p.add_argument("--baseline", type=float, default=60.0,
                   help="rolling-median baseline window (s)")
    p.add_argument("--pre", type=float, default=12.0, help="seconds kept before a peak")
    p.add_argument("--post", type=float, default=8.0, help="seconds kept after a peak")
    p.add_argument("--max-clips", type=int, default=0,
                   help="keep only the N strongest peaks (0 = all)")
    p.add_argument("--scene-filter", action="store_true",
                   help="keep only peaks followed by a camera cut (needs scenedetect)")
    p.add_argument("--scene-window", type=float, default=10.0,
                   help="seconds after a peak in which a cut must occur")
    p.add_argument("--no-reel", action="store_true", help="don't build the combined reel")
    p.add_argument("--dry-run", action="store_true", help="print timestamps only; no cutting")
    p.add_argument("--plot", action="store_true", help="save a tuning plot (needs matplotlib)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"File not found: {args.video}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "audio.wav")
        print("Extracting audio ...")
        duration = extract_audio(args.video, wav)
        print("Computing loudness ...")
        times, smooth, baseline, score = loudness_score(wav, args.smooth, args.baseline)

    peaks, mu, sigma = detect_peaks(score, args.threshold, args.min_gap, args.prominence)
    print(f"Audio peaks found: {len(peaks)}")

    if args.scene_filter:
        print("Detecting scene changes ...")
        cuts = scene_cut_times(args.video)
        peaks = filter_by_scene_cut(peaks, times, cuts, args.scene_window)
        print(f"Peaks after scene filter: {len(peaks)}")

    peaks = cap_strongest(peaks, score, args.max_clips)
    windows = build_windows(times[peaks], args.pre, args.post, duration)

    print(f"\n{len(peaks)} peaks -> {len(windows)} clips")
    for i in peaks:
        print(f"  peak at {fmt(times[i])}  (+{score[i]:.1f} dB over baseline)")

    report = {
        "peaks": [{"time_s": round(float(times[i]), 1), "score_db": round(float(score[i]), 2)}
                  for i in peaks],
        "windows": [[round(s, 1), round(e, 1)] for s, e in windows],
        "settings": {k: v for k, v in vars(args).items() if k not in ("video", "out_dir")},
    }
    (args.out_dir / "highlights.json").write_text(json.dumps(report, indent=2, default=str))

    if args.plot:
        plot_debug(times, smooth, baseline, score, peaks, mu, sigma,
                   args.threshold, str(args.out_dir / "tuning.png"))

    if args.dry_run or not windows:
        print("\nDry run / nothing to cut. Adjust --threshold and re-run.")
        return

    print("\nCutting clips ...")
    cut_clips(args.video, windows, args.out_dir, make_reel=not args.no_reel)
    print(f"\nDone. Output in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
