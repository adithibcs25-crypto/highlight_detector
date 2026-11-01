# Automatic Football Highlight Detector

Feed in a football match video and get back a highlight reel, using nothing but
**audio spikes**. Goals, near-misses and big saves make the crowd roar, so
loudness is a surprisingly good highlight signal.

> Only use match footage you own or have permission to use.

## How it works

| Step | What happens | Tool |
|------|--------------|------|
| 1 | Extract the audio track to a mono WAV | MoviePy |
| 2 | Compute RMS loudness every 0.1 s, convert to dB, smooth | librosa |
| 3 | Subtract a rolling 60 s median baseline, so a constantly loud stadium doesn't trigger everything | numpy / scipy |
| 4 | Find peaks (height, prominence and minimum gap) | `scipy.signal.find_peaks` |
| 5 | *(optional)* Keep only peaks followed by a camera cut | PySceneDetect |
| 6 | Cut a window around each peak, merge overlaps, build a reel | MoviePy |

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Minimum install: `pip install librosa moviepy numpy scipy`.
MoviePy ships its own ffmpeg through `imageio-ffmpeg`, so a separate ffmpeg install is normally not needed.

## Usage

```bash
# 1. Tune first: prints timestamps and saves a chart, cuts nothing
python highlight_detector.py match.mp4 --dry-run --plot

# 2. Full run
python highlight_detector.py match.mp4

# Stricter: top 15 moments only, with scene-change filtering
python highlight_detector.py match.mp4 --threshold 2.5 --max-clips 15 --scene-filter
```

### Output (`highlights/` by default)

```
highlights/
├── clip_01_00-12-34.mp4     individual clips (named by start time)
├── clip_02_00-41-07.mp4
├── highlights_reel.mp4      all clips joined
├── highlights.json          peak times, scores, windows, settings used
└── tuning.png               only with --plot
```

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--threshold` | 2.0 | Peak height in std-devs above the mean score. Higher gives fewer clips |
| `--prominence` | 0.5 | Minimum peak prominence (std-devs). Filters weak bumps |
| `--min-gap` | 30 | Minimum seconds between two peaks |
| `--smooth` | 1.5 | Loudness smoothing window (s) |
| `--baseline` | 60 | Rolling baseline window (s) |
| `--pre` / `--post` | 12 / 8 | Seconds kept before / after each peak |
| `--max-clips` | 0 | Keep only the N strongest peaks (0 = all) |
| `--scene-filter` | off | Keep peaks followed by a camera cut (needs `scenedetect`) |
| `--scene-window` | 10 | Seconds after a peak in which a cut must happen |
| `--no-reel` | off | Skip the combined reel |
| `--dry-run` | off | Print timestamps only |
| `--plot` | off | Save `tuning.png` (needs matplotlib) |

## Tuning to reduce false positives

Run `--dry-run --plot` and look at the red threshold line on the score chart.

| Problem | Try |
|---------|-----|
| Too many clips (whistles, chants, replays, ad jingles) | Raise `--threshold` (2.5 to 3), raise `--prominence`, use `--max-clips`, add `--scene-filter` |
| Missing real goals | Lower `--threshold` (about 1.5) or lower `--prominence` |
| Clip starts too late | Increase `--pre`; the buildup often comes before the roar peaks |
| Several clips for one moment | Increase `--min-gap` |
| Quiet stadium or low-volume recording | Lower `--baseline` (30 s) so the baseline adapts faster |
| Commentary shouting triggers clips | Raise `--smooth` (2 to 3 s) and `--threshold` |

## Tests

```bash
pip install pytest soundfile
pytest -q
```

The tests use synthetic audio (steady noise with injected "roars"), so no video is needed.

## Limitations

- Audio only: it can't tell a goal from a loud chant, a fight or a stadium announcement.
- Broadcast mixes with heavy commentary or music beds reduce accuracy; raw stadium audio works best.
- Processing a 90-minute match takes a few minutes. Most time goes to audio extraction and video encoding.
- Peaks are kept in time order; `--max-clips` picks the *loudest*, not the *best*.

## Ideas for extending

- Add a scene-detection or motion score and combine it with the audio score.
- Detect the commentator's pitch rise (`librosa.pyin`) as a second signal.
- Train a small classifier on labelled clips (goal / foul / other).
- Add a whistle detector (about 3 to 4 kHz) to remove false positives.

## Project layout

```
football-highlight-detector/
├── highlight_detector.py
├── requirements.txt
├── README.md
├── .gitignore
└── tests/test_core.py
```
