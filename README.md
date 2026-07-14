# Air Drums 🥁

Play a full drum kit in the air. Your webcam tracks your hands; striking downward into on-screen drum zones plays synthesized drum sounds. Hit harder (faster) = louder.

## Setup

Requires **Python 3.9–3.12** (mediapipe doesn't support newer versions yet — see `.python-version`).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python airdrums.py
```

Requires a webcam. Works best with decent lighting and your hands clearly visible.

## How to play

Six zones are drawn over the mirrored camera feed: crash, hi-hat, snare, hi tom, lo tom, kick. Make a fast downward striking motion into a zone — like hitting a real drum. Hovering or moving slowly through a zone does nothing; only a strike triggers a hit.

Keys: `q` quits, `d` toggles debug overlay (hand skeleton, per-hand velocity, FPS).

## How it works

**Hand tracking** — MediaPipe Hands detects up to two hands per frame (`model_complexity=0` for lowest latency).

**Stick tip** — each hand's "drumstick tip" is the midpoint of the index and middle fingertips (landmarks 8 and 12).

**Velocity strike detection** — the tip's vertical position is compared across frames to get downward velocity in screen-heights/second. A hit fires only when the tip is inside a zone *and* moving down faster than `VEL_THRESHOLD`. Velocity maps linearly to volume, so soft taps are quiet and hard hits are loud. A per-drum cooldown (`COOLDOWN`) stops one strike from double-triggering.

**Sound** — all drums are synthesized in `drum_sounds.py` with numpy (no sample files): the kick is a 150→45 Hz sine pitch-sweep, the snare a short tone plus high-passed noise, cymbals are shaped noise bursts, toms are pitched sine sweeps. pygame's mixer plays them with a 256-sample buffer for low latency.

## Tuning

All knobs are at the top of `airdrums.py`:

| Constant | Effect |
|---|---|
| `VEL_THRESHOLD` | Lower = easier to trigger, higher = requires harder strikes |
| `COOLDOWN` | Minimum time between hits on the same drum |
| `MIN_VOL` / `MAX_VOL` | Volume range mapped from strike speed |
| `DRUMS` | Zone positions, sizes, and colors (fractions of frame size) |

If hits feel laggy, keep the camera resolution at 1280×720 or lower and close other camera apps. If drums trigger accidentally, raise `VEL_THRESHOLD` to ~1.5.

## Project files

- `airdrums.py` — main app: tracking, zones, hit detection, rendering
- `drum_sounds.py` — numpy drum synthesis → pygame sounds
- `trackhands.py`, `thumbs_up.py`, `web_cam.py` — earlier experiments
