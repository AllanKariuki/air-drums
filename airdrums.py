"""Air Drums — play a drum kit in the air with your hands.

How it works:
1. MediaPipe tracks both hands via webcam.
2. The "stick tip" of each hand = midpoint of index & middle fingertips.
3. Each tip's vertical velocity is tracked frame-to-frame.
4. A hit fires when a tip is inside a drum zone AND moving downward
   fast enough. Hit speed controls volume, like a real drum.
5. A short cooldown per drum prevents double-triggers.

Controls: q = quit, d = toggle debug overlay.
"""

import time

import cv2
import mediapipe as mp
import numpy as np
import pygame

from audio.drum_sounds import make_sounds

# ---------------------------------------------------------------- tuning ----
VEL_THRESHOLD = 1.1     # downward speed (normalized screen heights/sec) to count as a strike
COOLDOWN = 0.15         # seconds before the same drum can re-trigger
MIN_VOL, MAX_VOL = 0.35, 1.0
VOL_VEL_RANGE = (VEL_THRESHOLD, 4.0)  # velocity mapped onto volume range
FLASH_TIME = 0.12       # seconds a drum flashes after a hit

# ------------------------------------------------------------------ audio ---
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=256)
pygame.init()
pygame.mixer.set_num_channels(16)
SOUNDS = make_sounds()

# ------------------------------------------------------------------- kit ----
# Positions are fractions of frame width/height (frame is mirrored,
# so this looks like a kit facing you).
DRUMS = [
    # name      sound     cx     cy     radius  color (BGR)
    ("crash",  "crash",  0.13,  0.22,  0.105, (60, 200, 255)),
    ("hi-hat", "hihat",  0.22,  0.58,  0.095, (80, 220, 180)),
    ("snare",  "snare",  0.40,  0.76,  0.105, (90, 120, 255)),
    ("hi tom", "hi_tom", 0.54,  0.46,  0.095, (200, 140, 90)),
    ("lo tom", "lo_tom", 0.70,  0.74,  0.105, (170, 100, 220)),
    ("kick",   "kick",   0.88,  0.30,  0.115, (120, 120, 120)),
]


class DrumZone:
    def __init__(self, name, sound_key, cx, cy, radius, color):
        self.name = name
        self.sound_key = sound_key
        self.cx, self.cy, self.radius = cx, cy, radius
        self.color = color
        self.last_hit = 0.0

    def contains(self, x, y, aspect):
        # correct x distance for aspect ratio so zones behave as true circles
        dx = (x - self.cx) * aspect
        dy = y - self.cy
        return dx * dx + dy * dy <= self.radius * self.radius * aspect * aspect

    def try_hit(self, velocity_y, now):
        if velocity_y < VEL_THRESHOLD or now - self.last_hit < COOLDOWN:
            return False
        self.last_hit = now
        lo, hi = VOL_VEL_RANGE
        vol = np.interp(velocity_y, [lo, hi], [MIN_VOL, MAX_VOL])
        sound = SOUNDS[self.sound_key]
        sound.set_volume(float(vol))
        sound.play()
        return True


class HandTracker:
    """Tracks per-hand stick-tip position and vertical velocity."""

    def __init__(self):
        self.prev = {}  # hand label -> (x, y, t)

    def tip(self, landmarks):
        idx, mid = landmarks.landmark[8], landmarks.landmark[12]
        return (idx.x + mid.x) / 2, (idx.y + mid.y) / 2

    def velocity(self, label, x, y, now):
        vy = 0.0
        if label in self.prev:
            px, py, pt = self.prev[label]
            dt = now - pt
            if 0 < dt < 0.25:
                vy = (y - py) / dt  # + = moving down
        self.prev[label] = (x, y, now)
        return vy


def draw_kit(frame, drums, now, debug_info=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for d in drums:
        cx, cy, r = int(d.cx * w), int(d.cy * h), int(d.radius * h)
        flashing = now - d.last_hit < FLASH_TIME
        color = (255, 255, 255) if flashing else d.color
        thickness = -1 if flashing else 2
        cv2.circle(overlay, (cx, cy), r, color, thickness)
        cv2.circle(overlay, (cx, cy), r, color, 2)
        label_size = cv2.getTextSize(d.name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(overlay, d.name, (cx - label_size[0] // 2, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    if debug_info:
        for i, line in enumerate(debug_info):
            cv2.putText(frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


def main():
    drums = [DrumZone(*d) for d in DRUMS]
    tracker = HandTracker()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,          # fastest — latency matters for drumming
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    debug = False
    prev_time = time.time()

    print("Air Drums running. Strike downward into a zone. q=quit, d=debug")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        h, w = frame.shape[:2]
        aspect = w / h
        now = time.time()

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        debug_lines = []
        if results.multi_hand_landmarks:
            handedness = results.multi_handedness or []
            for i, lm in enumerate(results.multi_hand_landmarks):
                label = handedness[i].classification[0].label if i < len(handedness) else str(i)

                x, y = tracker.tip(lm)
                vy = tracker.velocity(label, x, y, now)

                # draw stick tip
                cv2.circle(frame, (int(x * w), int(y * h)), 10, (0, 255, 255), -1)
                if debug:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    debug_lines.append(f"{label}: vy={vy:+.2f}")

                for d in drums:
                    if d.contains(x, y, aspect):
                        d.try_hit(vy, now)

        if debug:
            fps = 1.0 / max(now - prev_time, 1e-6)
            debug_lines.append(f"FPS: {fps:.0f}")
        prev_time = now

        draw_kit(frame, drums, now, debug_lines if debug else None)
        cv2.imshow("Air Drums", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
