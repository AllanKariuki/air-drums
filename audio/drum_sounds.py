"""Synthesized drum sounds — no sample files needed.

Each function returns a numpy float array in [-1, 1] at SAMPLE_RATE.
make_sounds() converts them into ready-to-play pygame Sound objects.
"""

import numpy as np
import pygame

SAMPLE_RATE = 44100


def _t(duration):
    """Time axis for a given duration in seconds."""
    return np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)


def _decay(t, rate):
    """Exponential decay envelope."""
    return np.exp(-rate * t)


def _noise(n):
    return np.random.uniform(-1, 1, n)


def kick():
    """Punchy kick: sine pitch-sweep 150Hz -> 45Hz + click transient."""
    t = _t(0.35)
    freq = 150 * np.exp(-8 * t) + 45
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    body = np.sin(phase) * _decay(t, 9)
    click = _noise(len(t)) * _decay(t, 120) * 0.4
    return body + click


def snare():
    """Snare: short 190Hz tone body + bright noise rattle."""
    t = _t(0.25)
    body = np.sin(2 * np.pi * 190 * t) * _decay(t, 25) * 0.5
    rattle = _noise(len(t)) * _decay(t, 18) * 0.8
    # crude high-pass on the noise (differentiate) for snap
    rattle = np.append(np.diff(rattle), 0) * 4
    return body + rattle


def hihat():
    """Closed hi-hat: very short high-passed noise burst."""
    t = _t(0.12)
    n = _noise(len(t))
    n = np.append(np.diff(n), 0) * 5  # high-pass
    return n * _decay(t, 55)


def crash():
    """Crash cymbal: long shimmering noise with slow decay."""
    t = _t(1.6)
    n = _noise(len(t))
    n = np.append(np.diff(n), 0) * 4
    shimmer = 1 + 0.3 * np.sin(2 * np.pi * 6 * t)
    return n * _decay(t, 3.2) * shimmer


def tom(base_freq):
    """Tom: sine sweep down from base_freq with medium decay."""
    t = _t(0.4)
    freq = base_freq * (1 + 0.5 * np.exp(-10 * t))
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    body = np.sin(phase) * _decay(t, 8)
    attack = _noise(len(t)) * _decay(t, 90) * 0.25
    return body + attack


def _to_pygame_sound(wave, gain=0.85):
    """Normalize, convert to 16-bit stereo, wrap in pygame Sound."""
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * gain
    pcm = (wave * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_sounds():
    """Build all drum sounds. Call AFTER pygame.mixer is initialized."""
    return {
        "kick":   _to_pygame_sound(kick(), gain=1.0),
        "snare":  _to_pygame_sound(snare()),
        "hihat":  _to_pygame_sound(hihat(), gain=0.7),
        "crash":  _to_pygame_sound(crash()),
        "hi_tom": _to_pygame_sound(tom(180)),
        "lo_tom": _to_pygame_sound(tom(110)),
    }
