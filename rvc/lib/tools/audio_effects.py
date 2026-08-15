import numpy as np
from scipy import signal as sp_signal
from pedalboard import (
    Chorus,
    Compressor,
    Delay,
    Distortion,
    Gain,
    Limiter,
    Pedalboard,
    Phaser,
    PitchShift,
    Reverb,
)

EFFECTS = [
    "None",
    "White Noise",
    "Pink Noise",
    "Brown Noise",
    "Rain",
    "Vinyl",
    "FNAF Radio",
    "FM Radio",
    "Walkie-Talkie",
    "Telephone",
    "Megaphone",
    "Muffled",
    "Underwater",
    "Old TV",
    "Lowpass",
    "Highpass",
    "Notch",
    "Bass Boost",
    "Treble Boost",
    "Tremolo",
    "Vibrato",
    "Phaser",
    "Flanger",
    "Auto-Wah",
    "Overdrive",
    "Fuzz",
    "Waveshaper",
    "Robot",
    "Alien",
    "Ghost",
    "Reverse",
    "Chipmunk",
    "Darth Vader",
    "Demon",
    "Echo Canyon",
    "Normalize",
    "Mono",
    "Stereo Widener",
    "Air",
]

_KEYS = {
    "White Noise": "white_noise",
    "Pink Noise": "pink_noise",
    "Brown Noise": "brown_noise",
    "Rain": "rain",
    "Vinyl": "vinyl",
    "FNAF Radio": "fnaf_radio",
    "FM Radio": "fm_radio",
    "Walkie-Talkie": "walkie_talkie",
    "Telephone": "telephone",
    "Megaphone": "megaphone",
    "Muffled": "muffled",
    "Underwater": "underwater",
    "Old TV": "old_tv",
    "Lowpass": "lowpass",
    "Highpass": "highpass",
    "Notch": "notch",
    "Bass Boost": "bass_boost",
    "Treble Boost": "treble_boost",
    "Tremolo": "tremolo",
    "Vibrato": "vibrato",
    "Phaser": "phaser",
    "Flanger": "flanger",
    "Auto-Wah": "auto_wah",
    "Overdrive": "overdrive",
    "Fuzz": "fuzz",
    "Waveshaper": "waveshaper",
    "Robot": "robot",
    "Alien": "alien",
    "Ghost": "ghost",
    "Reverse": "reverse",
    "Chipmunk": "chipmunk",
    "Darth Vader": "darth_vader",
    "Demon": "demon",
    "Echo Canyon": "echo_canyon",
    "Normalize": "normalize",
    "Mono": "mono",
    "Stereo Widener": "stereo_widener",
    "Air": "air",
}


def apply_audio_effect(audio, sample_rate, effect_name, intensity=0.5):
    if not effect_name or effect_name == "None":
        return audio
    key = _KEYS.get(effect_name)
    if key is None:
        return audio
    audio = np.asarray(audio, dtype=np.float32)
    was_mono = audio.ndim == 1
    if was_mono:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    audio = _run(key, audio, sample_rate, float(intensity))
    if was_mono:
        return audio[0]
    return audio.T


def _run(key, audio, sr, intensity):
    rng = np.random.default_rng(0)
    n = audio.shape[1]
    t = np.arange(n) / sr
    if key == "white_noise":
        noise = rng.standard_normal((audio.shape[0], n)).astype(np.float32)
        return _mix(audio, noise, 0.1 + 0.6 * intensity)
    if key == "pink_noise":
        return _mix(audio, _pink_noise(audio.shape[0], n, rng), 0.1 + 0.6 * intensity)
    if key == "brown_noise":
        noise = np.cumsum(
            rng.standard_normal((audio.shape[0], n)), axis=1
        ).astype(np.float32)
        noise /= np.max(np.abs(noise)) + 1e-8
        return _mix(audio, noise, 0.1 + 0.6 * intensity)
    if key == "rain":
        rain = _bandpass(_pink_noise(audio.shape[0], n, rng), sr, 600, 4000)
        am = 0.6 + 0.4 * np.sin(2 * np.pi * 1.8 * t + 0.5)
        rain = rain * am[np.newaxis, :].astype(np.float32)
        return _mix(audio, rain, 0.15 + 0.55 * intensity)
    if key == "vinyl":
        crackle = _crackle(audio.shape[0], n, sr, rng, density=0.05)
        hum = np.sin(2 * np.pi * 50 * t) * 0.004
        out = _lowpass(audio, sr, 9000)
        return _mix(out, crackle + hum[np.newaxis, :].astype(np.float32), 0.1 + 0.4 * intensity)
    if key == "fnaf_radio":
        radio = _bandpass(audio, sr, 400, 2800)
        crackle = _crackle(audio.shape[0], n, sr, rng, density=0.12, amp=0.25)
        hiss = rng.standard_normal((audio.shape[0], n)).astype(np.float32) * 0.01
        am = 0.85 + 0.15 * np.sin(2 * np.pi * 7 * t + 1.0)
        radio = radio * am[np.newaxis, :].astype(np.float32)
        return radio + (crackle + hiss) * (0.5 + intensity)
    if key == "fm_radio":
        out = _bandpass(audio, sr, 250, 4200)
        hiss = rng.standard_normal((audio.shape[0], n)).astype(np.float32) * 0.008
        am = 0.9 + 0.1 * np.sin(2 * np.pi * 3.2 * t)
        return out * am[np.newaxis, :].astype(np.float32) + hiss * (0.3 + intensity)
    if key == "walkie_talkie":
        out = _bandpass(audio, sr, 500, 2600)
        crackle = _crackle(audio.shape[0], n, sr, rng, density=0.06, amp=0.4)
        out = Pedalboard([Compressor(threshold_db=-18, ratio=4)])(out, sr)
        return out + crackle * (0.4 + intensity)
    if key == "telephone":
        out = _bandpass(audio, sr, 300, 3400)
        hiss = rng.standard_normal((audio.shape[0], n)).astype(np.float32) * 0.006
        return out + hiss * (0.3 + intensity)
    if key == "megaphone":
        out = _bandpass(audio, sr, 400, 3000)
        out = np.tanh(out * (1.5 + 1.5 * intensity)).astype(np.float32)
        return Gain(gain_db=3.0)(out, sr)
    if key == "muffled":
        return _lowpass(audio, sr, 900 - 500 * intensity)
    if key == "underwater":
        out = _lowpass(audio, sr, 700 - 300 * intensity)
        am = 0.75 + 0.25 * np.sin(2 * np.pi * 1.6 * t + 2.0)
        out = out * am[np.newaxis, :].astype(np.float32)
        return Reverb(room_size=0.7, damping=0.6, wet_level=0.35, dry_level=0.65)(out, sr)
    if key == "old_tv":
        out = _bandpass(audio, sr, 200, 3500)
        hum = np.sin(2 * np.pi * 50 * t + 0.7) * 0.012
        hiss = rng.standard_normal((audio.shape[0], n)).astype(np.float32) * 0.006
        return (out + (hum[np.newaxis, :] + hiss) * (0.4 + intensity)).astype(
            np.float32
        )
    if key == "lowpass":
        return _lowpass(audio, sr, 3500 - 2000 * intensity)
    if key == "highpass":
        return _highpass(audio, sr, 200 + 300 * intensity)
    if key == "notch":
        return _notch(audio, sr, 1000, 300 + 500 * intensity)
    if key == "bass_boost":
        return _low_shelf(audio, sr, 200, 4 + 8 * intensity)
    if key == "treble_boost":
        return _high_shelf(audio, sr, 6000, 4 + 8 * intensity)
    if key == "tremolo":
        depth = 0.3 + 0.7 * intensity
        mod = 1 - depth + depth * np.abs(np.sin(2 * np.pi * 5.5 * t))
        return (audio * mod[np.newaxis, :].astype(np.float32)).astype(np.float32)
    if key == "vibrato":
        return _vibrato(audio, sr, rate_hz=5.5, depth_st=0.1 + 0.9 * intensity)
    if key == "phaser":
        return Phaser(rate_hz=0.7, depth=0.3 + 0.6 * intensity, feedback=0.4, mix=1.0)(audio, sr)
    if key == "flanger":
        return Chorus(
            rate_hz=0.2,
            depth=0.6 + 0.4 * intensity,
            centre_delay_ms=0.8,
            feedback=0.6,
            mix=0.6 + 0.4 * intensity,
        )(audio, sr)
    if key == "auto_wah":
        out = _bandpass(audio, sr, 400, 2500)
        depth = 0.4 + 0.6 * intensity
        mod = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.5 * t + 1.2))
        return (out * mod[np.newaxis, :].astype(np.float32)).astype(np.float32)
    if key == "overdrive":
        drive = 1 + 6 * intensity
        return np.tanh(audio * drive).astype(np.float32)
    if key == "fuzz":
        drive = 1 + 18 * intensity
        return _lowpass(np.tanh(audio * drive).astype(np.float32), sr, 3500)
    if key == "waveshaper":
        g = 1 + 3 * intensity
        x = np.clip(audio * g, -1, 1)
        return (1.5 * x - 0.5 * x**3).astype(np.float32)
    if key == "robot":
        carrier = np.abs(np.sin(2 * np.pi * 55 * t))
        mix = 0.4 + 0.6 * intensity
        return (audio * (1 - mix) + audio * carrier[np.newaxis, :].astype(np.float32) * mix).astype(np.float32)
    if key == "alien":
        out = PitchShift(semitones=3.0)(audio, sr)
        out = _vibrato(out, sr, rate_hz=6.0, depth_st=0.2 + 0.5 * intensity)
        return Phaser(rate_hz=1.2, depth=0.5, feedback=0.3, mix=0.7)(out, sr)
    if key == "ghost":
        rev = Reverb(room_size=0.85, damping=0.4, wet_level=0.5, dry_level=0.5)
        return rev(audio[:, ::-1], sr)[:, ::-1]
    if key == "reverse":
        return audio[:, ::-1]
    if key == "chipmunk":
        return PitchShift(semitones=7.0)(audio, sr)
    if key == "darth_vader":
        out = PitchShift(semitones=-7.0)(audio, sr)
        return np.tanh(out * (1 + 2 * intensity)).astype(np.float32)
    if key == "demon":
        out = PitchShift(semitones=-4.0)(audio, sr)
        out = np.tanh(out * (1 + 6 * intensity)).astype(np.float32)
        return _lowpass(out, sr, 3000)
    if key == "echo_canyon":
        board = Pedalboard(
            [
                Delay(delay_seconds=0.35, feedback=0.55, mix=0.45),
                Reverb(room_size=0.9, damping=0.3, wet_level=0.4, dry_level=0.6),
            ]
        )
        return board(audio, sr)
    if key == "normalize":
        peak = np.max(np.abs(audio)) + 1e-8
        target = 10 ** (-1.0 / 20)
        return (audio / peak * target).astype(np.float32)
    if key == "mono":
        mono = np.mean(audio, axis=0, keepdims=True)
        return np.broadcast_to(mono, audio.shape).copy()
    if key == "stereo_widener":
        if audio.shape[0] < 2:
            return audio
        mid = np.mean(audio, axis=0, keepdims=True)
        side = (audio[0] - audio[1]) / 2
        side = np.tanh(side * (0.5 + 1.5 * intensity))
        left = mid[0] + side
        right = mid[0] - side
        return np.stack([left, right], axis=0).astype(np.float32)
    if key == "air":
        out = _highpass(audio, sr, 8000)
        hiss = rng.standard_normal((audio.shape[0], n)).astype(np.float32) * 0.004 * intensity
        return (out + hiss).astype(np.float32)
    return audio


def _mix(signal, noise, level):
    level = np.clip(float(level), 0.0, 1.0)
    noise = np.broadcast_to(noise, signal.shape)
    return (signal * np.float32(1 - level) + noise * np.float32(level)).astype(
        np.float32
    )


def _pink_noise(channels, n, rng):
    out = np.zeros((channels, n), dtype=np.float32)
    for c in range(channels):
        white = rng.standard_normal(n)
        f = np.fft.rfftfreq(n)
        f[0] = 1.0
        spec = np.fft.rfft(white) / np.sqrt(f)
        spec *= n / (np.abs(spec).max() + 1e-8)
        out[c] = np.fft.irfft(spec, n)
    peak = np.max(np.abs(out)) + 1e-8
    return out / peak


def _crackle(channels, n, sr, rng, density=0.05, amp=0.3):
    impulse_count = min(int(n * density), 200000)
    channel_impulses = rng.uniform(0, n, size=impulse_count).astype(int)
    noise = np.zeros((channels, n), dtype=np.float32)
    decay = np.exp(-np.arange(0, int(0.004 * sr)) * 12.0)
    for pos in channel_impulses:
        end = min(pos + len(decay), n)
        k = end - pos
        ch = rng.integers(0, channels)
        noise[ch, pos:end] += decay[:k] * rng.uniform(-amp, amp)
    return noise.astype(np.float32)


def _lowpass(audio, sr, freq):
    freq = min(freq, sr * 0.49)
    sos = sp_signal.butter(4, freq, btype="lowpass", fs=sr, output="sos")
    return sp_signal.sosfilt(sos, audio).astype(np.float32)


def _highpass(audio, sr, freq):
    freq = min(freq, sr * 0.49)
    sos = sp_signal.butter(4, freq, btype="highpass", fs=sr, output="sos")
    return sp_signal.sosfilt(sos, audio).astype(np.float32)


def _low_shelf(audio, sr, freq, gain_db):
    _b, _a = _rbj_shelf(freq, gain_db, sr, low=True)
    return sp_signal.lfilter(_b, _a, audio).astype(np.float32)


def _high_shelf(audio, sr, freq, gain_db):
    _b, _a = _rbj_shelf(freq, gain_db, sr, low=False)
    return sp_signal.lfilter(_b, _a, audio).astype(np.float32)


def _rbj_shelf(freq, gain_db, sr, low=True):
    import math

    freq = min(freq, sr * 0.49)
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / 2 * math.sqrt(2)
    cos_w0 = math.cos(w0)
    sqrt_a = math.sqrt(A)
    if low:
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_a * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_a * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_a * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_a * alpha
    else:
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_a * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_a * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_a * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_a * alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _bandpass(audio, sr, low, high):
    high = min(high, sr * 0.49)
    low = min(low, high - 1)
    sos = sp_signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sp_signal.sosfilt(sos, audio).astype(np.float32)


def _notch(audio, sr, freq, width):
    low = max(freq - width / 2, 1.0)
    high = min(freq + width / 2, sr * 0.49)
    sos = sp_signal.butter(
        2, [low, high], btype="bandstop", fs=sr, output="sos"
    )
    return sp_signal.sosfilt(sos, audio).astype(np.float32)


def _vibrato(audio, sr, rate_hz=5.5, depth_st=0.5):
    frame = int(0.05 * sr)
    hop = frame // 2
    n = audio.shape[1]
    window = np.hanning(frame).astype(np.float32)
    out = np.zeros_like(audio)
    norm = np.zeros(n, dtype=np.float32)
    pshift = PitchShift(semitones=0.0)
    for start in range(0, n, hop):
        end = min(start + frame, n)
        seg = audio[:, start:end]
        if seg.shape[1] < frame:
            seg = np.pad(seg, ((0, 0), (0, frame - seg.shape[1])))
        phase = (start / sr) * 2 * np.pi * rate_hz
        pshift.semitones = depth_st * np.sin(phase)
        shifted = pshift(seg, sr)
        k = end - start
        w = window[:k]
        out[:, start:end] += shifted[:, :k] * w
        norm[start:end] += w
    norm[norm == 0] = 1
    return (out / norm).astype(np.float32)
