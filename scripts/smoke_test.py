"""Engine-level smoke test with real models. Slow (~5 min full run).

Usage: .venv/bin/python scripts/smoke_test.py [--fast]
Covers: single generation x3 modes, batch x2 modes, ASR transcription.
Voice-clone reference audio is self-generated via custom voice.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from engine import TTSEngine

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="skip voice-design and ASR")
args = parser.parse_args()

engine = TTSEngine()
failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}", flush=True)
    except Exception as e:
        failures.append(name)
        print(f"FAIL {name}: {e}", flush=True)


def assert_audio(result, min_s=0.3):
    sr, audio = result
    dur = len(audio) / sr
    assert len(audio) > 0 and dur > min_s, f"suspicious audio: {dur:.2f}s"
    assert np.isfinite(audio).all(), "non-finite samples"


ref_wav = None


def single_custom():
    global ref_wav
    r = engine.generate_custom_voice(
        "This is the smoke test for single custom voice generation.",
        "ryan", "English")
    assert_audio(r)
    sr, audio = r
    ref_wav = Path(tempfile.mkdtemp()) / "ref.wav"
    sf.write(str(ref_wav), audio, sr)


def batch_custom():
    rs = engine.batch_generate_custom_voice(
        ["Batch line one.", "Batch line two, slightly longer.", "Batch line three."],
        "ryan", "English")
    assert len(rs) == 3
    for r in rs:
        assert_audio(r, min_s=0.2)


def single_design():
    r = engine.generate_voice_design(
        "Voice design smoke test line.", "English",
        "A calm, deep male narrator voice.")
    assert_audio(r)


def batch_design():
    rs = engine.batch_generate_voice_design(
        ["Design batch one.", "Design batch two."],
        "English", "A bright, energetic female voice.")
    assert len(rs) == 2
    for r in rs:
        assert_audio(r, min_s=0.2)


def single_clone():
    assert ref_wav is not None, "custom-voice step must run first"
    r = engine.generate_voice_clone(
        "Cloning smoke test output sentence.",
        str(ref_wav),
        "This is the smoke test for single custom voice generation.",
        language="English")
    assert_audio(r)


def batch_clone():
    assert ref_wav is not None, "custom-voice step must run first"
    rs = engine.batch_generate_voice_clone(
        ["Batched clone line one.", "Batched clone line two.", "Batched clone line three."],
        str(ref_wav),
        "This is the smoke test for single custom voice generation.",
        language="English")
    assert len(rs) == 3
    for r in rs:
        assert_audio(r, min_s=0.2)


def vad_trim_real():
    assert ref_wav is not None
    import soundfile as sf
    from audio_utils import trim_ref_silence
    audio, sr = sf.read(str(ref_wav))
    padded = np.concatenate(
        [np.zeros(int(2.0 * sr), dtype=np.float32), audio.astype(np.float32),
         np.zeros(int(3.0 * sr), dtype=np.float32)])
    padded_path = ref_wav.parent / "padded.wav"
    sf.write(str(padded_path), padded, sr)
    trimmed_path = trim_ref_silence(str(padded_path))
    assert trimmed_path != str(padded_path), "trim did not fire on padded clip"
    trimmed, tsr = sf.read(trimmed_path)
    orig_dur, trimmed_dur = len(audio) / sr, len(trimmed) / tsr
    assert abs(trimmed_dur - orig_dur) < 1.0, (
        f"trimmed {trimmed_dur:.1f}s vs original {orig_dur:.1f}s")


def asr():
    assert ref_wav is not None
    text = engine.transcribe(str(ref_wav), language="auto")
    assert isinstance(text, str) and len(text.split()) >= 4, f"thin transcript: {text!r}"


check("single custom_voice", single_custom)
check("batch custom_voice (canary)", batch_custom)
if not args.fast:
    check("single voice_design", single_design)
    check("batch voice_design", batch_design)
check("single voice_clone (ICL)", single_clone)
check("batch voice_clone shared-ref (canary)", batch_clone)
check("vad trim (real model)", vad_trim_real)
if not args.fast:
    check("asr transcribe", asr)

def cache_returned():
    import mlx.core as mx
    cache_mb = mx.get_cache_memory() / 1e6
    assert cache_mb < 50, f"MLX buffer cache not returned after generation: {cache_mb:.0f} MB"


check("mlx cache returned", cache_returned)

engine.unload_model()
if failures:
    print(f"\nSMOKE TEST FAILED: {failures}")
    sys.exit(1)
print("\nSMOKE TEST PASSED")
