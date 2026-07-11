import numpy as np
import soundfile as sf

import audio_utils


class FakeVAD:
    def __init__(self, ts):
        self.ts = ts

    def get_speech_timestamps(self, path, return_seconds=True):
        return self.ts


def make_wav(tmp_path, seconds=4.0, sr=24000):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(int(seconds * sr)) * 0.1).astype(np.float32)
    p = tmp_path / "ref.wav"
    sf.write(str(p), audio, sr)
    return str(p), sr


def test_trims_to_speech_span(tmp_path, monkeypatch):
    path, sr = make_wav(tmp_path)
    monkeypatch.setattr(audio_utils, "_get_vad", lambda: FakeVAD([{"start": 1.0, "end": 3.0}]))
    out = audio_utils.trim_ref_silence(path)
    assert out != path
    audio, out_sr = sf.read(out)
    assert abs(len(audio) / out_sr - 2.3) < 0.05     # 2.0s span + 0.15s pad each side


def test_spans_multiple_segments(tmp_path, monkeypatch):
    path, sr = make_wav(tmp_path)
    monkeypatch.setattr(audio_utils, "_get_vad",
                        lambda: FakeVAD([{"start": 0.5, "end": 1.0}, {"start": 2.5, "end": 3.5}]))
    out = audio_utils.trim_ref_silence(path)
    audio, out_sr = sf.read(out)
    assert abs(len(audio) / out_sr - 3.3) < 0.05     # 0.5..3.5 + pads


def test_passthrough_no_speech(tmp_path, monkeypatch):
    path, _ = make_wav(tmp_path)
    monkeypatch.setattr(audio_utils, "_get_vad", lambda: FakeVAD([]))
    assert audio_utils.trim_ref_silence(path) == path


def test_passthrough_tiny_saving(tmp_path, monkeypatch):
    path, _ = make_wav(tmp_path)
    monkeypatch.setattr(audio_utils, "_get_vad", lambda: FakeVAD([{"start": 0.05, "end": 3.99}]))
    assert audio_utils.trim_ref_silence(path) == path
