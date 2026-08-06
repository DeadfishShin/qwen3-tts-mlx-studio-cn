import os

import numpy as np
import pytest

from state import AppContext, AppSettings
from generation import (
    GenerationCancelled, GenerationTimeout, save_audio, stream_to_audio,
    is_model_cached, loading_status,
)


def make_ctx(fake_engine, fake_history, tmp_path):
    s = AppSettings()
    s.output_dir = str(tmp_path / "out")
    return AppContext(engine=fake_engine, library=None, history=fake_history,
                      yt=None, settings=s)


def test_stream_to_audio_concatenates(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    sr, audio = stream_to_audio(
        ctx, fake_engine.stream_generate_custom_voice("hi", "ryan", "English"))
    assert sr == fake_engine.sr
    assert len(audio) == 3 * int(sr * 0.5)          # 3 fake chunks of 0.5 s


def test_stream_to_audio_cancel(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    fake_engine.chunk_hook = lambda i: ctx.cancel_event.set() if i == 1 else None
    with pytest.raises(GenerationCancelled):
        stream_to_audio(
            ctx, fake_engine.stream_generate_custom_voice("hi", "ryan", "English"))


def test_stream_to_audio_timeout(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.settings.timeout = 0                        # any elapsed time > 0 trips it
    with pytest.raises(GenerationTimeout):
        stream_to_audio(
            ctx, fake_engine.stream_generate_custom_voice("hi", "ryan", "English"))


def test_save_audio_writes_wav(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    audio = (24000, np.zeros(24000, dtype=np.float32))
    msg = save_audio(ctx, audio, prefix="unit")
    assert msg.startswith("Saved: ")
    saved = os.listdir(ctx.settings.output_dir)
    assert len(saved) == 1 and saved[0].startswith("unit_") and saved[0].endswith(".wav")


def test_save_audio_none(fake_engine, fake_history, tmp_path, monkeypatch):
    import generation
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    assert save_audio(ctx, None) == "No audio to save"


def test_is_model_cached_false_for_garbage():
    assert is_model_cached("no-such/repo-xyz") is False


def test_loading_status(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    msg = loading_status(ctx, "custom_voice")          # not loaded
    assert msg is not None and "fake/custom_voice" in msg
    fake_engine.generate_custom_voice("x", "ryan", "English")  # marks loaded
    assert loading_status(ctx, "custom_voice") is None
