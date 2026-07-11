import os
import time

import numpy as np
import pytest

from state import AppContext, AppSettings
from generation import (
    GenerationTimeout, generate_with_timeout, save_audio,
    is_model_cached, loading_status,
)


def make_ctx(fake_engine, fake_history, tmp_path):
    s = AppSettings()
    s.output_dir = str(tmp_path / "out")
    return AppContext(engine=fake_engine, library=None, history=fake_history,
                      yt=None, settings=s)


def test_timeout_raises():
    with pytest.raises(GenerationTimeout):
        generate_with_timeout(time.sleep, 2, timeout_seconds=0.2)


def test_timeout_passes_result():
    assert generate_with_timeout(lambda: 42, timeout_seconds=5) == 42


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
