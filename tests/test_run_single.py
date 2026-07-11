import pytest

import generation
from generation import GenRequest, run_single
from state import AppContext, AppSettings


@pytest.fixture(autouse=True)
def silence_gradio(monkeypatch):
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)


class FakeLibrary:
    def __init__(self, voices=None):
        self.voices = voices or {}

    def load_voice(self, name):
        if name not in self.voices:
            raise FileNotFoundError(name)
        return self.voices[name]

    def get_ref_audio_path(self, name):
        if name not in self.voices:
            raise FileNotFoundError(name)
        return f"/fake/{name}.wav"


def make_ctx(fake_engine, fake_history, tmp_path, library=None, **settings):
    s = AppSettings()
    s.output_dir = str(tmp_path / "out")
    for k, v in settings.items():
        setattr(s, k, v)
    return AppContext(engine=fake_engine, library=library or FakeLibrary(),
                      history=fake_history, yt=None, settings=s)


def test_empty_text_rejected(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="  ",
                                          language="English", speaker="ryan")))
    assert out == [(None, "Enter text first")]
    assert fake_engine.calls == []


def test_design_requires_instruct(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_design", text="hello",
                                          language="English", instruct=" ")))
    assert out == [(None, "Describe the voice first")]


def test_success_yields_loading_then_result(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan",
                                          instruct="calm")))
    assert len(out) == 2                       # loading yield + result yield
    assert "…" in out[0][1]                    # loading/downloading message
    audio_update, status = out[1]
    assert status.startswith("Generated in ")
    assert "fake/custom_voice" in status
    assert len(fake_history.entries) == 1
    assert fake_history.entries[0]["mode"] == "custom_voice"
    assert fake_history.entries[0]["speaker"] == "ryan"
    assert fake_history.entries[0]["voice_params"] == "calm"


def test_no_loading_yield_when_loaded(fake_engine, fake_history, tmp_path):
    fake_engine.generate_custom_voice("warm", "ryan", "English")
    fake_engine.calls.clear()
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert len(out) == 1


def test_autosave(fake_engine, fake_history, tmp_path):
    import os
    ctx = make_ctx(fake_engine, fake_history, tmp_path, autosave=True)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert " | Saved: " in out[-1][1]
    assert len(os.listdir(ctx.settings.output_dir)) == 1


def test_engine_failure(fake_engine, fake_history, tmp_path):
    fake_engine.fail_modes = {"custom_voice"}
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert out[-1][0] is None
    assert out[-1][1].startswith("Error: ")
    assert fake_history.entries == []


def test_timeout_message(fake_engine, fake_history, tmp_path, monkeypatch):
    import time as _time

    def slow(*a, **k):
        _time.sleep(1.0)

    monkeypatch.setattr(fake_engine, "generate_custom_voice", slow)
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.settings.timeout = 0.2
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert "timed out" in out[-1][1]


# --- voice_clone specifics (behavior copied from app.py:325-385) ---

def test_clone_library_voice_resolution(fake_engine, fake_history, tmp_path):
    lib = FakeLibrary({"narrator": {"ref_text": "the ref transcript"}})
    ctx = make_ctx(fake_engine, fake_history, tmp_path, library=lib)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          library_voice="narrator")))
    assert out[-1][1].startswith("Generated in ")
    assert fake_history.entries[0]["voice_params"] == "ref: narrator"


def test_clone_missing_library_voice(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          library_voice="ghost")))
    assert out == [(None, "Voice not found")]


def test_clone_requires_ref_audio(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English")))
    assert out == [(None, "No reference audio")]


def test_clone_requires_ref_text(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          ref_audio="/tmp/x.wav", ref_text=" ")))
    assert out == [(None, "No reference transcript")]


def test_clone_uploaded_ref_history_and_denoise_status(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path, denoise_ref=True)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          ref_audio="/tmp/x.wav", ref_text="hi there")))
    assert " | Noise reduction applied" in out[-1][1]
    assert fake_history.entries[0]["voice_params"] == "ref: uploaded"
    assert fake_history.entries[0]["mode"] == "voice_clone"
