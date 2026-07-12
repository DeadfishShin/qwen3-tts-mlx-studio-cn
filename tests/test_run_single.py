import gradio as gr
import pytest

import generation
from generation import GenRequest, run_single
from state import AppContext, AppSettings

SKIP = gr.skip()


def is_skip(x):
    return x == SKIP


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
    assert out == [(SKIP, SKIP, "Enter text first")]
    assert fake_engine.calls == []


def test_design_requires_instruct(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_design", text="hello",
                                          language="English", instruct=" ")))
    assert out == [(SKIP, SKIP, "Describe the voice first")]


def test_success_streams_chunks_then_result(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan",
                                          instruct="calm")))
    # loading yield + 3 chunk yields + final
    assert len(out) == 5
    assert "…" in out[0][2]                        # loading/downloading message
    for audio_out, state_out, status in out[1:4]:
        sr, chunk = audio_out                      # streamed chunk tuples
        assert len(chunk) == int(sr * 0.5)
        assert is_skip(state_out)
        assert status.startswith("Generating…")
    final_audio, final_state, final_status = out[-1]
    assert is_skip(final_audio)                    # combine_stream owns playback
    sr, full = final_state
    assert len(full) == 3 * int(sr * 0.5)          # authoritative concat
    assert final_status.startswith("Generated in ")
    assert "fake/custom_voice" in final_status
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
    assert len(out) == 4                           # 3 chunks + final, no loading


def test_stream_playback_off_single_delivery(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path, stream_playback=False)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert all(is_skip(a) for a, _, _ in out[:-1])
    final_audio, final_state, _ = out[-1]
    sr, full = final_audio                          # full waveform delivered once
    assert len(full) == 3 * int(sr * 0.5)
    assert final_audio == final_state


def test_cancel_keeps_partial_no_history(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    fake_engine.chunk_hook = lambda i: ctx.cancel_event.set() if i == 1 else None
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    final_audio, final_state, status = out[-1]
    assert status.startswith("Stopped — kept ")
    sr, partial = final_state                      # manual save still works
    assert len(partial) == 2 * int(sr * 0.5)       # chunks 0 and 1 kept
    assert fake_history.entries == []


def test_timeout_autocancels_keeps_partial(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.settings.timeout = 0
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    _, final_state, status = out[-1]
    assert "Timed out" in status and "kept" in status
    assert final_state is not None and not is_skip(final_state)
    assert fake_history.entries == []


def test_run_clears_stale_cancel(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.cancel_event.set()                          # stale from a previous Stop
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert out[-1][2].startswith("Generated in ")


def test_autosave(fake_engine, fake_history, tmp_path):
    import os
    ctx = make_ctx(fake_engine, fake_history, tmp_path, autosave=True)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert " | Saved: " in out[-1][2]
    assert len(os.listdir(ctx.settings.output_dir)) == 1


def test_engine_failure(fake_engine, fake_history, tmp_path):
    fake_engine.fail_modes = {"custom_voice"}
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="English", speaker="ryan")))
    assert is_skip(out[-1][0])
    assert out[-1][2].startswith("Error: ")
    assert fake_history.entries == []


# --- voice_clone specifics (behavior carried over from the pre-Stage-3 pipeline) ---

def test_clone_library_voice_resolution(fake_engine, fake_history, tmp_path):
    lib = FakeLibrary({"narrator": {"ref_text": "the ref transcript"}})
    ctx = make_ctx(fake_engine, fake_history, tmp_path, library=lib)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          library_voice="narrator")))
    assert out[-1][2].startswith("Generated in ")
    assert fake_history.entries[0]["voice_params"] == "ref: narrator"


def test_clone_missing_library_voice(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          library_voice="ghost")))
    assert out == [(SKIP, SKIP, "Voice not found")]


def test_clone_requires_ref_audio(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English")))
    assert out == [(SKIP, SKIP, "No reference audio")]


def test_clone_requires_ref_text(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          ref_audio="/tmp/x.wav", ref_text=" ")))
    assert out == [(SKIP, SKIP, "No reference transcript")]


def test_clone_uploaded_ref_history_and_denoise_status(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path, denoise_ref=True)
    out = list(run_single(ctx, GenRequest(mode="voice_clone", text="hello",
                                          language="English",
                                          ref_audio="/tmp/x.wav", ref_text="hi there")))
    assert " | Noise reduction applied" in out[-1][2]
    assert fake_history.entries[0]["voice_params"] == "ref: uploaded"
    assert fake_history.entries[0]["mode"] == "voice_clone"
