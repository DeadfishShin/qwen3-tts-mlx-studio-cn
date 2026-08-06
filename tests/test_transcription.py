import pytest

from generation import stream_transcription
from state import AppContext, AppSettings


@pytest.fixture(autouse=True)
def silence_gradio(monkeypatch):
    import generation
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)


def make_ctx(fake_engine, fake_history):
    return AppContext(engine=fake_engine, library=None, history=fake_history,
                      yt=None, settings=AppSettings())


def test_stream_transcription_accumulates(fake_engine, fake_history):
    ctx = make_ctx(fake_engine, fake_history)
    out = list(stream_transcription(ctx, "/tmp/x.wav", "auto"))
    # loading + 2 deltas + final
    assert len(out) == 4
    final_update, final_status = out[-1]
    assert final_update["value"] == "fake transcript"
    assert final_status == "Transcribed (2 words)"


def test_stream_transcription_stop_keeps_partial(fake_engine, fake_history):
    ctx = make_ctx(fake_engine, fake_history)
    gen = stream_transcription(ctx, "/tmp/x.wav", "auto")
    assert next(gen)[1] == "Loading ASR model..."
    next(gen)                                     # first delta arrived
    ctx.cancel_event.set()
    update, status = next(gen)
    assert status == "Stopped — partial transcript kept"
    assert update["value"] == "fake"
    assert list(gen) == []                        # generator ends cleanly


def test_stream_transcription_error(fake_engine, fake_history):
    def boom(audio_path, language="auto"):
        raise RuntimeError("no asr")
        yield  # pragma: no cover — make it a generator

    fake_engine.stream_transcribe = boom
    ctx = make_ctx(fake_engine, fake_history)
    out = list(stream_transcription(ctx, "/tmp/x.wav", "auto"))
    assert out[-1][1].startswith("Error: ")
