import pytest

from generation import stream_transcription
from state import AppContext, AppSettings
from ui import strings as S


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
    assert final_status == S.TRANSCRIBED.format(words=2)


def test_stream_transcription_stop_keeps_partial(fake_engine, fake_history):
    ctx = make_ctx(fake_engine, fake_history)
    gen = stream_transcription(ctx, "/tmp/x.wav", "auto")
    assert next(gen)[1] == S.ASR_LOADING
    next(gen)                                     # first delta arrived
    ctx.cancel_event.set()
    update, status = next(gen)
    assert status == S.TRANSCRIBE_STOPPED
    assert update["value"] == "fake"
    assert list(gen) == []                        # generator ends cleanly


def test_tab_handler_maps_auto_detect(fake_engine, fake_history):
    from ui.tabs.transcription import transcribe_audio
    ctx = make_ctx(fake_engine, fake_history)
    seen = {}
    orig = fake_engine.stream_transcribe

    def spy(audio_path, language="auto"):
        seen["language"] = language
        return orig(audio_path, language=language)

    fake_engine.stream_transcribe = spy
    list(transcribe_audio(ctx, "/tmp/x.wav", "Auto-detect"))
    assert seen["language"] == "auto"


def test_stream_transcription_error(fake_engine, fake_history):
    def boom(audio_path, language="auto"):
        raise RuntimeError("no asr")
        yield  # pragma: no cover — make it a generator

    fake_engine.stream_transcribe = boom
    ctx = make_ctx(fake_engine, fake_history)
    out = list(stream_transcription(ctx, "/tmp/x.wav", "auto"))
    assert out[-1][1].startswith("错误：")
