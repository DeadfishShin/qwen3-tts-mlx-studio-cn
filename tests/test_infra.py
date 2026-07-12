import pytest

from conftest import FakeEngine


def test_fake_engine_generates(fake_engine):
    sr, audio = fake_engine.generate_custom_voice("hi", "ryan", "English")
    assert sr == 24000 and len(audio) > 0
    assert fake_engine.is_model_loaded("custom_voice")


def test_fake_engine_streams(fake_engine):
    chunks = list(fake_engine.stream_generate_custom_voice("hi", "ryan", "English"))
    assert len(chunks) == 3
    sr, first = chunks[0]
    assert sr == fake_engine.sr and len(first) == int(sr * 0.5)


def test_fake_engine_stream_failure():
    eng = FakeEngine(fail_modes={"custom_voice"})
    gen = eng.stream_generate_custom_voice("hi", "ryan", "English")
    with pytest.raises(RuntimeError):
        next(gen)


def test_fake_engine_batch_failure():
    eng = FakeEngine(fail_batch=True)
    with pytest.raises(RuntimeError):
        eng.batch_generate_custom_voice(["a"], "ryan", "English")
