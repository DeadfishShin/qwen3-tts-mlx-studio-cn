import pytest

import generation
from generation import GenRequest, api_language, run_single
from state import AppContext, AppSettings
from ui import strings as S


@pytest.fixture(autouse=True)
def silence_gradio(monkeypatch):
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)


def test_api_language_maps_auto():
    assert api_language("Auto-detect") == "auto"


def test_api_language_passthrough():
    assert api_language("English") == "English"


def test_engine_gets_auto_history_keeps_label(fake_engine, fake_history, tmp_path):
    s = AppSettings()
    s.output_dir = str(tmp_path / "out")
    ctx = AppContext(engine=fake_engine, library=None, history=fake_history,
                     yt=None, settings=s)
    out = list(run_single(ctx, GenRequest(mode="custom_voice", text="hello",
                                          language="Auto-detect", speaker="ryan")))
    assert out[-1][1].startswith("生成用时")
    # engine received the API value (via the streaming path)
    assert fake_engine.calls[-1] == ("stream_generate_custom_voice", "hello", "auto")
    # history records what the user chose
    assert fake_history.entries[0]["language"] == "Auto-detect"
