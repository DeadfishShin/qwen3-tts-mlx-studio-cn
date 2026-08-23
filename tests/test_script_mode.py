import gradio as gr
import pytest

import ui.tabs.script_mode as sm
from state import AppContext, AppSettings
from test_run_single import FakeLibrary
from ui import strings as S


@pytest.fixture(autouse=True)
def silence_gradio(monkeypatch):
    import generation
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)
    monkeypatch.setattr(sm.gr, "Warning", lambda *a, **k: None)


class NoProgress:
    def __call__(self, *a, **k):
        pass


def make_ctx(fake_engine, fake_history, tmp_path):
    s = AppSettings()
    s.output_dir = str(tmp_path / "out")
    s.batch_size = 2
    return AppContext(engine=fake_engine, library=FakeLibrary(),
                      history=fake_history, yt=None, settings=s)


SCRIPT = "ANNA: Line one.\nANNA: Line two.\nANNA: Line three."
ASSIGN = {"ANNA": {"mode": "custom_voice", "speaker": "ryan",
                   "language": "English", "instruct": "", "library_voice": "None"}}


def drain(gen):
    ys = list(gen)
    return ys, ys[-1]


def test_script_success_generates_all_lines(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ys, (audio_update, table_md, status) = drain(
        sm.generate_script_handler(ctx, SCRIPT, ASSIGN, 300, NoProgress()))
    assert status == S.SM_GENERATED.format(done=3, total=3)
    assert "Line one." in table_md
    assert len(fake_history.entries) == 1
    progress = [s for _, _, s in ys[:-1]]
    assert any(s.startswith("行 2/3") for s in progress)


def test_script_cancel_keeps_completed_lines(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    orig = fake_engine.batch_generate_custom_voice

    def cancel_after_first(*a, **k):
        r = orig(*a, **k)
        ctx.cancel_event.set()
        return r

    fake_engine.batch_generate_custom_voice = cancel_after_first
    ys, (audio_update, table_md, status) = drain(
        sm.generate_script_handler(ctx, SCRIPT, ASSIGN, 300, NoProgress()))
    assert status == S.SCRIPT_STOPPED.format(done=2, total=3)
    assert audio_update is not None
    assert S.SM_STOPPED in table_md
    assert fake_history.entries == []


def test_script_stale_cancel_cleared(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.cancel_event.set()
    _, (_, _, status) = drain(
        sm.generate_script_handler(ctx, SCRIPT, ASSIGN, 300, NoProgress()))
    assert status == S.SM_GENERATED.format(done=3, total=3)


def test_script_clone_lines_batched_by_library_voice(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.library = FakeLibrary({"narrator": {"ref_text": "the ref transcript"}})
    assign = {"ANNA": {"mode": "voice_clone", "speaker": "ryan",
                       "language": "English", "instruct": "",
                       "library_voice": "narrator"}}
    _, (audio_update, table_md, status) = drain(
        sm.generate_script_handler(ctx, SCRIPT, assign, 300, NoProgress()))
    assert status == S.SM_GENERATED.format(done=3, total=3)
    batch_calls = [c for c in fake_engine.calls if c[0] == "batch_generate_voice_clone"]
    assert len(batch_calls) == 2               # batch_size=2 -> 2 + 1 lines
