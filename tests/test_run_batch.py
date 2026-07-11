import pytest

import generation
from generation import GenRequest, run_batch
from state import AppContext, AppSettings

from test_run_single import FakeLibrary


@pytest.fixture(autouse=True)
def silence_gradio(monkeypatch):
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)


class NoProgress:
    def __call__(self, *a, **k):
        pass

    def tqdm(self, iterable, **k):
        return iterable


def make_ctx(fake_engine, fake_history, tmp_path, library=None):
    s = AppSettings()
    s.output_dir = str(tmp_path / "out")
    s.batch_size = 2
    return AppContext(engine=fake_engine, library=library or FakeLibrary(),
                      history=fake_history, yt=None, settings=s)


def req():
    return GenRequest(mode="custom_voice", text="One.\n\nTwo.\n\nThree.",
                      language="English", speaker="ryan")


def test_batch_success(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    audio_update, rows, status = run_batch(ctx, req(), "paragraph", 300, NoProgress())
    assert status == "Generated 3/3 segments"
    assert len(rows) == 3
    assert rows[0][0] == "1" and rows[0][2].endswith("s")
    # batch_size=2 -> two batched calls
    batch_calls = [c for c in fake_engine.calls if c[0] == "batch_generate_custom_voice"]
    assert len(batch_calls) == 2
    assert len(fake_history.entries) == 1
    assert fake_history.entries[0]["text"] == "[Batch: 3 segments]"
    assert fake_history.entries[0]["speaker"] == "ryan"
    assert fake_history.entries[0]["voice_params"] == "batch (paragraph)"


def test_batch_failure_falls_back_to_individual(fake_engine, fake_history, tmp_path):
    fake_engine.fail_batch = True
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    audio_update, rows, status = run_batch(ctx, req(), "paragraph", 300, NoProgress())
    assert status == "Generated 3/3 segments"
    singles = [c for c in fake_engine.calls if c[0] == "generate_custom_voice"]
    assert len(singles) == 3


def test_individual_failures_reported(fake_engine, fake_history, tmp_path):
    fake_engine.fail_batch = True
    fake_engine.fail_modes = {"custom_voice"}
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    audio_update, rows, status = run_batch(ctx, req(), "paragraph", 300, NoProgress())
    assert audio_update is None
    assert status == "All segments failed"
    assert all(r[2].startswith("Failed: ") for r in rows)
    assert fake_history.entries == []


def test_empty_text(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="custom_voice", text="   ", language="English", speaker="ryan")
    audio_update, rows, status = run_batch(ctx, r, "paragraph", 300, NoProgress())
    assert audio_update is None and status == "No segments"


def test_too_many_segments(fake_engine, fake_history, tmp_path):
    import config
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    many = "\n\n".join(f"Seg {i}." for i in range(config.MAX_BATCH_SEGMENTS + 1))
    r = GenRequest(mode="custom_voice", text=many, language="English", speaker="ryan")
    audio_update, rows, status = run_batch(ctx, r, "paragraph", 300, NoProgress())
    assert audio_update is None and "Too many segments" in status


def test_design_batch_requires_instruct(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="voice_design", text="One.\n\nTwo.", language="English",
                   instruct="  ")
    audio_update, rows, status = run_batch(ctx, r, "paragraph", 300, NoProgress())
    # batch handler wording differs from single ("Describe voice first")
    assert audio_update is None and status == "Describe voice first"
    assert rows == [["(empty)", "", ""]]


def test_clone_mode_runs_sequentially(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="voice_clone", text="One.\n\nTwo.", language="English",
                   ref_audio="/tmp/ref.wav", ref_text="ref transcript")
    audio_update, rows, status = run_batch(ctx, r, "paragraph", 300, NoProgress())
    assert status == "Generated 2/2 segments"
    clone_calls = [c for c in fake_engine.calls if c[0] == "generate_voice_clone"]
    assert len(clone_calls) == 2               # no batch method used
    assert fake_history.entries[0]["voice_params"] == "batch ref: uploaded"
    assert "speaker" not in fake_history.entries[0]


def test_clone_batch_validation_strings(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="voice_clone", text="One.", language="English",
                   ref_audio="/tmp/ref.wav", ref_text=" ")
    audio_update, rows, status = run_batch(ctx, r, "paragraph", 300, NoProgress())
    # batch wording is "No transcript" (single says "No reference transcript")
    assert status == "No transcript"
    assert rows == [["(error)", "", ""]]


def test_clone_batch_denoise_suffix(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.settings.denoise_ref = True
    r = GenRequest(mode="voice_clone", text="One.\n\nTwo.", language="English",
                   ref_audio="/tmp/ref.wav", ref_text="ref transcript")
    audio_update, rows, status = run_batch(ctx, r, "paragraph", 300, NoProgress())
    assert status == "Generated 2/2 segments | Noise reduction applied"
