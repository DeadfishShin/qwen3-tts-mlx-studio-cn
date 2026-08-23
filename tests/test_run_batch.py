import pytest

import generation
from generation import GenRequest, run_batch
from state import AppContext, AppSettings
from ui import strings as S

from test_run_single import FakeLibrary


@pytest.fixture(autouse=True)
def silence_gradio(monkeypatch):
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)


class NoProgress:
    def __call__(self, *a, **k):
        pass

    def tqdm(self, iterable, **k):
        return iterable


def drain(gen):
    """Run a generator pipeline to completion, return (all_yields, final)."""
    ys = list(gen)
    return ys, ys[-1]


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
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, req(), "paragraph", 300, NoProgress()))
    assert status == S.BATCH_GENERATED_STATUS.format(done=3, total=3)
    assert len(rows) == 3
    assert rows[0][0] == "1" and rows[0][2].endswith("s")
    # batch_size=2 -> two batched calls
    batch_calls = [c for c in fake_engine.calls if c[0] == "batch_generate_custom_voice"]
    assert len(batch_calls) == 2
    assert len(fake_history.entries) == 1
    assert fake_history.entries[0]["text"] == "[Batch: 3 segments]"
    assert fake_history.entries[0]["speaker"] == "ryan"
    assert fake_history.entries[0]["voice_params"] == "batch (paragraph)"


def test_batch_live_progress_statuses(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ys, final = drain(run_batch(ctx, req(), "paragraph", 300, NoProgress()))
    progress_statuses = [s for _, _, s in ys[:-1]]
    assert any(s.startswith("片段 2/3") for s in progress_statuses)
    assert final[2] == S.BATCH_GENERATED_STATUS.format(done=3, total=3)


def test_batch_cancel_keeps_completed(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)   # batch_size=2 -> 2 calls
    orig = fake_engine.batch_generate_custom_voice

    def cancel_after_first(*a, **k):
        r = orig(*a, **k)
        ctx.cancel_event.set()
        return r

    fake_engine.batch_generate_custom_voice = cancel_after_first
    ys, (audio_update, rows, status) = drain(
        run_batch(ctx, req(), "paragraph", 300, NoProgress()))
    assert status == S.BATCH_STOPPED.format(done=2, total=3)
    assert audio_update is not None                # completed audio combined
    assert fake_history.entries == []


def test_batch_stale_cancel_cleared(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.cancel_event.set()
    _, (_, _, status) = drain(run_batch(ctx, req(), "paragraph", 300, NoProgress()))
    assert status == S.BATCH_GENERATED_STATUS.format(done=3, total=3)


def test_batch_failure_falls_back_to_individual(fake_engine, fake_history, tmp_path):
    fake_engine.fail_batch = True
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, req(), "paragraph", 300, NoProgress()))
    assert status == S.BATCH_GENERATED_STATUS.format(done=3, total=3)
    # fallback retries go through the cancel-aware streaming path
    singles = [c for c in fake_engine.calls if c[0] == "stream_generate_custom_voice"]
    assert len(singles) == 3


def test_individual_failures_reported(fake_engine, fake_history, tmp_path):
    fake_engine.fail_batch = True
    fake_engine.fail_modes = {"custom_voice"}
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, req(), "paragraph", 300, NoProgress()))
    assert audio_update is None
    assert status == S.ALL_SEGMENTS_FAILED
    assert all(r[2].startswith("失败：") for r in rows)
    assert fake_history.entries == []


def test_empty_text(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="custom_voice", text="   ", language="English", speaker="ryan")
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, r, "paragraph", 300, NoProgress()))
    assert audio_update is None and status == S.NO_SEGMENTS


def test_too_many_segments(fake_engine, fake_history, tmp_path):
    import config
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    many = "\n\n".join(f"Seg {i}." for i in range(config.MAX_BATCH_SEGMENTS + 1))
    r = GenRequest(mode="custom_voice", text=many, language="English", speaker="ryan")
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, r, "paragraph", 300, NoProgress()))
    assert audio_update is None and status.startswith("片段过多")


def test_design_batch_requires_instruct(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="voice_design", text="One.\n\nTwo.", language="English",
                   instruct="  ")
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, r, "paragraph", 300, NoProgress()))
    # batch handler wording differs from single ("Describe voice first")
    assert audio_update is None and status == S.VD_BATCH_DESCRIPTION_REQUIRED
    assert rows == [[S.SEGMENT_EMPTY, "", ""]]


def test_clone_mode_batches(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="voice_clone", text="One.\n\nTwo.", language="English",
                   ref_audio="/tmp/ref.wav", ref_text="ref transcript")
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, r, "paragraph", 300, NoProgress()))
    assert status == S.BATCH_GENERATED_STATUS.format(done=2, total=2)
    batch_calls = [c for c in fake_engine.calls if c[0] == "batch_generate_voice_clone"]
    assert len(batch_calls) == 1               # one shared-ref batched call
    assert fake_history.entries[0]["voice_params"] == "batch ref: uploaded"
    assert "speaker" not in fake_history.entries[0]


def test_clone_batch_falls_back_individually(fake_engine, fake_history, tmp_path):
    fake_engine.fail_batch = True
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="voice_clone", text="One.\n\nTwo.", language="English",
                   ref_audio="/tmp/ref.wav", ref_text="ref transcript")
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, r, "paragraph", 300, NoProgress()))
    assert status == S.BATCH_GENERATED_STATUS.format(done=2, total=2)
    singles = [c for c in fake_engine.calls if c[0] == "stream_generate_voice_clone"]
    assert len(singles) == 2


def test_clone_batch_validation_strings(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    r = GenRequest(mode="voice_clone", text="One.", language="English",
                   ref_audio="/tmp/ref.wav", ref_text=" ")
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, r, "paragraph", 300, NoProgress()))
    # batch wording is "No transcript" (single says "No reference transcript")
    assert status == S.NO_REF_TEXT
    assert rows == [[S.SEGMENT_ERROR, "", ""]]


def test_clone_batch_denoise_suffix(fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    ctx.settings.denoise_ref = True
    r = GenRequest(mode="voice_clone", text="One.\n\nTwo.", language="English",
                   ref_audio="/tmp/ref.wav", ref_text="ref transcript")
    _, (audio_update, rows, status) = drain(
        run_batch(ctx, r, "paragraph", 300, NoProgress()))
    assert status == S.BATCH_GENERATED_STATUS.format(done=2, total=2) + S.NOISE_REDUCTION_SUFFIX
