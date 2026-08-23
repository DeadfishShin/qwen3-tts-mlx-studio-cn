import json
import types

import gradio as gr
import numpy as np
import soundfile as sf

import generation
from generation import (
    GenRequest,
    normalize_voice_design_seed,
    resolve_voice_design_seed,
    run_single,
)
from history import GenerationHistory
from state import AppContext, AppSettings
from ui import strings as S
from ui.tabs.history_tab import history_regenerate
from ui.tabs.voice_design import save_design_to_library
from voice_library import VoiceLibrary


def test_fixed_seed_preserves_requested_integer():
    assert resolve_voice_design_seed(False, 123456) == 123456
    assert normalize_voice_design_seed("123456") == 123456


def test_random_mode_resolves_fresh_seed_from_injected_entropy(monkeypatch):
    values = iter((111, 222))
    monkeypatch.setattr(generation.secrets, "randbelow", lambda _: next(values))
    assert resolve_voice_design_seed(True, 999) == 111
    assert resolve_voice_design_seed(True, 999) == 222


def test_seed_range_rejects_fractional_boolean_and_out_of_range_values():
    for value in (1.5, True, -1, 2**32, None):
        try:
            normalize_voice_design_seed(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid seed {value!r}")


def test_voice_design_seed_reaches_engine_history_and_status(
    fake_engine, fake_history, tmp_path, monkeypatch
):
    captured = {}

    def stream_voice_design(text, language, instruct, **kwargs):
        captured.update(kwargs)
        yield from fake_engine._stream("voice_design", "voice_design")

    fake_engine.stream_generate_voice_design = stream_voice_design
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)
    settings = AppSettings()
    settings.output_dir = str(tmp_path / "out")
    ctx = AppContext(
        engine=fake_engine, library=None, history=fake_history,
        yt=None, settings=settings,
    )
    result = list(run_single(ctx, GenRequest(
        mode="voice_design", text="你好。", language="Chinese",
        voice_description="温和、清晰的中文女声",
        style_instruction="自然停顿", random_seed=False, seed=123456,
    )))

    assert captured["seed"] == 123456
    assert result[-1][1].endswith(S.VD_SEED_USED.format(seed=123456))
    entry = fake_history.entries[0]
    assert entry["seed"] == 123456
    assert entry["seed_mode"] == "fixed"
    assert entry["voice_description"] == "温和、清晰的中文女声"
    assert entry["style_instruction"] == "自然停顿"
    assert entry["temperature"] == settings.temperature
    assert ctx.last_voice_design_seed == 123456


def test_random_voice_design_seed_is_recorded_as_actual_value(
    fake_engine, fake_history, tmp_path, monkeypatch
):
    monkeypatch.setattr(generation.secrets, "randbelow", lambda _: 654321)
    settings = AppSettings()
    settings.output_dir = str(tmp_path / "out")
    ctx = AppContext(
        engine=fake_engine, library=None, history=fake_history,
        yt=None, settings=settings,
    )
    result = list(run_single(ctx, GenRequest(
        mode="voice_design", text="你好。", language="Chinese",
        voice_description="清晰的中文女声", random_seed=True,
    )))

    assert result[-1][1].endswith(S.VD_SEED_USED.format(seed=654321))
    assert fake_history.entries[0]["seed"] == 654321
    assert fake_history.entries[0]["seed_mode"] == "random"


def test_invalid_fixed_seed_fails_before_engine_dispatch(
    fake_engine, fake_history, tmp_path, monkeypatch
):
    monkeypatch.setattr(generation.gr, "Warning", lambda *a, **k: None)
    settings = AppSettings()
    settings.output_dir = str(tmp_path / "out")
    ctx = AppContext(
        engine=fake_engine, library=None, history=fake_history,
        yt=None, settings=settings,
    )
    output = list(run_single(ctx, GenRequest(
        mode="voice_design", text="你好。", language="Chinese",
        voice_description="清晰的中文女声", random_seed=False, seed=2**32,
    )))

    assert output == [(gr.skip(), S.VD_SEED_INVALID)]
    assert fake_engine.calls == []
    assert fake_history.entries == []


def test_legacy_history_without_seed_loads_with_safe_defaults(tmp_path):
    payload = [{
        "id": "legacy",
        "timestamp": "2026-01-01T00:00:00",
        "mode": "voice_design",
        "text": "旧记录",
        "language": "Chinese",
        "duration": 1.0,
        "speaker": "",
        "voice_params": "old description",
        "audio_file": "legacy.wav",
    }]
    (tmp_path / "index.json").write_text(json.dumps(payload), encoding="utf-8")
    history = GenerationHistory(str(tmp_path))
    assert history._entries[0].seed is None
    assert history._entries[0].seed_mode == ""


def test_voice_library_and_design_save_keep_seed_separate_from_identity(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24000)
    library = VoiceLibrary(str(tmp_path / "voices"))
    library.save_voice(
        name="播音员", ref_audio_path=str(source), ref_text="测试",
        language="Chinese", description="稳定音色", source="design",
        seed=123456, style_instruction="从容沉稳",
    )
    profile = library.load_voice("播音员")
    assert profile["description"] == "稳定音色"
    assert profile["seed"] == 123456
    assert profile["style_instruction"] == "从容沉稳"
    assert "声音身份" not in profile["description"]


def test_history_settings_show_seed_and_sampling_values(tmp_path):
    history = GenerationHistory(str(tmp_path))
    history.add(
        mode="voice_design", text="你好", language="Chinese",
        audio=(24000, np.zeros(240, dtype=np.float32)),
        voice_description="稳定音色", style_instruction="从容沉稳",
        seed=123456, seed_mode="fixed", temperature=0.9, top_k=50,
        top_p=1.0, repetition_penalty=1.05, max_tokens=4096,
    )
    details = history_regenerate(
        type("Context", (), {"history": history})(), history._entries[0].id
    )
    assert "随机种子：123456（固定）" in details
    assert "最大 token=4096" in details


def test_design_save_passes_last_seed_and_style_provenance(tmp_path, monkeypatch):
    saved = {}

    class Library:
        def save_voice(self, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr("ui.tabs.voice_design.OUTPUT_DIR", str(tmp_path))
    ctx = types.SimpleNamespace(
        library=Library(), last_voice_design_seed=123456,
        last_voice_design_style_instruction="从容沉稳",
    )
    save_design_to_library(
        ctx, (24000, np.zeros(240, dtype=np.float32)), "设计声音",
        "Chinese", "稳定音色", "你好",
    )
    assert saved["seed"] == 123456
    assert saved["description"] == "稳定音色"
    assert saved["style_instruction"] == "从容沉稳"


def test_voice_design_ui_exposes_seed_controls(fake_engine, fake_history):
    class Library:
        def list_voices(self):
            return []

    ctx = AppContext(
        engine=fake_engine, library=Library(), history=fake_history,
        yt=None, settings=AppSettings(),
    )
    from ui.tabs.voice_design import build

    with gr.Blocks():
        ui = build(ctx)
    assert ui.vd_random_seed.label == S.VD_RANDOM_EACH
    assert ui.vd_seed.label == S.VD_SEED
    assert ui.vd_use_last_seed.value == S.VD_USE_LAST_SEED
