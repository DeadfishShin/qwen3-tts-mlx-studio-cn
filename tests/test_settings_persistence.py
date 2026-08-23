import json
from pathlib import Path

import gradio as gr

from state import AppContext, AppSettings
from engine import TTSEngine
from settings_store import (
    PERSISTED_SETTING_KEYS,
    load_settings,
    save_settings,
    settings_to_dict,
)
from ui.tabs import settings as settings_tab


def test_missing_settings_file_uses_current_defaults(tmp_path):
    result = load_settings(tmp_path / "settings.json")

    assert result.warnings == []
    assert result.settings == AppSettings()
    assert result.settings.jit_compile is False


def test_save_and_restart_restore_all_allow_listed_preferences(tmp_path):
    path = tmp_path / "config" / "settings.json"
    original = AppSettings(
        model_size="0.6B",
        quantization="8bit",
        temperature=0.55,
        top_k=35,
        top_p=0.85,
        repetition_penalty=1.2,
        max_tokens=2048,
        timeout=90,
        batch_size=2,
        output_dir=str(tmp_path / "audio"),
        autosave=True,
        jit_compile=True,
        default_language="Chinese",
        export_format="mp3",
        mp3_bitrate=256,
        loudnorm=True,
        trim_silence=True,
        denoise_ref=True,
    )

    saved, warnings = save_settings(original, path)
    loaded = load_settings(path)

    assert warnings == []
    assert saved == original
    assert loaded.settings == original
    assert loaded.warnings == []
    assert set(json.loads(path.read_text())["version"] for _ in [0]) == {1}
    assert set(json.loads(path.read_text())) == {"version", *PERSISTED_SETTING_KEYS}


def test_missing_and_extra_keys_are_tolerated(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"version": 1, "temperature": 0.55, "future_key": "ignored"}))

    result = load_settings(path)

    assert result.settings.temperature == 0.55
    assert result.settings.top_k == AppSettings().top_k
    assert not any("future" in warning for warning in result.warnings)


def test_malformed_json_does_not_break_startup(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not-json")

    result = load_settings(path)

    assert result.settings == AppSettings()
    assert result.warnings
    assert "设置文件无法读取" in result.warnings[0]


def test_invalid_values_fall_back_safely(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "temperature": 9,
        "top_k": "not-an-int",
        "default_language": "not-a-language",
        "jit_compile": "yes",
    }))

    result = load_settings(path)

    assert result.settings.temperature == AppSettings().temperature
    assert result.settings.top_k == AppSettings().top_k
    assert result.settings.default_language == AppSettings().default_language
    assert result.settings.jit_compile is False
    assert len(result.warnings) == 4


def test_settings_ui_initializes_from_loaded_values(tmp_path):
    settings = AppSettings(temperature=0.55, top_k=35, autosave=True, default_language="Chinese")
    ctx = AppContext(engine=_FakeEngine(), library=None, history=None, yt=None, settings=settings)

    with gr.Blocks():
        ui = settings_tab.build(ctx)

    assert ui.set_temperature.value == 0.55
    assert ui.set_top_k.value == 35
    assert ui.set_autosave.value is True
    assert ui.set_default_language.value == "Chinese"


def test_apply_persists_runtime_settings_without_loading_models(monkeypatch, tmp_path):
    ctx = AppContext(engine=_FakeEngine(), library=None, history=None, yt=None, settings=AppSettings())
    captured = {}

    def fake_save(settings):
        captured["settings"] = settings
        return settings, []

    monkeypatch.setattr(settings_tab, "save_settings", fake_save)
    output_dir = str(tmp_path / "audio")
    result = settings_tab.apply_settings(
        ctx,
        "1.7B", "bf16", 0.55, 35, 0.85, 1.1, 2048, 90,
        output_dir, True, False, "Chinese", "wav", 192, False, False, False, 4,
    )

    assert result[0].startswith("设置已应用")
    assert ctx.settings.temperature == 0.55
    assert ctx.settings.top_k == 35
    assert ctx.settings.default_language == "Chinese"
    assert ctx.settings.autosave is True
    assert ctx.engine.jit_compile is False
    assert captured["settings"] == ctx.settings
    assert Path(output_dir).is_dir()
    assert ctx.engine.load_calls == 0


def test_restored_model_preferences_do_not_eagerly_load_weights(monkeypatch):
    def fail_if_loaded(self, model_type):
        raise AssertionError(f"unexpected eager model load: {model_type}")

    monkeypatch.setattr(TTSEngine, "_load_model", fail_if_loaded)
    # The Codex runner has no Metal device; this test only exercises startup
    # configuration, so avoid invoking the real MLX shutdown primitives here.
    monkeypatch.setattr(TTSEngine, "_shutdown_on_owner", lambda self: None)
    engine = TTSEngine(model_size="0.6B", quantization="8bit", jit_compile=True)
    try:
        assert engine.model_size == "0.6B"
        assert engine.quantization == "8bit"
        assert engine.jit_compile is True
        assert engine.current_model is None
    finally:
        engine.shutdown()


def test_reset_then_apply_can_persist_defaults(monkeypatch):
    ctx = AppContext(
        engine=_FakeEngine(), library=None, history=None, yt=None,
        settings=AppSettings(temperature=0.55, top_k=35, autosave=True),
    )
    saved = {}
    monkeypatch.setattr(settings_tab, "save_settings", lambda settings: (saved.setdefault("value", settings), []))
    reset = settings_tab.reset_settings_defaults()

    # Gradio updates are dict-like in the installed Gradio version.
    values = [item["value"] for item in reset]
    settings_tab.apply_settings(
        ctx, *values[:2], *values[6:14], values[14], values[4], values[3],
        values[15], values[16], values[17], values[18], values[2],
    )

    assert saved["value"] == AppSettings()
    assert ctx.settings == AppSettings()


def test_settings_file_is_external_and_ignored():
    repo = Path(__file__).resolve().parents[1]
    assert not (repo / "config" / "settings.json").exists()
    assert "config/settings.json" in (repo / ".gitignore").read_text()


class _FakeEngine:
    model_size = "1.7B"
    quantization = "bf16"
    jit_compile = False
    current_model = None
    current_model_type = None
    load_calls = 0

    def get_repo_id(self, model_type):
        return f"fake/{model_type}"

    def unload_model(self):
        self.current_model_type = None
