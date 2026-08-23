import types

import gradio as gr
import numpy as np

import generation
from generation import GenRequest, compose_voice_design_instruct, run_single
from state import AppContext, AppSettings
from ui import strings as S
from ui.tabs.voice_design import build, save_design_to_library


def test_language_display_labels_keep_canonical_values():
    choices = dict(S.LANGUAGE_CHOICES)
    assert choices["自动检测"] == "Auto-detect"
    assert choices["中文"] == "Chinese"
    assert choices["英语"] == "English"
    assert generation.api_language(choices["自动检测"]) == "auto"
    assert generation.api_language(choices["中文"]) == "Chinese"


def test_voice_design_description_only_prompt_is_compatible():
    description = "低沉、沉稳的中文男声旁白"
    assert compose_voice_design_instruct(description, "") == description
    assert compose_voice_design_instruct(description, "   ") == description


def test_voice_design_description_and_style_prompt_composition():
    result = compose_voice_design_instruct(
        "低沉、沉稳的中文男声旁白",
        "语速稍慢，从容沉稳，逻辑转折处自然停顿，不要急促。",
    )
    assert result == (
        "声音身份：\n低沉、沉稳的中文男声旁白\n\n"
        "本次说话风格：\n语速稍慢，从容沉稳，逻辑转折处自然停顿，不要急促。"
    )


def test_voice_design_engine_receives_composed_prompt_and_history_keeps_fields(
    fake_engine, fake_history, tmp_path, monkeypatch
):
    captured = {}

    def stream_voice_design(text, language, instruct, **kwargs):
        captured["instruct"] = instruct
        yield from fake_engine._stream("voice_design", "voice_design")

    fake_engine.stream_generate_voice_design = stream_voice_design
    monkeypatch.setattr(generation.gr, "Warning", lambda *args, **kwargs: None)
    settings = AppSettings()
    settings.output_dir = str(tmp_path / "out")
    ctx = AppContext(
        engine=fake_engine, library=None, history=fake_history,
        yt=None, settings=settings,
    )
    description = "温和、清晰的中文女声"
    style = "语速稍慢、自然停顿"
    list(run_single(ctx, GenRequest(
        mode="voice_design", text="你好。", language="Chinese",
        instruct=description, voice_description=description,
        style_instruction=style,
    )))

    assert captured["instruct"] == compose_voice_design_instruct(description, style)
    assert fake_history.entries[0]["voice_description"] == description
    assert fake_history.entries[0]["style_instruction"] == style
    assert fake_history.entries[0]["voice_params"] == description


def test_voice_design_library_saves_stable_description_not_transient_style(tmp_path, monkeypatch):
    saved = {}

    class Library:
        def save_voice(self, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr("ui.tabs.voice_design.OUTPUT_DIR", str(tmp_path))
    ctx = types.SimpleNamespace(library=Library())
    audio = (24000, np.zeros(2400, dtype=np.float32))
    description = "稳定、亲切的中文播音员音色"
    result = save_design_to_library(
        ctx, audio, "播音员", "Chinese", description, "测试文本"
    )

    assert result == S.VD_SAVED.format(name="播音员")
    assert saved["description"] == description
    assert "风格指令" not in saved["description"]
    assert saved["ref_text"] == "测试文本"


def test_voice_design_ui_exposes_chinese_description_and_style_fields(fake_engine, fake_history):
    class Library:
        def list_voices(self):
            return []

    ctx = AppContext(
        engine=fake_engine, library=Library(), history=fake_history,
        yt=None, settings=AppSettings(),
    )
    with gr.Blocks():
        ui = build(ctx)

    assert ui.vd_voice_description.label == S.VD_DESCRIPTION
    assert ui.vd_style_instruction.label == S.VD_STYLE
    assert dict(ui.vd_language.choices)["中文"] == "Chinese"
