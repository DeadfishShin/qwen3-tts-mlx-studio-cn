"""Voice Design tab: generate speech from a natural-language voice description."""
import os
import time
import types

import gradio as gr
import soundfile as sf

from config import OUTPUT_DIR
from generation import GenRequest, run_batch, run_single, save_audio
from ui import strings as S
from ui.components import (
    build_batch_accordion, build_lib_save_accordion, build_output_column,
    voice_choices, wire_run_lifecycle, wire_stop,
)


def build(ctx):
    with gr.Tab(S.TAB_VOICE_DESIGN):
        with gr.Row():
            with gr.Column(scale=2):
                vd_text = gr.Textbox(
                    label=S.TEXT_TO_SPEAK,
                    lines=5,
                    placeholder=S.TEXT_PLACEHOLDER,
                )
                vd_description = gr.Textbox(
                    label=S.VD_DESCRIPTION,
                    lines=2,
                    placeholder=S.VD_DESCRIPTION_PLACEHOLDER,
                )
                gr.Markdown(S.VD_INFO, elem_classes=["text-hint"])
                vd_style_instruction = gr.Textbox(
                    label=S.VD_STYLE,
                    lines=2,
                    placeholder=S.VD_STYLE_PLACEHOLDER,
                )
                vd_language = gr.Dropdown(
                    choices=S.LANGUAGE_CHOICES,
                    value=S.LANGUAGE_AUTO_VALUE, label=S.LANGUAGE
                )
                gr.Markdown(S.TIP_TEXT_LENGTH, elem_classes=["text-hint"])
                vd_generate = gr.Button(S.GENERATE, variant="primary")
                lib = build_lib_save_accordion(S.VD_LIB_NAME_PLACEHOLDER)
                batch = build_batch_accordion()
            out = build_output_column()
    return types.SimpleNamespace(
        vd_text=vd_text, vd_voice_description=vd_description,
        vd_instruct=vd_description, vd_style_instruction=vd_style_instruction,
        vd_language=vd_language,
        vd_generate=vd_generate,
        vd_lib_name=lib.lib_name, vd_lib_save=lib.lib_save, vd_lib_status=lib.lib_status,
        vd_batch_split=batch.batch_split, vd_batch_silence=batch.batch_silence,
        vd_batch_generate=batch.batch_generate, vd_batch_table=batch.batch_table,
        vd_batch_audio=batch.batch_audio, vd_batch_save=batch.batch_save,
        vd_batch_status=batch.batch_status,
        vd_audio=out.audio, vd_stop=out.stop,
        vd_save=out.save, vd_save_status=out.save_status,
    )


def save_design_to_library(ctx, audio_tuple, name, language, description, spoken_text):
    if audio_tuple is None:
        gr.Warning(S.VD_SAVE_NO_AUDIO_WARN)
        return S.VD_SAVE_NO_AUDIO
    if not name.strip():
        gr.Warning(S.VD_SAVE_NO_NAME_WARN)
        return S.VD_SAVE_NO_NAME
    sr, audio = audio_tuple
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = os.path.join(OUTPUT_DIR, f"_tmp_design_{int(time.time())}.wav")
    sf.write(tmp_path, audio, sr)
    ctx.library.save_voice(
        name=name,
        ref_audio_path=tmp_path,
        ref_text=spoken_text,
        language=language,
        description=description,
        source="design",
    )
    os.remove(tmp_path)
    return S.VD_SAVED.format(name=name)


def wire(ctx, ui):
    t = ui.vd

    def on_generate(text, language, voice_description, style_instruction):
        yield from run_single(ctx, GenRequest(
            mode="voice_design", text=text, language=language,
            instruct=voice_description, voice_description=voice_description,
            style_instruction=style_instruction))

    def on_batch(text, language, voice_description, style_instruction,
                 split_mode, silence_ms,
                 progress=gr.Progress()):
        yield from run_batch(ctx, GenRequest(
            mode="voice_design", text=text, language=language,
            instruct=voice_description, voice_description=voice_description,
            style_instruction=style_instruction),
            split_mode, silence_ms, progress)

    def save_and_refresh(audio_tuple, name, language, description, spoken_text):
        result = save_design_to_library(ctx, audio_tuple, name, language,
                                        description, spoken_text)
        return result, gr.update(choices=voice_choices(ctx))

    wire_stop(ctx, t.vd_stop, ui.status)
    wire_run_lifecycle(
        t.vd_generate, t.vd_stop, on_generate,
        inputs=[t.vd_text, t.vd_language, t.vd_voice_description,
                t.vd_style_instruction],
        outputs=[t.vd_audio, ui.status],
    )
    t.vd_save.click(
        fn=lambda audio: save_audio(ctx, audio, "design"),
        inputs=[t.vd_audio],
        outputs=[t.vd_save_status],
    )
    wire_run_lifecycle(
        t.vd_batch_generate, t.vd_stop, on_batch,
        inputs=[t.vd_text, t.vd_language, t.vd_voice_description,
                t.vd_style_instruction,
                t.vd_batch_split, t.vd_batch_silence],
        outputs=[t.vd_batch_audio, t.vd_batch_table, t.vd_batch_status],
        show_progress="full",
    )
    t.vd_batch_save.click(
        fn=lambda audio: save_audio(ctx, audio, "batch_design"),
        inputs=[t.vd_batch_audio],
        outputs=[t.vd_batch_status],
    )
    t.vd_lib_save.click(
        fn=save_and_refresh,
        inputs=[t.vd_audio, t.vd_lib_name, t.vd_language,
                t.vd_voice_description, t.vd_text],
        outputs=[t.vd_lib_status, ui.vc.vc_library_voice],
    )
