"""Voice Design tab: generate speech from a natural-language voice description."""
import os
import time
import types

import gradio as gr
import soundfile as sf

from config import LANGUAGES, OUTPUT_DIR
from generation import GenRequest, run_batch, run_single, save_audio
from ui import strings as S
from ui.components import (
    build_batch_accordion, build_lib_save_accordion, build_output_column,
    voice_choices,
)


def build(ctx):
    with gr.Tab(S.TAB_VOICE_DESIGN):
        with gr.Row():
            with gr.Column(scale=2):
                vd_text = gr.Textbox(
                    label=S.VD_TEXT_LABEL,
                    lines=5,
                    placeholder=S.TEXT_PLACEHOLDER,
                )
                vd_instruct = gr.Textbox(
                    label=S.VD_INSTRUCT,
                    lines=2,
                    placeholder=S.VD_INSTRUCT_PLACEHOLDER,
                )
                vd_language = gr.Dropdown(
                    choices=LANGUAGES, value="English", label=S.LANGUAGE
                )
                gr.Markdown(S.TIP_TEXT_LENGTH, elem_classes=["text-hint"])
                vd_generate = gr.Button(S.GENERATE, variant="primary")
                lib = build_lib_save_accordion(S.VD_LIB_NAME_PLACEHOLDER)
                batch = build_batch_accordion()
            out = build_output_column()
    return types.SimpleNamespace(
        vd_text=vd_text, vd_instruct=vd_instruct, vd_language=vd_language,
        vd_generate=vd_generate,
        vd_lib_name=lib.lib_name, vd_lib_save=lib.lib_save, vd_lib_status=lib.lib_status,
        vd_batch_split=batch.batch_split, vd_batch_silence=batch.batch_silence,
        vd_batch_generate=batch.batch_generate, vd_batch_table=batch.batch_table,
        vd_batch_audio=batch.batch_audio, vd_batch_save=batch.batch_save,
        vd_batch_status=batch.batch_status,
        vd_audio=out.audio, vd_save=out.save, vd_save_status=out.save_status,
    )


def save_design_to_library(ctx, audio_tuple, name, language, description, spoken_text):
    if audio_tuple is None:
        gr.Warning("Generate audio first before saving to library.")
        return "No audio to save"
    if not name.strip():
        gr.Warning("Please enter a name for this voice.")
        return "Enter a voice name"
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
    return f"Voice '{name}' saved to library"


def wire(ctx, ui):
    t = ui.vd

    def on_generate(text, language, instruct):
        yield from run_single(ctx, GenRequest(
            mode="voice_design", text=text, language=language, instruct=instruct))

    def on_batch(text, language, instruct, split_mode, silence_ms,
                 progress=gr.Progress()):
        return run_batch(ctx, GenRequest(
            mode="voice_design", text=text, language=language, instruct=instruct),
            split_mode, silence_ms, progress)

    def save_and_refresh(audio_tuple, name, language, description, spoken_text):
        result = save_design_to_library(ctx, audio_tuple, name, language,
                                        description, spoken_text)
        return result, gr.update(choices=voice_choices(ctx))

    t.vd_generate.click(
        fn=on_generate,
        inputs=[t.vd_text, t.vd_language, t.vd_instruct],
        outputs=[t.vd_audio, ui.status],
        show_progress="minimal",
    )
    t.vd_save.click(
        fn=lambda audio: save_audio(ctx, audio, "design"),
        inputs=[t.vd_audio],
        outputs=[t.vd_save_status],
    )
    t.vd_batch_generate.click(
        fn=on_batch,
        inputs=[t.vd_text, t.vd_language, t.vd_instruct,
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
        inputs=[t.vd_audio, t.vd_lib_name, t.vd_language, t.vd_instruct, t.vd_text],
        outputs=[t.vd_lib_status, ui.vc.vc_library_voice],
    )
