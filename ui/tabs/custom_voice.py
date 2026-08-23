"""Custom Voice tab: preset-speaker generation with optional style instruction."""
import types

import gradio as gr

from config import DEFAULT_SPEAKERS
from generation import GenRequest, run_batch, run_single, save_audio
from ui import strings as S
from ui.components import (
    build_batch_accordion, build_output_column, wire_run_lifecycle, wire_stop,
)


def build(ctx):
    with gr.Tab(S.TAB_CUSTOM_VOICE):
        with gr.Row():
            with gr.Column(scale=2):
                cv_text = gr.Textbox(
                    label=S.TEXT_TO_SPEAK,
                    lines=5,
                    placeholder=S.TEXT_PLACEHOLDER,
                )
                with gr.Row():
                    cv_speaker = gr.Dropdown(
                        choices=DEFAULT_SPEAKERS,
                        value=DEFAULT_SPEAKERS[0],
                        label=S.CV_SPEAKER,
                    )
                    cv_language = gr.Dropdown(
                        choices=S.LANGUAGE_CHOICES,
                        value=S.LANGUAGE_AUTO_VALUE,
                        label=S.LANGUAGE,
                    )
                cv_instruct = gr.Textbox(
                    label=S.CV_INSTRUCT,
                    lines=1,
                    placeholder=S.CV_INSTRUCT_PLACEHOLDER,
                )
                gr.Markdown(S.TIP_TEXT_LENGTH, elem_classes=["text-hint"])
                cv_generate = gr.Button(S.GENERATE, variant="primary")
                batch = build_batch_accordion()
            out = build_output_column()
    return types.SimpleNamespace(
        cv_text=cv_text, cv_speaker=cv_speaker, cv_language=cv_language,
        cv_instruct=cv_instruct, cv_generate=cv_generate,
        cv_batch_split=batch.batch_split, cv_batch_silence=batch.batch_silence,
        cv_batch_generate=batch.batch_generate, cv_batch_table=batch.batch_table,
        cv_batch_audio=batch.batch_audio, cv_batch_save=batch.batch_save,
        cv_batch_status=batch.batch_status,
        cv_audio=out.audio, cv_stop=out.stop,
        cv_save=out.save, cv_save_status=out.save_status,
    )


def wire(ctx, ui):
    t = ui.cv

    def on_generate(text, speaker, language, instruct):
        yield from run_single(ctx, GenRequest(
            mode="custom_voice", text=text, speaker=speaker,
            language=language, instruct=instruct))

    def on_batch(text, speaker, language, instruct, split_mode, silence_ms,
                 progress=gr.Progress()):
        yield from run_batch(ctx, GenRequest(
            mode="custom_voice", text=text, speaker=speaker,
            language=language, instruct=instruct),
            split_mode, silence_ms, progress)

    wire_stop(ctx, t.cv_stop, ui.status)
    wire_run_lifecycle(
        t.cv_generate, t.cv_stop, on_generate,
        inputs=[t.cv_text, t.cv_speaker, t.cv_language, t.cv_instruct],
        outputs=[t.cv_audio, ui.status],
    )
    t.cv_save.click(
        fn=lambda audio: save_audio(ctx, audio, "custom"),
        inputs=[t.cv_audio],
        outputs=[t.cv_save_status],
    )
    wire_run_lifecycle(
        t.cv_batch_generate, t.cv_stop, on_batch,
        inputs=[t.cv_text, t.cv_speaker, t.cv_language, t.cv_instruct,
                t.cv_batch_split, t.cv_batch_silence],
        outputs=[t.cv_batch_audio, t.cv_batch_table, t.cv_batch_status],
        show_progress="full",
    )
    t.cv_batch_save.click(
        fn=lambda audio: save_audio(ctx, audio, "batch_custom"),
        inputs=[t.cv_batch_audio],
        outputs=[t.cv_batch_status],
    )
