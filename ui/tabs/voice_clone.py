"""Voice Cloning tab: clone a voice from reference audio + transcript."""
import types

import gradio as gr

from config import LANGUAGES
from generation import GenRequest, run_batch, run_single, save_audio, stream_transcription
from ui import strings as S
from ui.components import (
    build_batch_accordion, build_lib_save_accordion, build_output_column,
    voice_choices, wire_run_lifecycle, wire_stop,
)


def build(ctx):
    with gr.Tab(S.TAB_VOICE_CLONE):
        gr.HTML(S.VC_NOTICE_HTML)
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    vc_language = gr.Dropdown(
                        choices=[S.LANGUAGE_AUTO] + LANGUAGES,
                        value=S.LANGUAGE_AUTO, label=S.LANGUAGE
                    )
                    vc_library_voice = gr.Dropdown(
                        choices=voice_choices(ctx),
                        value="None",
                        label=S.VC_LIBRARY_VOICE,
                    )
                vc_ref_audio = gr.Audio(
                    label=S.VC_REF_AUDIO,
                    type="filepath",
                    sources=["upload", "microphone"],
                    buttons=["download"],
                )
                vc_trim_ref = gr.Checkbox(value=True, label=S.TRIM_REF_LABEL)
                with gr.Row():
                    vc_transcribe_btn = gr.Button(S.VC_TRANSCRIBE, variant="secondary", scale=1)
                gr.HTML(S.VC_TRANSCRIBE_HINT_HTML)
                vc_ref_text = gr.Textbox(
                    label=S.VC_REF_TEXT,
                    lines=2,
                    placeholder=S.VC_REF_TEXT_PLACEHOLDER,
                )
                vc_text = gr.Textbox(
                    label=S.VC_TEXT_LABEL,
                    lines=5,
                    placeholder=S.VC_TEXT_PLACEHOLDER,
                )
                vc_generate = gr.Button(S.GENERATE, variant="primary")
                lib = build_lib_save_accordion(S.VC_LIB_NAME_PLACEHOLDER)
                batch = build_batch_accordion()
            out = build_output_column()
    return types.SimpleNamespace(
        vc_language=vc_language, vc_library_voice=vc_library_voice,
        vc_trim_ref=vc_trim_ref,
        vc_ref_audio=vc_ref_audio, vc_transcribe_btn=vc_transcribe_btn,
        vc_ref_text=vc_ref_text, vc_text=vc_text, vc_generate=vc_generate,
        vc_lib_name=lib.lib_name, vc_lib_save=lib.lib_save, vc_lib_status=lib.lib_status,
        vc_batch_split=batch.batch_split, vc_batch_silence=batch.batch_silence,
        vc_batch_generate=batch.batch_generate, vc_batch_table=batch.batch_table,
        vc_batch_audio=batch.batch_audio, vc_batch_save=batch.batch_save,
        vc_batch_status=batch.batch_status,
        vc_audio=out.audio, vc_stop=out.stop,
        vc_save=out.save, vc_save_status=out.save_status,
    )


def transcribe_reference(ctx, ref_audio):
    """Transcribe reference audio and fill the transcript box (streams live)."""
    if not ref_audio:
        gr.Warning("Upload reference audio first.")
        yield gr.update(), "No audio to transcribe"
        return
    yield from stream_transcription(ctx, ref_audio, "auto")


def save_clone_to_library(ctx, ref_audio, ref_text, name, language):
    if not ref_audio:
        gr.Warning("No reference audio to save.")
        return "No reference audio"
    if not name.strip():
        gr.Warning("Please enter a name for this voice.")
        return "Enter a voice name"
    if not ref_text or not ref_text.strip():
        gr.Warning("Reference transcript is required.")
        return "Enter transcript"
    ctx.library.save_voice(
        name=name,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
        language=language,
        source="clone",
    )
    return f"Voice '{name}' saved to library"


def wire(ctx, ui):
    t = ui.vc

    def on_generate(text, ref_audio, ref_text, language, library_voice, trim_ref):
        yield from run_single(ctx, GenRequest(
            mode="voice_clone", text=text, language=language,
            ref_audio=ref_audio, ref_text=ref_text, library_voice=library_voice,
            trim_ref=trim_ref))

    def on_batch(text, ref_audio, ref_text, language, library_voice, trim_ref,
                 split_mode, silence_ms, progress=gr.Progress()):
        yield from run_batch(ctx, GenRequest(
            mode="voice_clone", text=text, language=language,
            ref_audio=ref_audio, ref_text=ref_text, library_voice=library_voice,
            trim_ref=trim_ref),
            split_mode, silence_ms, progress)

    def save_and_refresh(ref_audio, ref_text, name, language):
        result = save_clone_to_library(ctx, ref_audio, ref_text, name, language)
        return result, gr.update(choices=voice_choices(ctx))

    def refresh_clone_library():
        return gr.update(choices=voice_choices(ctx), value="None")

    def on_transcribe(ref_audio):
        yield from transcribe_reference(ctx, ref_audio)

    wire_stop(ctx, t.vc_stop, ui.status)
    wire_run_lifecycle(
        t.vc_transcribe_btn, t.vc_stop, on_transcribe,
        inputs=[t.vc_ref_audio],
        outputs=[t.vc_ref_text, ui.status],
    )
    wire_run_lifecycle(
        t.vc_generate, t.vc_stop, on_generate,
        inputs=[t.vc_text, t.vc_ref_audio, t.vc_ref_text, t.vc_language, t.vc_library_voice,
                t.vc_trim_ref],
        outputs=[t.vc_audio, ui.status],
    )
    t.vc_save.click(
        fn=lambda audio: save_audio(ctx, audio, "clone"),
        inputs=[t.vc_audio],
        outputs=[t.vc_save_status],
    )
    wire_run_lifecycle(
        t.vc_batch_generate, t.vc_stop, on_batch,
        inputs=[t.vc_text, t.vc_ref_audio, t.vc_ref_text, t.vc_language, t.vc_library_voice,
                t.vc_trim_ref, t.vc_batch_split, t.vc_batch_silence],
        outputs=[t.vc_batch_audio, t.vc_batch_table, t.vc_batch_status],
        show_progress="full",
    )
    t.vc_batch_save.click(
        fn=lambda audio: save_audio(ctx, audio, "batch_clone"),
        inputs=[t.vc_batch_audio],
        outputs=[t.vc_batch_status],
    )
    t.vc_lib_save.click(
        fn=save_and_refresh,
        inputs=[t.vc_ref_audio, t.vc_ref_text, t.vc_lib_name, t.vc_language],
        outputs=[t.vc_lib_status, t.vc_library_voice],
    )
    t.vc_library_voice.focus(
        fn=refresh_clone_library,
        outputs=[t.vc_library_voice],
    )
