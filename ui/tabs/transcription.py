"""Transcription tab: local ASR via Qwen3-ASR."""
import os
import types
from datetime import datetime

import gradio as gr

from config import LANGUAGES
from generation import api_language, stream_transcription
from ui import strings as S
from ui.components import wire_run_lifecycle, wire_stop


def build(ctx):
    with gr.Tab(S.TAB_TRANSCRIPTION):
        gr.HTML(S.ASR_NOTICE_HTML)
        with gr.Row():
            with gr.Column(scale=2):
                asr_audio = gr.Audio(
                    label=S.ASR_AUDIO,
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                with gr.Row():
                    asr_language = gr.Dropdown(
                        choices=[S.LANGUAGE_AUTO] + LANGUAGES,
                        value=S.LANGUAGE_AUTO,
                        label=S.LANGUAGE,
                    )
                with gr.Row():
                    asr_transcribe_btn = gr.Button(S.ASR_TRANSCRIBE, variant="primary")
                    asr_stop_btn = gr.Button(S.STOP, variant="stop", visible=False)
                asr_output = gr.Textbox(
                    label=S.ASR_OUTPUT,
                    lines=12,
                )
                with gr.Row():
                    asr_save_btn = gr.Button(S.ASR_SAVE_TXT)
                    asr_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder=S.ASR_SAVE_PLACEHOLDER,
                        elem_classes=["save-status-text"],
                    )
            with gr.Column(scale=1, elem_classes=["output-col"]):
                asr_info = gr.Markdown(S.ASR_INFO_MD)
    return types.SimpleNamespace(
        asr_audio=asr_audio, asr_language=asr_language,
        asr_transcribe_btn=asr_transcribe_btn, asr_stop_btn=asr_stop_btn,
        asr_output=asr_output,
        asr_save_btn=asr_save_btn, asr_save_status=asr_save_status,
        asr_info=asr_info,
    )


def transcribe_audio(ctx, audio_path, language):
    """Standalone transcription handler — streams tokens into the textbox."""
    if not audio_path:
        gr.Warning(S.ASR_NO_AUDIO_WARN)
        yield gr.update(), S.ASR_NO_AUDIO
        return
    yield from stream_transcription(ctx, audio_path, api_language(language))


def save_transcript(ctx, text):
    """Save transcription text to .txt file."""
    if not text or not text.strip():
        gr.Warning(S.ASR_NOTHING_TO_SAVE_WARN)
        return S.ASR_NOTHING_TO_SAVE
    out_dir = ctx.settings.output_dir
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"transcript_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return S.ASR_SAVED.format(path=path)


def wire(ctx, ui):
    t = ui.asr

    def on_transcribe(audio_path, language):
        yield from transcribe_audio(ctx, audio_path, language)

    wire_stop(ctx, t.asr_stop_btn, ui.status)
    wire_run_lifecycle(
        t.asr_transcribe_btn, t.asr_stop_btn, on_transcribe,
        inputs=[t.asr_audio, t.asr_language],
        outputs=[t.asr_output, ui.status],
    )
    t.asr_save_btn.click(
        fn=lambda text: save_transcript(ctx, text),
        inputs=[t.asr_output],
        outputs=[t.asr_save_status],
    )
