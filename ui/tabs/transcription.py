"""Transcription tab: local ASR via Qwen3-ASR."""
import os
import types
from datetime import datetime

import gradio as gr

from config import LANGUAGES
from ui import strings as S


def build(ctx):
    with gr.Tab(S.TAB_TRANSCRIPTION):
        gr.HTML(
            "<div class='info-notice'>"
            "Transcribe audio files locally using Qwen3-ASR. "
            "Supports up to ~20 minutes of audio."
            "</div>"
        )
        with gr.Row():
            with gr.Column(scale=2):
                asr_audio = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                with gr.Row():
                    asr_language = gr.Dropdown(
                        choices=["Auto"] + LANGUAGES,
                        value="Auto",
                        label=S.LANGUAGE,
                    )
                asr_transcribe_btn = gr.Button("Transcribe", variant="primary")
                asr_output = gr.Textbox(
                    label="Transcription",
                    lines=12,
                )
                with gr.Row():
                    asr_save_btn = gr.Button("Save as .txt")
                    asr_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Save path appears here...",
                        elem_classes=["save-status-text"],
                    )
            with gr.Column(scale=1, elem_classes=["output-col"]):
                asr_info = gr.Markdown(
                    "**Model:** Qwen3-ASR-1.7B-8bit\n\n"
                    "**Supported languages:** Auto-detect, English, Chinese, Japanese, Korean, "
                    "German, French, Russian, Portuguese, Spanish, Italian\n\n"
                    "**Max duration:** ~20 minutes per file"
                )
    return types.SimpleNamespace(
        asr_audio=asr_audio, asr_language=asr_language,
        asr_transcribe_btn=asr_transcribe_btn, asr_output=asr_output,
        asr_save_btn=asr_save_btn, asr_save_status=asr_save_status,
        asr_info=asr_info,
    )


def transcribe_audio(ctx, audio_path, language):
    """Standalone transcription handler."""
    if not audio_path:
        gr.Warning("Upload or record audio first.")
        return gr.update(), "No audio"
    lang = "auto" if language == "Auto" else language
    yield gr.update(), "Loading ASR model..."
    try:
        text = ctx.engine.transcribe(audio_path, language=lang)
        if not text or not text.strip():
            yield gr.update(), "Transcription returned empty"
            return
        yield gr.update(value=text.strip()), f"Transcribed ({len(text.strip().split())} words)"
    except Exception as e:
        gr.Warning(f"Transcription failed: {e}")
        yield gr.update(), f"Error: {e}"


def save_transcript(ctx, text):
    """Save transcription text to .txt file."""
    if not text or not text.strip():
        gr.Warning("No transcription to save.")
        return "Nothing to save"
    out_dir = ctx.settings.output_dir
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"transcript_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved: {path}"


def wire(ctx, ui):
    t = ui.asr

    def on_transcribe(audio_path, language):
        yield from transcribe_audio(ctx, audio_path, language)

    t.asr_transcribe_btn.click(
        fn=on_transcribe,
        inputs=[t.asr_audio, t.asr_language],
        outputs=[t.asr_output, ui.status],
        show_progress="minimal",
    )
    t.asr_save_btn.click(
        fn=lambda text: save_transcript(ctx, text),
        inputs=[t.asr_output],
        outputs=[t.asr_save_status],
    )
