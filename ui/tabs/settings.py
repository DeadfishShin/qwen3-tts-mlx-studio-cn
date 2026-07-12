"""Settings tab: model, generation, and output configuration."""
import os
import shutil
import types

import gradio as gr

from audio_utils import unload_deepfilter
from config import (
    LANGUAGE_AUTO,
    DEFAULT_AUTOSAVE, DEFAULT_BATCH_SIZE, DEFAULT_DENOISE_REF,
    DEFAULT_EXPORT_FORMAT, DEFAULT_LOUDNORM, DEFAULT_MAX_TOKENS,
    DEFAULT_MP3_BITRATE, DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_TRIM_SILENCE,
    ENABLE_JIT_COMPILE, HISTORY_DIR, LANGUAGES, MAX_BATCH_SIZE, MIN_BATCH_SIZE,
    OUTPUT_DIR, VOICE_LIBRARY_DIR, YT_CACHE_DIR,
)
from generation import get_hf_cache_dir
from ui import strings as S


GENERATION_PRESETS = {
    "Balanced": (0.9, 50, 1.0, 1.05, 4096, 120),
    "Creative": (1.2, 80, 0.95, 1.0, 4096, 120),
    "Precise": (0.3, 20, 0.9, 1.1, 4096, 120),
}


def get_model_status(ctx):
    if ctx.engine.current_model is None:
        return "No model loaded"
    repo = ctx.engine.get_repo_id(ctx.engine.current_model_type)
    return f"Loaded: {repo}"


def build(ctx):
    with gr.Tab("Settings"):
        with gr.Row():
            # --- Column 1: Model & Language ---
            with gr.Column(scale=1):
                gr.Markdown("### Model")
                set_size = gr.Radio(
                    ["0.6B", "1.7B"],
                    value=ctx.engine.model_size,
                    label="Model Size",
                )
                set_quant = gr.Radio(
                    ["4bit", "6bit", "8bit", "bf16"],
                    value=ctx.engine.quantization,
                    label="Quantization",
                )
                set_status = gr.Textbox(
                    label="Loaded Model",
                    value=get_model_status(ctx),
                    interactive=False,
                    elem_classes=["model-status"],
                )
                set_unload = gr.Button("Unload Model / Free RAM")
                gr.Markdown("### Reference Audio")
                set_denoise_ref = gr.Checkbox(
                    value=DEFAULT_DENOISE_REF,
                    label="Denoise reference audio (DeepFilterNet, 8MB model)",
                    info="Pre-processes voice clone references to remove background noise",
                )
                gr.Markdown("### Language")
                set_default_language = gr.Dropdown(
                    choices=[LANGUAGE_AUTO] + LANGUAGES,
                    value=LANGUAGE_AUTO,
                    label="Default Language",
                )
                set_jit = gr.Checkbox(
                    value=ENABLE_JIT_COMPILE,
                    label="JIT compile model (faster after first run; unloads model on change)",
                    container=False,
                )
                with gr.Accordion("Model Cache & ASR", open=False, elem_classes=["settings-accordion"]):
                    gr.Markdown("### Model Cache")
                    gr.Textbox(
                        label="HuggingFace Cache Directory",
                        value=os.path.abspath(get_hf_cache_dir()),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    set_delete_models = gr.Button(
                        "Delete Downloaded Models",
                        variant="stop",
                    )
                    set_delete_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Models will be re-downloaded on next use.",
                        elem_classes=["save-status-text"],
                    )
                    gr.Markdown("### Speech Recognition")
                    set_asr_status = gr.Textbox(
                        label="ASR Model",
                        value="Not loaded (loads on demand)",
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    set_asr_unload = gr.Button("Unload ASR Model")

            # --- Column 2: Generation ---
            with gr.Column(scale=1):
                gr.Markdown("### Generation")
                set_preset = gr.Radio(
                    ["Balanced", "Creative", "Precise", "Custom"],
                    value="Balanced",
                    label="Generation Presets",
                    info="Presets fill sliders below. Adjust freely afterward.",
                )
                set_temperature = gr.Slider(
                    0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.05,
                    label="Temperature",
                )
                set_top_k = gr.Slider(
                    0, 100, value=DEFAULT_TOP_K, step=1,
                    label="Top-K",
                )
                set_top_p = gr.Slider(
                    0.0, 1.0, value=DEFAULT_TOP_P, step=0.05,
                    label="Top-P",
                )
                set_rep_penalty = gr.Slider(
                    1.0, 2.0, value=DEFAULT_REPETITION_PENALTY, step=0.05,
                    label="Repetition Penalty",
                    info="Voice cloning always uses at least 1.5",
                )
                set_max_tokens = gr.Slider(
                    512, 8192, value=DEFAULT_MAX_TOKENS, step=256,
                    label="Max Tokens",
                )
                set_timeout = gr.Slider(
                    30, 300, value=DEFAULT_TIMEOUT, step=10,
                    label="Generation Timeout (seconds)",
                    info=S.TIMEOUT_SLIDER_INFO,
                )
                set_batch_size = gr.Slider(
                    MIN_BATCH_SIZE, MAX_BATCH_SIZE,
                    value=DEFAULT_BATCH_SIZE, step=1,
                    label="Batch Size",
                    info="Segments processed in parallel (Custom Voice & Voice Design batch/script modes)",
                )
                set_reset = gr.Button("Reset to Defaults")

            # --- Column 3: Output ---
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                set_output_dir = gr.Textbox(
                    value=OUTPUT_DIR,
                    label="Output Directory",
                )
                set_autosave = gr.Checkbox(
                    value=DEFAULT_AUTOSAVE,
                    label="Auto-save generated audio",
                )
                gr.Markdown("### Export Format")
                set_export_format = gr.Radio(
                    ["wav", "mp3", "ogg"],
                    value=DEFAULT_EXPORT_FORMAT,
                    label="Audio Format",
                )
                set_mp3_bitrate = gr.Slider(
                    64, 320, value=DEFAULT_MP3_BITRATE, step=32,
                    label="MP3 Bitrate (kbps)",
                    visible=False,
                )
                gr.Markdown("### Post-Processing")
                set_loudnorm = gr.Checkbox(
                    value=DEFAULT_LOUDNORM,
                    label="EBU R128 loudness normalization",
                )
                set_trim_silence = gr.Checkbox(
                    value=DEFAULT_TRIM_SILENCE,
                    label="Trim leading/trailing silence",
                )
                with gr.Accordion("Storage & Cache", open=False, elem_classes=["settings-accordion"]):
                    gr.Markdown("### YT Cache")
                    set_yt_cache_btn = gr.Button("Clear YT Cache")
                    set_yt_cache_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder=f"Cache: {YT_CACHE_DIR}/",
                        elem_classes=["save-status-text"],
                    )
                    gr.Markdown("### Storage Paths")
                    gr.Textbox(
                        label="Voice Library",
                        value=os.path.abspath(VOICE_LIBRARY_DIR),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    gr.Textbox(
                        label="History",
                        value=os.path.abspath(HISTORY_DIR),
                        interactive=False,
                        elem_classes=["model-status"],
                    )

        set_apply = gr.Button("Apply Settings", variant="primary")
    return types.SimpleNamespace(
        set_size=set_size, set_quant=set_quant, set_status=set_status,
        set_unload=set_unload, set_denoise_ref=set_denoise_ref,
        set_default_language=set_default_language, set_jit=set_jit,
        set_delete_models=set_delete_models, set_delete_status=set_delete_status,
        set_asr_status=set_asr_status, set_asr_unload=set_asr_unload,
        set_preset=set_preset, set_temperature=set_temperature,
        set_top_k=set_top_k, set_top_p=set_top_p, set_rep_penalty=set_rep_penalty,
        set_max_tokens=set_max_tokens, set_timeout=set_timeout,
        set_batch_size=set_batch_size, set_reset=set_reset,
        set_output_dir=set_output_dir, set_autosave=set_autosave,
        set_export_format=set_export_format, set_mp3_bitrate=set_mp3_bitrate,
        set_loudnorm=set_loudnorm, set_trim_silence=set_trim_silence,
        set_yt_cache_btn=set_yt_cache_btn, set_yt_cache_status=set_yt_cache_status,
        set_apply=set_apply,
    )


def apply_preset(preset_name):
    if preset_name == "Custom" or preset_name not in GENERATION_PRESETS:
        return [gr.update()] * 6
    temp, top_k, top_p, rep_pen, max_tok, timeout = GENERATION_PRESETS[preset_name]
    return (
        gr.update(value=temp),
        gr.update(value=top_k),
        gr.update(value=top_p),
        gr.update(value=rep_pen),
        gr.update(value=max_tok),
        gr.update(value=timeout),
    )


def reset_generation_defaults():
    return (
        gr.update(value=DEFAULT_TEMPERATURE),
        gr.update(value=DEFAULT_TOP_K),
        gr.update(value=DEFAULT_TOP_P),
        gr.update(value=DEFAULT_REPETITION_PENALTY),
        gr.update(value=DEFAULT_MAX_TOKENS),
        gr.update(value=DEFAULT_TIMEOUT),
        gr.update(value="Balanced"),
    )


def delete_cached_models(ctx):
    """Delete all Qwen3-TTS model files from the HuggingFace cache."""
    ctx.engine.unload_model()
    hf_cache = get_hf_cache_dir()
    if not os.path.isdir(hf_cache):
        return "HuggingFace cache directory not found", "No model loaded"
    deleted = []
    failed = []
    for entry in os.listdir(hf_cache):
        if entry.startswith("models--mlx-community--Qwen3-TTS-12Hz-"):
            path = os.path.join(hf_cache, entry)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    deleted.append(entry.replace("models--mlx-community--", ""))
                except OSError as e:
                    failed.append(f"{entry.replace('models--mlx-community--', '')}: {e}")
    parts = []
    if deleted:
        parts.append(f"Deleted {len(deleted)} model(s): {', '.join(deleted)}")
    if failed:
        parts.append(f"Failed to delete {len(failed)}: {'; '.join(failed)}")
    if parts:
        return " | ".join(parts), "No model loaded"
    return "No Qwen3-TTS models found in cache", "No model loaded"


def wire(ctx, ui):
    t = ui.settings

    def apply_settings(
        model_size, quantization,
        temperature, top_k, top_p, repetition_penalty, max_tokens, timeout,
        output_dir, autosave, jit_compile, default_language,
        export_format, mp3_bitrate, loudnorm, trim_silence, denoise_ref,
        batch_size,
    ):
        engine = ctx.engine
        model_changed = (
            model_size != engine.model_size
            or quantization != engine.quantization
            or jit_compile != engine.jit_compile
        )
        engine.model_size = model_size
        engine.quantization = quantization
        engine.jit_compile = jit_compile
        if model_changed:
            engine.unload_model()

        s = ctx.settings
        s.temperature = temperature
        s.top_k = int(top_k)
        s.top_p = top_p
        s.repetition_penalty = repetition_penalty
        s.max_tokens = int(max_tokens)
        s.timeout = int(timeout)
        s.output_dir = output_dir.strip() or OUTPUT_DIR
        s.autosave = autosave
        s.export_format = export_format
        s.mp3_bitrate = int(mp3_bitrate)
        s.loudnorm = loudnorm
        s.trim_silence = trim_silence
        s.denoise_ref = denoise_ref
        if not denoise_ref:
            unload_deepfilter()
        s.batch_size = int(batch_size)
        s.default_language = default_language

        os.makedirs(s.output_dir, exist_ok=True)

        parts = [f"size: {model_size}, quant: {quantization}"]
        if model_changed:
            parts.append("model unloaded")
        msg = f"Settings applied — {', '.join(parts)}."
        lang_update = gr.update(value=default_language)
        if default_language == LANGUAGE_AUTO:
            # ASR uses its own "Auto" convention; library import has no auto option.
            asr_update = gr.update(value="Auto")
            lib_update = gr.update()
        else:
            asr_update = lang_update
            lib_update = lang_update
        return msg, msg, lang_update, lang_update, lang_update, lang_update, asr_update, lib_update

    def unload_model():
        ctx.engine.unload_model()
        return "Model unloaded. RAM freed.", "Not loaded (loads on demand)"

    def unload_asr_setting():
        ctx.engine.unload_asr()
        return "ASR unloaded. RAM freed."

    def clear_yt_cache():
        n = ctx.yt.clear_cache()
        return f"YT cache cleared — {n} entr{'y' if n == 1 else 'ies'} removed"

    t.set_export_format.change(
        fn=lambda fmt: gr.update(visible=(fmt == "mp3")),
        inputs=[t.set_export_format],
        outputs=[t.set_mp3_bitrate],
    )
    t.set_apply.click(
        fn=apply_settings,
        inputs=[
            t.set_size, t.set_quant,
            t.set_temperature, t.set_top_k, t.set_top_p, t.set_rep_penalty,
            t.set_max_tokens, t.set_timeout,
            t.set_output_dir, t.set_autosave, t.set_jit, t.set_default_language,
            t.set_export_format, t.set_mp3_bitrate, t.set_loudnorm, t.set_trim_silence,
            t.set_denoise_ref,
            t.set_batch_size,
        ],
        outputs=[
            t.set_status, ui.status,
            ui.cv.cv_language, ui.vd.vd_language, ui.vc.vc_language,
            ui.yt.yt_language, ui.asr.asr_language, ui.lib.lib_import_language,
        ],
    )
    t.set_preset.change(
        fn=apply_preset,
        inputs=[t.set_preset],
        outputs=[t.set_temperature, t.set_top_k, t.set_top_p, t.set_rep_penalty,
                 t.set_max_tokens, t.set_timeout],
    )
    t.set_reset.click(
        fn=reset_generation_defaults,
        outputs=[t.set_temperature, t.set_top_k, t.set_top_p, t.set_rep_penalty,
                 t.set_max_tokens, t.set_timeout, t.set_preset],
    )
    t.set_unload.click(
        fn=unload_model,
        outputs=[t.set_status, t.set_asr_status],
    )
    t.set_asr_unload.click(
        fn=unload_asr_setting,
        outputs=[t.set_asr_status],
    )
    t.set_delete_models.click(
        fn=lambda: delete_cached_models(ctx),
        outputs=[t.set_delete_status, t.set_status],
    )
    t.set_yt_cache_btn.click(fn=clear_yt_cache, outputs=[t.set_yt_cache_status])
