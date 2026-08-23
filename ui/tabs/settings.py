"""Settings tab: model, generation, and output configuration."""
import os
import shutil
import types

import gradio as gr

from config import (
    LANGUAGE_AUTO,
    DEFAULT_AUTOSAVE, DEFAULT_BATCH_SIZE, DEFAULT_DENOISE_REF,
    DEFAULT_EXPORT_FORMAT, DEFAULT_LOUDNORM, DEFAULT_MAX_TOKENS,
    DEFAULT_MP3_BITRATE, DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_TRIM_SILENCE,
    ENABLE_JIT_COMPILE, HISTORY_DIR, MAX_BATCH_SIZE, MIN_BATCH_SIZE,
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
    if hasattr(ctx.engine, "get_model_state"):
        model_type, _, _, _ = ctx.engine.get_model_state()
    else:
        model_type = ctx.engine.current_model_type if ctx.engine.current_model is not None else None
    if model_type is None:
        return S.SET_NO_MODEL
    repo = ctx.engine.get_repo_id(model_type)
    return S.SET_MODEL_LOADED.format(repo=repo)


def build(ctx):
    if hasattr(ctx.engine, "get_model_state"):
        _, engine_model_size, engine_quantization, engine_jit = ctx.engine.get_model_state()
    else:
        engine_model_size = ctx.engine.model_size
        engine_quantization = ctx.engine.quantization
        engine_jit = ENABLE_JIT_COMPILE
    with gr.Tab(S.TAB_SETTINGS):
        with gr.Row():
            # --- Column 1: Model & Language ---
            with gr.Column(scale=1):
                gr.Markdown(S.SET_MODEL_HEADER)
                set_size = gr.Radio(
                    S.SET_MODEL_SIZE_CHOICES,
                    value=engine_model_size,
                    label=S.SET_MODEL_SIZE,
                )
                set_quant = gr.Radio(
                    S.SET_QUANT_CHOICES,
                    value=engine_quantization,
                    label=S.SET_QUANT,
                    info=S.SET_QUANT_INFO,
                )
                set_status = gr.Textbox(
                    label=S.SET_LOADED_MODEL,
                    value=get_model_status(ctx),
                    interactive=False,
                    elem_classes=["model-status"],
                )
                set_unload = gr.Button(S.SET_UNLOAD)
                gr.Markdown(S.SET_REF_HEADER)
                set_denoise_ref = gr.Checkbox(
                    value=DEFAULT_DENOISE_REF,
                    label=S.SET_DENOISE,
                    info=S.SET_DENOISE_INFO,
                )
                gr.Markdown(S.SET_LANGUAGE_HEADER)
                set_default_language = gr.Dropdown(
                    choices=S.LANGUAGE_CHOICES,
                    value=LANGUAGE_AUTO,
                    label=S.SET_DEFAULT_LANGUAGE,
                )
                set_jit = gr.Checkbox(
                    value=engine_jit,
                    label=S.SET_JIT,
                    info=S.SET_JIT_INFO,
                )
                with gr.Accordion(S.SET_CACHE_ACCORDION, open=False, elem_classes=["settings-accordion"]):
                    gr.Markdown(S.SET_CACHE_HEADER)
                    gr.Textbox(
                        label=S.SET_CACHE_DIR,
                        value=os.path.abspath(get_hf_cache_dir()),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    set_delete_models = gr.Button(
                        S.SET_DELETE_MODELS,
                        variant="stop",
                    )
                    set_delete_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder=S.SET_DELETE_PLACEHOLDER,
                        elem_classes=["save-status-text"],
                    )
                    gr.Markdown(S.SET_ASR_HEADER)
                    set_asr_status = gr.Textbox(
                        label=S.SET_ASR_STATUS,
                        value=S.SET_ASR_NOT_LOADED,
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    set_asr_unload = gr.Button(S.SET_ASR_UNLOAD)

            # --- Column 2: Generation ---
            with gr.Column(scale=1):
                gr.Markdown(S.SET_GENERATION_HEADER)
                set_preset = gr.Radio(
                    S.SET_PRESET_CHOICES,
                    value="Balanced",
                    label=S.SET_PRESET,
                    info=S.SET_PRESET_INFO,
                )
                set_temperature = gr.Slider(
                    0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.05,
                    label=S.SET_TEMP,
                    info=S.SET_TEMP_INFO,
                )
                set_top_k = gr.Slider(
                    0, 100, value=DEFAULT_TOP_K, step=1,
                    label=S.SET_TOP_K,
                    info=S.SET_TOP_K_INFO,
                )
                set_top_p = gr.Slider(
                    0.0, 1.0, value=DEFAULT_TOP_P, step=0.05,
                    label=S.SET_TOP_P,
                    info=S.SET_TOP_P_INFO,
                )
                set_rep_penalty = gr.Slider(
                    1.0, 2.0, value=DEFAULT_REPETITION_PENALTY, step=0.05,
                    label=S.SET_REP_PENALTY,
                    info=S.SET_REP_PENALTY_INFO,
                )
                set_max_tokens = gr.Slider(
                    512, 8192, value=DEFAULT_MAX_TOKENS, step=256,
                    label=S.SET_MAX_TOKENS,
                    info=S.SET_MAX_TOKENS_INFO,
                )
                set_timeout = gr.Slider(
                    30, 300, value=DEFAULT_TIMEOUT, step=10,
                    label=S.SET_TIMEOUT,
                    info=S.TIMEOUT_SLIDER_INFO,
                )
                set_batch_size = gr.Slider(
                    MIN_BATCH_SIZE, MAX_BATCH_SIZE,
                    value=DEFAULT_BATCH_SIZE, step=1,
                    label=S.SET_BATCH_SIZE,
                    info=S.SET_BATCH_SIZE_INFO,
                )
                set_reset = gr.Button(S.SET_RESET)

            # --- Column 3: Output ---
            with gr.Column(scale=1):
                gr.Markdown(S.SET_OUTPUT_HEADER)
                set_output_dir = gr.Textbox(
                    value=OUTPUT_DIR,
                    label=S.SET_OUTPUT_DIR,
                )
                set_autosave = gr.Checkbox(
                    value=DEFAULT_AUTOSAVE,
                    label=S.SET_AUTOSAVE,
                )
                gr.Markdown(S.SET_EXPORT_HEADER)
                set_export_format = gr.Radio(
                    S.SET_EXPORT_FORMAT_CHOICES,
                    value=DEFAULT_EXPORT_FORMAT,
                    label=S.SET_EXPORT_FORMAT,
                )
                set_mp3_bitrate = gr.Slider(
                    64, 320, value=DEFAULT_MP3_BITRATE, step=32,
                    label=S.SET_MP3_BITRATE,
                    visible=False,
                )
                gr.Markdown(S.SET_POST_HEADER)
                set_loudnorm = gr.Checkbox(
                    value=DEFAULT_LOUDNORM,
                    label=S.SET_LOUDNORM,
                    info=S.SET_LOUDNORM_INFO,
                )
                set_trim_silence = gr.Checkbox(
                    value=DEFAULT_TRIM_SILENCE,
                    label=S.SET_TRIM_SILENCE,
                )
                with gr.Accordion(S.SET_STORAGE_ACCORDION, open=False, elem_classes=["settings-accordion"]):
                    gr.Markdown(S.SET_YT_CACHE_HEADER)
                    set_yt_cache_btn = gr.Button(S.SET_YT_CACHE_CLEAR)
                    set_yt_cache_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder=S.SET_YT_CACHE_PLACEHOLDER.format(cache_dir=YT_CACHE_DIR),
                        elem_classes=["save-status-text"],
                    )
                    gr.Markdown(S.SET_STORAGE_HEADER)
                    gr.Textbox(
                        label=S.SET_STORAGE_LIBRARY,
                        value=os.path.abspath(VOICE_LIBRARY_DIR),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    gr.Textbox(
                        label=S.SET_STORAGE_HISTORY,
                        value=os.path.abspath(HISTORY_DIR),
                        interactive=False,
                        elem_classes=["model-status"],
                    )

        set_apply = gr.Button(S.SET_APPLY, variant="primary")
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
        return S.SET_CACHE_DIR_MISSING, S.SET_NO_MODEL
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
        parts.append(S.SET_DELETED_MODELS.format(n=len(deleted), names=", ".join(deleted)))
    if failed:
        parts.append(S.SET_DELETE_FAILED.format(n=len(failed), details="; ".join(failed)))
    if parts:
        return " | ".join(parts), S.SET_NO_MODEL
    return S.SET_NO_MODELS_FOUND, S.SET_NO_MODEL


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
        if hasattr(engine, "configure"):
            model_changed = engine.configure(model_size, quantization, jit_compile)
        else:
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
            if hasattr(engine, "unload_audio_preprocessors"):
                engine.unload_audio_preprocessors()
        s.batch_size = int(batch_size)
        s.default_language = default_language

        os.makedirs(s.output_dir, exist_ok=True)

        parts = [S.SET_APPLIED_SIZE_QUANT.format(size=model_size, quant=quantization)]
        if model_changed:
            parts.append(S.SET_APPLIED_UNLOADED)
        msg = S.SET_APPLIED.format(details=", ".join(parts))
        lang_update = gr.update(value=default_language)
        if default_language == LANGUAGE_AUTO:
            # library import has no auto option (a saved voice has a concrete language)
            lib_update = gr.update()
        else:
            lib_update = lang_update
        return msg, msg, lang_update, lang_update, lang_update, lang_update, lang_update, lib_update

    def unload_model():
        ctx.engine.unload_model()
        return S.SET_UNLOADED_MSG, S.SET_ASR_NOT_LOADED

    def unload_asr_setting():
        ctx.engine.unload_asr()
        return S.SET_ASR_UNLOADED_MSG

    def clear_yt_cache():
        n = ctx.yt.clear_cache()
        return S.SET_YT_CACHE_CLEARED.format(n=n, plural="y" if n == 1 else "ies")

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
