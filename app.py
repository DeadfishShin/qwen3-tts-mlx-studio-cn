import os
import warnings

# Suppress known-harmless upstream warnings:
# - "model of type qwen3_tts to instantiate a model of type ." (unregistered model type)
# - "incorrect regex pattern ... fix_mistral_regex=True" (Qwen2Tokenizer regex issue)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# - "Trying to convert audio automatically from float32 to 16-bit int format" (Gradio)
warnings.filterwarnings("ignore", message="Trying to convert audio")

import argparse
import concurrent.futures
import shutil
import sys
import time
import types
from datetime import datetime

import gradio as gr
import numpy as np
import soundfile as sf

from audio_utils import concatenate_audio, export_audio, split_text, unload_deepfilter
from config import (
    DEFAULT_AUTOSAVE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SPLIT_MODE,
    DEFAULT_DENOISE_REF,
    DEFAULT_EXPORT_FORMAT,
    DEFAULT_LOUDNORM,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_SIZE,
    DEFAULT_MP3_BITRATE,
    DEFAULT_QUANTIZATION,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SCRIPT_SILENCE_MS,
    DEFAULT_SILENCE_GAP_MS,
    DEFAULT_SPEAKERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TRIM_SILENCE,
    ENABLE_JIT_COMPILE,
    HISTORY_DIR,
    LANGUAGES,
    MAX_BATCH_SEGMENTS,
    MAX_BATCH_SIZE,
    MAX_SCRIPT_SPEAKERS,
    MIN_BATCH_SIZE,
    OUTPUT_DIR,
    SERVER_HOST,
    SERVER_PORT,
    VOICE_LIBRARY_DIR,
    YT_CACHE_DIR,
)
from engine import TTSEngine
from history import GenerationHistory
from script_parser import parse_script, group_by_model_type
from state import AppContext, AppSettings
from theme import build_theme, custom_css
from ui.tabs import custom_voice as cv_tab
from ui.tabs import history_tab as hist_tab
from ui.tabs import script_mode as sm_tab
from ui.tabs import transcription as asr_tab
from ui.tabs import voice_clone as vc_tab
from ui.tabs import voice_design as vd_tab
from ui.tabs import voice_library as lib_tab
from ui.tabs import yt_clone as yt_tab
from voice_library import VoiceLibrary
from yt_voice import get_yt_extractor

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Qwen3-TTS MLX Studio")
parser.add_argument("--host", default=SERVER_HOST, help="Server host")
parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
parser.add_argument(
    "--model-size", choices=["0.6B", "1.7B"], default=DEFAULT_MODEL_SIZE
)
parser.add_argument(
    "--quant", choices=["4bit", "6bit", "8bit", "bf16"], default=DEFAULT_QUANTIZATION
)
parser.add_argument(
    "--share", action="store_true", help="Create public Gradio link"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def check_startup():
    warnings = []
    if sys.version_info < (3, 10):
        warnings.append("Python 3.10+ required")
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        warnings.append("mlx-audio not installed — run: pip install mlx-audio")
    if not shutil.which("ffmpeg"):
        warnings.append("ffmpeg not found — run: brew install ffmpeg")
    try:
        # Skip ffmpeg — already checked above
        yt_missing = [w for w in get_yt_extractor().check_dependencies()
                      if "ffmpeg" not in w]
        warnings.extend(yt_missing)
    except Exception as e:
        warnings.append(f"YT Voice Clone init error: {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    return warnings

startup_warnings = check_startup()

# ---------------------------------------------------------------------------
# Engine, library & history
# ---------------------------------------------------------------------------
engine = TTSEngine()
engine.model_size = args.model_size
engine.quantization = args.quant

library = VoiceLibrary()
history = GenerationHistory()
yt_extractor = get_yt_extractor()

ctx = AppContext(
    engine=engine, library=library, history=history, yt=yt_extractor,
    settings=AppSettings(), startup_warnings=startup_warnings,
)

# ---------------------------------------------------------------------------
# Runtime settings (mutated by Settings tab)
# ---------------------------------------------------------------------------
app_settings = {
    "temperature": DEFAULT_TEMPERATURE,
    "top_k": DEFAULT_TOP_K,
    "top_p": DEFAULT_TOP_P,
    "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "timeout": DEFAULT_TIMEOUT,
    "output_dir": OUTPUT_DIR,
    "autosave": DEFAULT_AUTOSAVE,
    "export_format": DEFAULT_EXPORT_FORMAT,
    "mp3_bitrate": DEFAULT_MP3_BITRATE,
    "loudnorm": DEFAULT_LOUDNORM,
    "trim_silence": DEFAULT_TRIM_SILENCE,
    "denoise_ref": DEFAULT_DENOISE_REF,
    "batch_size": DEFAULT_BATCH_SIZE,
    "default_language": "English",
}

# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------
class GenerationTimeout(Exception):
    pass


def generate_with_timeout(func, *func_args, timeout_seconds=120, **func_kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *func_args, **func_kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise GenerationTimeout(
                "Generation timed out — try shorter text or lower max_new_tokens"
            )


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------
def _gen_kwargs():
    """Build kwargs dict for engine generate calls from current settings."""
    return {
        "temperature": app_settings["temperature"],
        "top_k": app_settings["top_k"],
        "top_p": app_settings["top_p"],
        "repetition_penalty": app_settings["repetition_penalty"],
        "max_tokens": app_settings["max_tokens"],
    }


def save_audio(audio_tuple, prefix="output"):
    """Save generated audio to the configured output directory."""
    if audio_tuple is None:
        gr.Warning("No audio to save.")
        return "No audio to save"
    sr, audio = audio_tuple
    out_dir = app_settings["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    fmt = app_settings["export_format"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{timestamp}.wav")
    final_path = export_audio(
        audio=audio,
        sr=sr,
        output_path=path,
        fmt=fmt,
        mp3_bitrate=app_settings["mp3_bitrate"],
        loudnorm=app_settings["loudnorm"],
        trim_silence=app_settings["trim_silence"],
    )
    if fmt != "wav" and final_path.endswith(".wav"):
        gr.Warning(f"ffmpeg not available — saved as WAV instead of {fmt.upper()}.")
    return f"Saved: {final_path}"


def _get_hf_cache_dir() -> str:
    """Return the HuggingFace hub cache directory path."""
    return (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )


def _is_model_cached(repo_id: str) -> bool:
    """Check whether a HuggingFace model repo is already in the local cache."""
    hf_home = _get_hf_cache_dir()
    cache_name = "models--" + repo_id.replace("/", "--")
    snapshots = os.path.join(hf_home, cache_name, "snapshots")
    return os.path.isdir(snapshots) and bool(os.listdir(snapshots))


# ---------------------------------------------------------------------------
# Generation handlers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ASR transcription handlers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Batch generation handlers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# YT Voice Clone handlers
# ---------------------------------------------------------------------------
def clear_yt_cache():
    n = yt_extractor.clear_cache()
    return f"YT cache cleared — {n} entr{'y' if n == 1 else 'ies'} removed"


# ---------------------------------------------------------------------------
# Settings handlers
# ---------------------------------------------------------------------------
def apply_settings(
    model_size, quantization,
    temperature, top_k, top_p, repetition_penalty, max_tokens, timeout,
    output_dir, autosave, jit_compile, default_language,
    export_format, mp3_bitrate, loudnorm, trim_silence, denoise_ref,
    batch_size,
):
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

    app_settings["temperature"] = temperature
    app_settings["top_k"] = int(top_k)
    app_settings["top_p"] = top_p
    app_settings["repetition_penalty"] = repetition_penalty
    app_settings["max_tokens"] = int(max_tokens)
    app_settings["timeout"] = int(timeout)
    app_settings["output_dir"] = output_dir.strip() or OUTPUT_DIR
    app_settings["autosave"] = autosave
    app_settings["export_format"] = export_format
    app_settings["mp3_bitrate"] = int(mp3_bitrate)
    app_settings["loudnorm"] = loudnorm
    app_settings["trim_silence"] = trim_silence
    app_settings["denoise_ref"] = denoise_ref
    if not denoise_ref:
        unload_deepfilter()
    app_settings["batch_size"] = int(batch_size)
    app_settings["default_language"] = default_language

    # Transitional bridge: keep ctx.settings in sync until every tab reads it directly.
    for k, v in app_settings.items():
        setattr(ctx.settings, k, v)

    os.makedirs(app_settings["output_dir"], exist_ok=True)

    parts = [f"size: {model_size}, quant: {quantization}"]
    if model_changed:
        parts.append("model unloaded")
    msg = f"Settings applied — {', '.join(parts)}."
    lang_update = gr.update(value=default_language)
    return msg, msg, lang_update, lang_update, lang_update, lang_update, lang_update, lang_update


GENERATION_PRESETS = {
    "Balanced": (0.9, 50, 1.0, 1.05, 4096, 120),
    "Creative": (1.2, 80, 0.95, 1.0, 4096, 120),
    "Precise": (0.3, 20, 0.9, 1.1, 4096, 120),
}


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


def unload_model():
    engine.unload_model()
    return "Model unloaded. RAM freed.", "Not loaded (loads on demand)"


def unload_asr_setting():
    engine.unload_asr()
    return "ASR unloaded. RAM freed."


def delete_cached_models():
    """Delete all Qwen3-TTS model files from the HuggingFace cache."""
    engine.unload_model()
    hf_cache = _get_hf_cache_dir()
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


def get_model_status():
    if engine.current_model is None:
        return "No model loaded"
    repo = engine.get_repo_id(engine.current_model_type)
    return f"Loaded: {repo}"


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Qwen3-TTS MLX Studio") as app:

    gr.HTML(
        "<div class='app-header'>"
        "<h1>Qwen3-TTS MLX Studio</h1>"
        "<p class='subtitle'>Local AI Text-to-Speech &middot; MLX &middot; Apple Silicon</p>"
        "</div>"
    )

    with gr.Tabs():
        ui_ns = types.SimpleNamespace()
        ui_ns.cv = cv_tab.build(ctx)
        ui_ns.vd = vd_tab.build(ctx)

        ui_ns.vc = vc_tab.build(ctx)

        ui_ns.yt = yt_tab.build(ctx)

        ui_ns.sm = sm_tab.build(ctx)

        ui_ns.asr = asr_tab.build(ctx)

        ui_ns.lib = lib_tab.build(ctx)

        ui_ns.hist = hist_tab.build(ctx)

        # =================================================================
        # Tab 9: Settings
        # =================================================================
        with gr.Tab("Settings"):
            with gr.Row():
                # --- Column 1: Model & Language ---
                with gr.Column(scale=1):
                    gr.Markdown("### Model")
                    set_size = gr.Radio(
                        ["0.6B", "1.7B"],
                        value=engine.model_size,
                        label="Model Size",
                    )
                    set_quant = gr.Radio(
                        ["4bit", "6bit", "8bit", "bf16"],
                        value=engine.quantization,
                        label="Quantization",
                    )
                    set_status = gr.Textbox(
                        label="Loaded Model",
                        value=get_model_status(),
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
                        choices=LANGUAGES,
                        value="English",
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
                            value=os.path.abspath(_get_hf_cache_dir()),
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
                    )
                    set_max_tokens = gr.Slider(
                        512, 8192, value=DEFAULT_MAX_TOKENS, step=256,
                        label="Max Tokens",
                    )
                    set_timeout = gr.Slider(
                        30, 300, value=DEFAULT_TIMEOUT, step=10,
                        label="Generation Timeout (seconds)",
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

    # Status bar
    status = gr.Textbox(
        show_label=False,
        interactive=False,
        elem_classes=["status-bar"],
        value="Ready" + (f" | Warnings: {'; '.join(startup_warnings)}" if startup_warnings else ""),
    )

    # ===================================================================
    # Event wiring
    # ===================================================================

    ui_ns.status = status
    cv_tab.wire(ctx, ui_ns)
    vd_tab.wire(ctx, ui_ns)

    vc_tab.wire(ctx, ui_ns)

    yt_tab.wire(ctx, ui_ns)

    sm_tab.wire(ctx, ui_ns)

    lib_tab.wire(ctx, ui_ns)

    hist_tab.wire(ctx, ui_ns)

    # --- Settings ---
    set_export_format.change(
        fn=lambda fmt: gr.update(visible=(fmt == "mp3")),
        inputs=[set_export_format],
        outputs=[set_mp3_bitrate],
    )
    set_apply.click(
        fn=apply_settings,
        inputs=[
            set_size, set_quant,
            set_temperature, set_top_k, set_top_p, set_rep_penalty,
            set_max_tokens, set_timeout,
            set_output_dir, set_autosave, set_jit, set_default_language,
            set_export_format, set_mp3_bitrate, set_loudnorm, set_trim_silence,
            set_denoise_ref,
            set_batch_size,
        ],
        outputs=[
            set_status, status,
            ui_ns.cv.cv_language, ui_ns.vd.vd_language, ui_ns.vc.vc_language, ui_ns.yt.yt_language, ui_ns.asr.asr_language, ui_ns.lib.lib_import_language,
        ],
    )
    set_preset.change(
        fn=apply_preset,
        inputs=[set_preset],
        outputs=[set_temperature, set_top_k, set_top_p, set_rep_penalty, set_max_tokens, set_timeout],
    )
    set_reset.click(
        fn=reset_generation_defaults,
        outputs=[set_temperature, set_top_k, set_top_p, set_rep_penalty, set_max_tokens, set_timeout, set_preset],
    )
    set_unload.click(
        fn=unload_model,
        outputs=[set_status, set_asr_status],
    )
    set_asr_unload.click(
        fn=unload_asr_setting,
        outputs=[set_asr_status],
    )
    set_delete_models.click(
        fn=delete_cached_models,
        outputs=[set_delete_status, set_status],
    )
    set_yt_cache_btn.click(fn=clear_yt_cache, outputs=[set_yt_cache_status])

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.queue(max_size=5).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        css=custom_css,
        theme=build_theme(),
    )
