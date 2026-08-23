import os
import warnings

# Suppress known-harmless upstream warnings:
# - "model of type qwen3_tts to instantiate a model of type ." (unregistered model type)
# - "incorrect regex pattern ... fix_mistral_regex=True" (Qwen2Tokenizer regex issue)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# - "Trying to convert audio automatically from float32 to 16-bit int format" (Gradio)
warnings.filterwarnings("ignore", message="Trying to convert audio")

import argparse
import shutil
import sys
import types

import gradio as gr

from config import (
    DEFAULT_MODEL_SIZE,
    DEFAULT_QUANTIZATION,
    OUTPUT_DIR,
    SERVER_HOST,
    SERVER_PORT,
)
from engine import TTSEngine
from history import GenerationHistory
from state import AppContext, AppSettings
from theme import build_theme, custom_css
from ui import strings as S
from ui.tabs import (
    custom_voice as cv_tab,
    history_tab as hist_tab,
    script_mode as sm_tab,
    settings as settings_tab,
    transcription as asr_tab,
    voice_clone as vc_tab,
    voice_design as vd_tab,
    voice_library as lib_tab,
    yt_clone as yt_tab,
)
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
        warnings.append(S.STARTUP_PYTHON_REQUIRED)
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        warnings.append(S.STARTUP_MLX_AUDIO_MISSING)
    if not shutil.which("ffmpeg"):
        warnings.append(S.STARTUP_FFMPEG_MISSING)
    try:
        # Skip ffmpeg — already checked above
        yt_missing = []
        for warning in get_yt_extractor().check_dependencies():
            if "ffmpeg" in warning:
                continue
            if "yt-dlp" in warning:
                yt_missing.append(S.STARTUP_YT_DLP_MISSING)
            elif "pysrt" in warning:
                yt_missing.append(S.STARTUP_PYSRT_MISSING)
            else:
                yt_missing.append(warning)
        warnings.extend(yt_missing)
    except Exception as e:
        warnings.append(S.STARTUP_YT_ERROR.format(err=e))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    return warnings

startup_warnings = check_startup()

# ---------------------------------------------------------------------------
# Application context
# ---------------------------------------------------------------------------
engine = TTSEngine(model_size=args.model_size, quantization=args.quant)

ctx = AppContext(
    engine=engine,
    library=VoiceLibrary(),
    history=GenerationHistory(),
    yt=get_yt_extractor(),
    settings=AppSettings(),
    startup_warnings=startup_warnings,
)

# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
TABS = [
    ("cv", cv_tab), ("vd", vd_tab), ("vc", vc_tab), ("yt", yt_tab),
    ("sm", sm_tab), ("asr", asr_tab), ("lib", lib_tab), ("hist", hist_tab),
    ("settings", settings_tab),
]

with gr.Blocks(title=S.APP_TITLE) as app:
    gr.HTML(S.APP_HEADER_HTML)

    ui_ns = types.SimpleNamespace()
    with gr.Tabs():
        for name, module in TABS:
            setattr(ui_ns, name, module.build(ctx))

    # Status bar
    ui_ns.status = gr.Textbox(
        show_label=False,
        interactive=False,
        elem_classes=["status-bar"],
        value=S.STATUS_READY + (
            "｜" + S.STATUS_WARNINGS.format(warnings="；".join(startup_warnings))
            if startup_warnings else ""
        ),
    )

    for name, module in TABS:
        module.wire(ctx, ui_ns)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        app.queue(max_size=5).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=True,
            css=custom_css,
            theme=build_theme(),
        )
    finally:
        engine.shutdown()
