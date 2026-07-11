"""Shared generation pipeline: validation, timeout, engine calls, history, autosave.

All tab handlers route generation through this module; it is the only caller
of the engine's generate methods.
"""
import concurrent.futures
import os
from datetime import datetime

import gradio as gr

from audio_utils import export_audio


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


def save_audio(ctx, audio_tuple, prefix="output"):
    """Save generated audio to the configured output directory."""
    if audio_tuple is None:
        gr.Warning("No audio to save.")
        return "No audio to save"
    sr, audio = audio_tuple
    out_dir = ctx.settings.output_dir
    os.makedirs(out_dir, exist_ok=True)
    fmt = ctx.settings.export_format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{timestamp}.wav")
    final_path = export_audio(
        audio=audio,
        sr=sr,
        output_path=path,
        fmt=fmt,
        mp3_bitrate=ctx.settings.mp3_bitrate,
        loudnorm=ctx.settings.loudnorm,
        trim_silence=ctx.settings.trim_silence,
    )
    if fmt != "wav" and final_path.endswith(".wav"):
        gr.Warning(f"ffmpeg not available — saved as WAV instead of {fmt.upper()}.")
    return f"Saved: {final_path}"


def get_hf_cache_dir() -> str:
    """Return the HuggingFace hub cache directory path."""
    return (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )


def is_model_cached(repo_id: str) -> bool:
    """Check whether a HuggingFace model repo is already in the local cache."""
    hf_home = get_hf_cache_dir()
    cache_name = "models--" + repo_id.replace("/", "--")
    snapshots = os.path.join(hf_home, cache_name, "snapshots")
    return os.path.isdir(snapshots) and bool(os.listdir(snapshots))


def loading_status(ctx, model_type):
    """Status message to show before a generation that must load/download the model."""
    if ctx.engine.is_model_loaded(model_type):
        return None
    repo_id = ctx.engine.get_repo_id(model_type)
    if is_model_cached(repo_id):
        return f"Loading model into memory… ({repo_id})"
    return (
        f"Downloading model on first run (~6 GB) — this may take several minutes… ({repo_id})"
    )
