"""Shared generation pipeline: validation, timeout, engine calls, history, autosave.

All tab handlers route generation through this module; it is the only caller
of the engine's generate methods.
"""
import concurrent.futures
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import gradio as gr

from audio_utils import concatenate_audio, export_audio, split_text
from config import MAX_BATCH_SEGMENTS


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


# ---------------------------------------------------------------------------
# Mode specs — per-mode validation, engine calls, and history fields
# ---------------------------------------------------------------------------
@dataclass
class GenRequest:
    mode: str                       # "custom_voice" | "voice_design" | "voice_clone"
    text: str
    language: str
    speaker: str = ""               # custom_voice
    instruct: str = ""              # style (custom_voice) or description (voice_design)
    ref_audio: Optional[str] = None  # voice_clone
    ref_text: str = ""              # voice_clone
    library_voice: str = "None"     # voice_clone
    trim_ref: bool = True           # voice_clone: VAD-trim reference silence


def _clone_voice_info(req):
    return req.library_voice if req.library_voice and req.library_voice != "None" else "uploaded"


def _validate_common(ctx, req):
    if not req.text.strip():
        gr.Warning("Please enter text to speak.")
        return "Enter text first"
    return None


def _validate_design(ctx, req):
    err = _validate_common(ctx, req)
    if err:
        return err
    if not req.instruct.strip():
        gr.Warning("Please describe the voice you want.")
        return "Describe the voice first"
    return None


def _resolve_clone_library(ctx, req):
    """Resolve a selected library voice onto ref_audio/ref_text. Returns error string or None."""
    if req.library_voice and req.library_voice != "None":
        try:
            voice = ctx.library.load_voice(req.library_voice)
            req.ref_audio = ctx.library.get_ref_audio_path(req.library_voice)
            req.ref_text = voice["ref_text"]
        except FileNotFoundError:
            return "not found"
    return None


def _prepare_clone(ctx, req):
    """Single-generation clone prep: library resolution + ref checks (app.py wording)."""
    if _resolve_clone_library(ctx, req):
        gr.Warning(f"Voice '{req.library_voice}' not found in library.")
        return None, "Voice not found"
    if not req.ref_audio:
        gr.Warning("Please upload reference audio or select from library.")
        return None, "No reference audio"
    if not req.ref_text or not req.ref_text.strip():
        gr.Warning("Reference transcript is required for voice cloning.")
        return None, "No reference transcript"
    return req, None


def _no_prepare(ctx, req):
    return req, None


def _no_extras(ctx):
    return ""


def _clone_extras(ctx):
    return " | Noise reduction applied" if ctx.settings.denoise_ref else ""


def _batch_prepare_identity(ctx, req):
    return req, None


def _batch_prepare_design(ctx, req):
    """Voice Design batch checks instruct up front (app.py batch wording)."""
    if not req.instruct.strip():
        gr.Warning("Please describe the voice.")
        return None, ([["(empty)", "", ""]], "Describe voice first")
    return req, None


def _batch_prepare_clone(ctx, req):
    """Voice Clone batch: library resolution + ref checks (app.py batch wording)."""
    if _resolve_clone_library(ctx, req):
        gr.Warning(f"Voice '{req.library_voice}' not found.")
        return None, ([["(error)", "", ""]], "Voice not found")
    if not req.ref_audio:
        gr.Warning("No reference audio.")
        return None, ([["(error)", "", ""]], "No reference audio")
    if not req.ref_text or not req.ref_text.strip():
        gr.Warning("Reference transcript required.")
        return None, ([["(error)", "", ""]], "No transcript")
    return req, None


@dataclass
class ModeSpec:
    model_type: str
    save_prefix: str
    validate: Callable              # (ctx, req) -> error string | None
    prepare: Callable               # (ctx, req) -> (req | None, error string | None)
    call_single: Callable           # (ctx, req, **gen_kwargs) -> (sr, audio)
    history_kwargs: Callable        # (req) -> dict of extra history.add fields
    status_extras: Callable         # (ctx) -> str appended to success status
    batch_prepare: Callable         # (ctx, req) -> (req | None, (table_rows, error) | None)
    call_batch: Callable            # (ctx, texts, req, **gen_kwargs) -> [(sr, audio)]
    batch_history_kwargs: Callable  # (req, split_mode) -> dict of extra history.add fields


MODES = {
    "custom_voice": ModeSpec(
        model_type="custom_voice", save_prefix="custom",
        validate=_validate_common,
        prepare=_no_prepare,
        call_single=lambda ctx, req, **kw: ctx.engine.generate_custom_voice(
            req.text, req.speaker, req.language, req.instruct, **kw),
        history_kwargs=lambda req: dict(
            speaker=req.speaker,
            voice_params=req.instruct if req.instruct else ""),
        status_extras=_no_extras,
        batch_prepare=_batch_prepare_identity,
        call_batch=lambda ctx, texts, req, **kw: ctx.engine.batch_generate_custom_voice(
            texts, req.speaker, req.language, req.instruct, **kw),
        batch_history_kwargs=lambda req, split_mode: dict(
            speaker=req.speaker, voice_params=f"batch ({split_mode})"),
    ),
    "voice_design": ModeSpec(
        model_type="voice_design", save_prefix="design",
        validate=_validate_design,
        prepare=_no_prepare,
        call_single=lambda ctx, req, **kw: ctx.engine.generate_voice_design(
            req.text, req.language, req.instruct, **kw),
        history_kwargs=lambda req: dict(voice_params=req.instruct),
        status_extras=_no_extras,
        batch_prepare=_batch_prepare_design,
        call_batch=lambda ctx, texts, req, **kw: ctx.engine.batch_generate_voice_design(
            texts, req.language, req.instruct, **kw),
        batch_history_kwargs=lambda req, split_mode: dict(
            voice_params=f"batch ({split_mode})"),
    ),
    "voice_clone": ModeSpec(
        model_type="base", save_prefix="clone",
        validate=_validate_common,
        prepare=_prepare_clone,
        call_single=lambda ctx, req, **kw: ctx.engine.generate_voice_clone(
            req.text, req.ref_audio, req.ref_text, req.language,
            denoise_ref=ctx.settings.denoise_ref, trim_ref=req.trim_ref, **kw),
        history_kwargs=lambda req: dict(
            voice_params=f"ref: {_clone_voice_info(req)}"),
        status_extras=_clone_extras,
        batch_prepare=_batch_prepare_clone,
        call_batch=lambda ctx, texts, req, **kw: ctx.engine.batch_generate_voice_clone(
            texts, req.ref_audio, req.ref_text, req.language,
            denoise_ref=ctx.settings.denoise_ref, trim_ref=req.trim_ref, **kw),
        batch_history_kwargs=lambda req, split_mode: dict(
            voice_params=f"batch ref: {_clone_voice_info(req)}"),
    ),
}


def run_single(ctx, req):
    """Shared single-generation pipeline. Yields (audio_update, status_text)."""
    spec = MODES[req.mode]
    err = spec.validate(ctx, req)
    if err:
        yield None, err
        return
    req, err = spec.prepare(ctx, req)
    if err:
        yield None, err
        return
    msg = loading_status(ctx, spec.model_type)
    if msg:
        yield None, msg
    try:
        start = time.time()
        sr, audio = generate_with_timeout(
            spec.call_single, ctx, req,
            timeout_seconds=ctx.settings.timeout,
            **ctx.settings.gen_kwargs(),
        )
        elapsed = time.time() - start
        result = (sr, audio)
        ctx.history.add(mode=req.mode, text=req.text, language=req.language,
                        audio=result, **spec.history_kwargs(req))
        save_msg = ""
        if ctx.settings.autosave:
            save_msg = " | " + save_audio(ctx, result, spec.save_prefix)
        yield (
            gr.update(value=result),
            f"Generated in {elapsed:.1f}s | Model: "
            f"{ctx.engine.get_repo_id(spec.model_type)}{spec.status_extras(ctx)}{save_msg}",
        )
    except GenerationTimeout as e:
        gr.Warning(str(e))
        yield None, str(e)
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        yield None, f"Error: {e}"


def _segment_preview(seg):
    return seg[:50] + "..." if len(seg) > 50 else seg


def run_batch(ctx, req, split_mode, silence_ms, progress):
    """Shared batch pipeline. Returns (audio_update, table_rows, status_msg)."""
    from dataclasses import replace

    spec = MODES[req.mode]
    req, err_pack = spec.batch_prepare(ctx, req)
    if err_pack:
        rows, msg = err_pack
        return None, rows, msg
    segments = split_text(req.text, split_mode)
    if not segments:
        gr.Warning("No text segments found.")
        return None, [["(empty)", "", ""]], "No segments"
    if len(segments) > MAX_BATCH_SEGMENTS:
        gr.Warning(f"Too many segments ({len(segments)}). Max is {MAX_BATCH_SEGMENTS}.")
        return None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"

    batch_size = ctx.settings.batch_size
    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0
    total = len(segments)

    def run_one(idx, seg):
        nonlocal succeeded, failed
        preview = _segment_preview(seg)
        try:
            sr, audio = generate_with_timeout(
                spec.call_single, ctx, replace(req, text=seg),
                timeout_seconds=ctx.settings.timeout,
                **ctx.settings.gen_kwargs(),
            )
            audio_parts.append((sr, audio))
            table_rows.append([str(idx + 1), preview, f"{len(audio) / sr:.1f}s"])
            succeeded += 1
        except Exception as e:
            table_rows.append([str(idx + 1), preview, f"Failed: {e}"])
            failed += 1

    for batch_start in range(0, total, batch_size):
        batch_segs = segments[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_segs)))
        try:
            results = generate_with_timeout(
                spec.call_batch, ctx, batch_segs, req,
                timeout_seconds=ctx.settings.timeout,
                **ctx.settings.gen_kwargs(),
            )
            for j, (sr, audio) in enumerate(results):
                idx = batch_indices[j]
                audio_parts.append((sr, audio))
                table_rows.append(
                    [str(idx + 1), _segment_preview(batch_segs[j]), f"{len(audio) / sr:.1f}s"])
                succeeded += 1
                progress((batch_start + j + 1) / total, desc="Generating segments")
        except Exception:
            # Batch failed — retry each segment individually
            for j, seg in enumerate(batch_segs):
                run_one(batch_indices[j], seg)
                progress((batch_start + j + 1) / total, desc="Generating segments")

    if not audio_parts:
        return None, table_rows, "All segments failed"

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    ctx.history.add(
        mode=req.mode, text=f"[Batch: {succeeded} segments]",
        language=req.language, audio=combined,
        **spec.batch_history_kwargs(req, split_mode),
    )
    status_msg = f"Generated {succeeded}/{total} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    status_msg += spec.status_extras(ctx)
    return gr.update(value=combined), table_rows, status_msg
