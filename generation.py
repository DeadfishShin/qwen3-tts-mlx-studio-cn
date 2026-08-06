"""Shared generation pipeline: validation, cancel/timeout, engine calls, history, autosave.

All tab handlers route generation through this module; it is the only caller
of the engine's generate methods.
"""
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import gradio as gr
import numpy as np

from audio_utils import concatenate_audio, export_audio, split_text
from config import LANGUAGE_AUTO, MAX_BATCH_SEGMENTS, STREAMING_INTERVAL_S
from ui import strings as S


class GenerationTimeout(Exception):
    pass


class GenerationCancelled(Exception):
    pass


def stream_to_audio(ctx, stream):
    """Drain an engine streaming generator into one (sr, audio) array.

    Cancel and timeout are checked between chunks — abandoning the generator
    is the only way to interrupt generation (the old thread-pool timeout could
    not). The timeout clock starts at the first chunk so a first-run model
    download doesn't count against it. Does NOT clear the cancel event; run
    boundaries own that.
    """
    sr, parts, first_chunk_at = None, [], None
    try:
        for csr, chunk in stream:
            sr = csr
            parts.append(chunk)
            now = time.monotonic()
            if first_chunk_at is None:
                first_chunk_at = now
            if ctx.cancel_event.is_set():
                raise GenerationCancelled()
            if now - first_chunk_at > ctx.settings.timeout:
                raise GenerationTimeout(S.TIMEOUT_MSG)
    finally:
        stream.close()
    if not parts:
        raise RuntimeError("no audio produced")
    return sr, np.concatenate(parts)


def generate_once(ctx, req):
    """Blocking single generation via the streaming path (cancel/timeout aware)."""
    spec = MODES[req.mode]
    return stream_to_audio(ctx, spec.call_stream(ctx, req, **ctx.settings.gen_kwargs()))


def stream_transcription(ctx, audio_path, language):
    """Stream ASR text into a textbox. Yields (textbox_update, status).

    Cancel is checked between token deltas; a stopped run keeps the partial
    transcript in the textbox.
    """
    ctx.cancel_event.clear()
    yield gr.skip(), S.ASR_LOADING
    text = ""
    stream = ctx.engine.stream_transcribe(audio_path, language=language)
    try:
        for delta in stream:
            text += delta
            words = len(text.split())
            yield gr.update(value=text.strip()), S.TRANSCRIBING.format(words=words)
            if ctx.cancel_event.is_set():
                yield gr.update(value=text.strip()), S.TRANSCRIBE_STOPPED
                return
    except Exception as e:
        gr.Warning(f"Transcription failed: {e}")
        yield gr.skip(), f"Error: {e}"
        return
    finally:
        stream.close()
    if not text.strip():
        yield gr.skip(), "Transcription returned empty"
        return
    yield gr.update(value=text.strip()), f"Transcribed ({len(text.split())} words)"


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


def api_language(language):
    """Map the UI language choice to the engine API value."""
    return "auto" if language == LANGUAGE_AUTO else language


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
    call_stream: Callable           # (ctx, req, **gen_kwargs) -> iterator of (sr, chunk)
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
        call_stream=lambda ctx, req, **kw: ctx.engine.stream_generate_custom_voice(
            req.text, req.speaker, api_language(req.language), req.instruct,
            streaming_interval=STREAMING_INTERVAL_S, **kw),
        history_kwargs=lambda req: dict(
            speaker=req.speaker,
            voice_params=req.instruct if req.instruct else ""),
        status_extras=_no_extras,
        batch_prepare=_batch_prepare_identity,
        call_batch=lambda ctx, texts, req, **kw: ctx.engine.batch_generate_custom_voice(
            texts, req.speaker, api_language(req.language), req.instruct, **kw),
        batch_history_kwargs=lambda req, split_mode: dict(
            speaker=req.speaker, voice_params=f"batch ({split_mode})"),
    ),
    "voice_design": ModeSpec(
        model_type="voice_design", save_prefix="design",
        validate=_validate_design,
        prepare=_no_prepare,
        call_stream=lambda ctx, req, **kw: ctx.engine.stream_generate_voice_design(
            req.text, api_language(req.language), req.instruct,
            streaming_interval=STREAMING_INTERVAL_S, **kw),
        history_kwargs=lambda req: dict(voice_params=req.instruct),
        status_extras=_no_extras,
        batch_prepare=_batch_prepare_design,
        call_batch=lambda ctx, texts, req, **kw: ctx.engine.batch_generate_voice_design(
            texts, api_language(req.language), req.instruct, **kw),
        batch_history_kwargs=lambda req, split_mode: dict(
            voice_params=f"batch ({split_mode})"),
    ),
    "voice_clone": ModeSpec(
        model_type="base", save_prefix="clone",
        validate=_validate_common,
        prepare=_prepare_clone,
        call_stream=lambda ctx, req, **kw: ctx.engine.stream_generate_voice_clone(
            req.text, req.ref_audio, req.ref_text, api_language(req.language),
            denoise_ref=ctx.settings.denoise_ref, trim_ref=req.trim_ref,
            streaming_interval=STREAMING_INTERVAL_S, **kw),
        history_kwargs=lambda req: dict(
            voice_params=f"ref: {_clone_voice_info(req)}"),
        status_extras=_clone_extras,
        batch_prepare=_batch_prepare_clone,
        call_batch=lambda ctx, texts, req, **kw: ctx.engine.batch_generate_voice_clone(
            texts, req.ref_audio, req.ref_text, api_language(req.language),
            denoise_ref=ctx.settings.denoise_ref, trim_ref=req.trim_ref, **kw),
        batch_history_kwargs=lambda req, split_mode: dict(
            voice_params=f"batch ref: {_clone_voice_info(req)}"),
    ),
}


def run_single(ctx, req):
    """Shared single-generation pipeline.

    Yields (audio_update, status). Generation still runs through the engine's
    streaming generator internally — that is what makes the live "Generating…
    Xs" status and cooperative Stop/timeout possible — but the player only
    receives the complete waveform at the end (live chunk playback proved
    unusable on hardware where generation is slower than real-time).
    """
    spec = MODES[req.mode]
    err = spec.validate(ctx, req)
    if err:
        yield gr.skip(), err
        return
    req, err = spec.prepare(ctx, req)
    if err:
        yield gr.skip(), err
        return
    msg = loading_status(ctx, spec.model_type)
    if msg:
        yield gr.skip(), msg

    ctx.cancel_event.clear()
    sr, parts = None, []
    start = time.monotonic()
    first_chunk_at = None
    stopped = timed_out = False
    stream = spec.call_stream(ctx, req, **ctx.settings.gen_kwargs())
    try:
        for csr, chunk in stream:
            sr = csr
            parts.append(chunk)
            now = time.monotonic()
            if first_chunk_at is None:
                first_chunk_at = now
            secs = sum(len(p) for p in parts) / sr
            yield gr.skip(), S.GENERATING_STATUS.format(secs=secs)
            if ctx.cancel_event.is_set():
                stopped = True
                break
            if now - first_chunk_at > ctx.settings.timeout:
                timed_out = True
                break
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        yield gr.skip(), f"Error: {e}"
        return
    finally:
        stream.close()

    if not parts:
        yield gr.skip(), "No audio produced"
        return
    result = (sr, np.concatenate(parts))
    secs = len(result[1]) / sr
    if stopped or timed_out:
        # Partial takes are for immediate listening/manual save only:
        # no history entry, no autosave.
        status = (S.TIMED_OUT_KEPT.format(timeout=ctx.settings.timeout, secs=secs)
                  if timed_out else S.STOPPED_KEPT.format(secs=secs))
        yield result, status
        return
    elapsed = time.monotonic() - start
    ctx.history.add(mode=req.mode, text=req.text, language=req.language,
                    audio=result, **spec.history_kwargs(req))
    save_msg = ""
    if ctx.settings.autosave:
        save_msg = " | " + save_audio(ctx, result, spec.save_prefix)
    yield result, (
        f"Generated in {elapsed:.1f}s | Model: "
        f"{ctx.engine.get_repo_id(spec.model_type)}{spec.status_extras(ctx)}{save_msg}")


def _segment_preview(seg):
    return seg[:50] + "..." if len(seg) > 50 else seg


def run_batch(ctx, req, split_mode, silence_ms, progress):
    """Shared batch pipeline. Generator yielding (audio_update, table_rows, status).

    Batched engine calls stay blocking (continuous-batching throughput);
    cancel is checked between batch calls and between fallback retries.
    Fallback retries go through stream_to_audio, so they are cancel- and
    timeout-aware per chunk. On cancel, completed segments are combined but
    NOT recorded to history.
    """
    from dataclasses import replace

    spec = MODES[req.mode]
    req, err_pack = spec.batch_prepare(ctx, req)
    if err_pack:
        rows, msg = err_pack
        yield None, rows, msg
        return
    segments = split_text(req.text, split_mode)
    if not segments:
        gr.Warning("No text segments found.")
        yield None, [["(empty)", "", ""]], "No segments"
        return
    if len(segments) > MAX_BATCH_SEGMENTS:
        gr.Warning(f"Too many segments ({len(segments)}). Max is {MAX_BATCH_SEGMENTS}.")
        yield None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"
        return

    ctx.cancel_event.clear()
    batch_size = ctx.settings.batch_size
    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0
    cancelled = False
    total = len(segments)

    def run_one(idx, seg):
        nonlocal succeeded, failed, cancelled
        preview = _segment_preview(seg)
        try:
            sr, audio = stream_to_audio(
                ctx, spec.call_stream(ctx, replace(req, text=seg),
                                      **ctx.settings.gen_kwargs()))
            audio_parts.append((sr, audio))
            table_rows.append([str(idx + 1), preview, f"{len(audio) / sr:.1f}s"])
            succeeded += 1
        except GenerationCancelled:
            table_rows.append([str(idx + 1), preview, "Stopped"])
            cancelled = True
        except Exception as e:
            table_rows.append([str(idx + 1), preview, f"Failed: {e}"])
            failed += 1

    def progress_status():
        secs = sum(len(a) / s for s, a in audio_parts)
        return S.BATCH_SEGMENT_PROGRESS.format(
            done=succeeded + failed, total=total, secs=secs)

    for batch_start in range(0, total, batch_size):
        if cancelled or ctx.cancel_event.is_set():
            cancelled = True
            break
        batch_segs = segments[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_segs)))
        try:
            results = spec.call_batch(ctx, batch_segs, req, **ctx.settings.gen_kwargs())
            for j, (sr, audio) in enumerate(results):
                idx = batch_indices[j]
                audio_parts.append((sr, audio))
                table_rows.append(
                    [str(idx + 1), _segment_preview(batch_segs[j]), f"{len(audio) / sr:.1f}s"])
                succeeded += 1
                progress((batch_start + j + 1) / total, desc="Generating segments")
            yield gr.skip(), list(table_rows), progress_status()
        except Exception:
            # Batch failed — retry each segment individually
            for j, seg in enumerate(batch_segs):
                if cancelled or ctx.cancel_event.is_set():
                    cancelled = True
                    break
                run_one(batch_indices[j], seg)
                progress((batch_start + j + 1) / total, desc="Generating segments")
                yield gr.skip(), list(table_rows), progress_status()

    if not audio_parts:
        yield None, table_rows, ("Stopped" if cancelled else "All segments failed")
        return

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    if cancelled:
        yield gr.update(value=combined), table_rows, S.BATCH_STOPPED.format(
            done=succeeded, total=total)
        return
    ctx.history.add(
        mode=req.mode, text=f"[Batch: {succeeded} segments]",
        language=req.language, audio=combined,
        **spec.batch_history_kwargs(req, split_mode),
    )
    status_msg = f"Generated {succeeded}/{total} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    status_msg += spec.status_extras(ctx)
    yield gr.update(value=combined), table_rows, status_msg
