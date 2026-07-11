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
from ui.tabs import transcription as asr_tab
from ui.tabs import voice_clone as vc_tab
from ui.tabs import voice_design as vd_tab
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


def _voice_choices():
    """Return list of saved voice names for dropdowns."""
    return ["None"] + [v["name"] for v in library.list_voices()]


def _voice_table():
    """Return voice data as a Markdown table string (gr.Dataframe is buggy; see project memory)."""
    voices = library.list_voices()
    rows = [
        [v["name"], v.get("source", ""), v.get("language", ""), v.get("description", "")]
        for v in voices
    ]
    return _format_table_md(
        ["Name", "Source", "Language", "Description"],
        rows,
        "*No voices saved.*",
    )


# ---------------------------------------------------------------------------
# Generation handlers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ASR transcription handlers
# ---------------------------------------------------------------------------
def transcribe_yt_clip(clip_audio):
    """Transcribe extracted YT clip audio."""
    if not clip_audio:
        gr.Warning("Extract a clip first (Step 2).")
        return gr.update(), "No clip to transcribe"
    yield gr.update(), "Loading ASR model..."
    try:
        text = engine.transcribe(clip_audio, language="auto")
        if not text or not text.strip():
            yield gr.update(), "Transcription returned empty"
            return
        yield gr.update(value=text.strip()), f"Transcribed ({len(text.split())} words)"
    except Exception as e:
        gr.Warning(f"Transcription failed: {e}")
        yield gr.update(), f"Error: {e}"


# ---------------------------------------------------------------------------
# Batch generation handlers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Script mode handlers
# ---------------------------------------------------------------------------
def parse_script_handler(raw_text):
    """Parse script and return speaker info for voice assignment UI."""
    if not raw_text.strip():
        gr.Warning("Enter a script first.")
        return gr.update(), "Enter a script first", {}

    parsed = parse_script(raw_text)

    if parsed.errors:
        gr.Warning(parsed.errors[0])
        return gr.update(), "; ".join(parsed.errors), {}

    speaker_info = {
        "speakers": parsed.speakers,
        "line_count": len(parsed.lines),
        "lines_per_speaker": {},
    }
    for line in parsed.lines:
        speaker_info["lines_per_speaker"].setdefault(line.speaker, 0)
        speaker_info["lines_per_speaker"][line.speaker] += 1

    summary_parts = [f"Found {len(parsed.speakers)} speakers, {len(parsed.lines)} lines:"]
    for spk in parsed.speakers:
        count = speaker_info["lines_per_speaker"].get(spk, 0)
        summary_parts.append(f"  {spk}: {count} lines")

    # Build visibility updates for speaker slots
    updates = []
    for i in range(MAX_SCRIPT_SPEAKERS):
        if i < len(parsed.speakers):
            updates.append(gr.update(visible=True, label=f"Speaker: {parsed.speakers[i]}"))
        else:
            updates.append(gr.update(visible=False))

    return updates, "\n".join(summary_parts), speaker_info


def _batch_custom_voice_for_script(engine, texts, speakers, language, instructs, **kwargs):
    """Batch generate for script mode — supports per-line speakers and instructs."""
    engine._acquire_lock()
    try:
        engine._load_model("custom_voice")
        results = list(
            engine.current_model.batch_generate(
                texts=texts,
                voices=speakers,
                instructs=instructs,
                lang_code=language,
                **kwargs,
            )
        )
        results.sort(key=lambda r: r.sequence_idx)
        return [engine._to_numpy(r) for r in results]
    finally:
        engine._lock.release()


def _batch_voice_design_for_script(engine, texts, language, instructs, **kwargs):
    """Batch generate for script mode — supports per-line instructs."""
    engine._acquire_lock()
    try:
        engine._load_model("voice_design")
        results = list(
            engine.current_model.batch_generate(
                texts=texts,
                instructs=instructs,
                lang_code=language,
                **kwargs,
            )
        )
        results.sort(key=lambda r: r.sequence_idx)
        return [engine._to_numpy(r) for r in results]
    finally:
        engine._lock.release()


def _format_table_md(headers, rows, empty_msg="*No entries.*"):
    """Format rows as a Markdown table. Avoids gr.Dataframe AG Grid recursion bug."""
    if not rows:
        return empty_msg
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = [str(c).replace("|", "\\|") for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _history_table_md():
    """Return history data formatted as a Markdown table string."""
    return _format_table_md(
        ["ID", "Time", "Mode", "Text", "Duration"],
        history.table_data(),
        "*No history entries.*",
    )


def generate_script_handler(raw_text, assignments_state, silence_ms, progress=gr.Progress()):
    """Generate audio for a parsed multi-speaker script.

    assignments_state is a dict mapping speaker name to voice config:
      {speaker: {"mode": ..., "speaker": ..., "language": ..., "instruct": ..., "library_voice": ...}}
    """
    if not raw_text.strip():
        gr.Warning("Enter a script first.")
        return None, "*Enter a script first.*", "Enter script first"

    parsed = parse_script(raw_text)
    if parsed.errors:
        gr.Warning(parsed.errors[0])
        return None, f"*{parsed.errors[0]}*", parsed.errors[0]

    if not assignments_state:
        gr.Warning("Parse the script and assign voices first.")
        return None, "*Parse the script and assign voices first.*", "Parse script first"

    # Group lines by model type for efficient model swapping
    groups = group_by_model_type(parsed.lines, assignments_state)

    # Generate audio for each line, grouped by model type
    audio_by_line_number = {}  # line_number -> (sr, audio)
    table_rows = []
    succeeded, failed = 0, 0

    model_type_labels = {
        "custom_voice": "Custom Voice",
        "voice_design": "Voice Design",
        "base": "Voice Clone",
    }

    total_lines = len(parsed.lines)
    done = 0

    for model_type, lines in groups.items():
        label = model_type_labels.get(model_type, model_type)
        batch_size = app_settings["batch_size"]

        if model_type in ("custom_voice", "voice_design") and batch_size > 1:
            # Batch generation for Custom Voice and Voice Design
            for batch_start in range(0, len(lines), batch_size):
                batch_lines = lines[batch_start:batch_start + batch_size]
                texts = [l.text for l in batch_lines]

                try:
                    if model_type == "custom_voice":
                        speakers = []
                        instructs = []
                        lang = None
                        for line in batch_lines:
                            assignment = assignments_state.get(line.speaker, {})
                            lang = assignment.get("language", "English")
                            speakers.append(assignment.get("speaker", DEFAULT_SPEAKERS[0]))
                            instructs.append(assignment.get("instruct", ""))

                        results = generate_with_timeout(
                            _batch_custom_voice_for_script,
                            engine, texts, speakers, lang, instructs,
                            timeout_seconds=app_settings["timeout"],
                            **_gen_kwargs(),
                        )
                    else:  # voice_design
                        instructs = []
                        lang = None
                        for line in batch_lines:
                            assignment = assignments_state.get(line.speaker, {})
                            lang = assignment.get("language", "English")
                            instructs.append(assignment.get("instruct", ""))

                        results = generate_with_timeout(
                            _batch_voice_design_for_script,
                            engine, texts, lang, instructs,
                            timeout_seconds=app_settings["timeout"],
                            **_gen_kwargs(),
                        )

                    for j, (sr, audio) in enumerate(results):
                        audio_by_line_number[batch_lines[j].line_number] = (sr, audio)
                        succeeded += 1
                        done += 1
                        progress(done / total_lines, desc=f"Generating {label} lines...")

                except Exception:
                    # Batch failed — retry each line individually
                    for line in batch_lines:
                        assignment = assignments_state.get(line.speaker, {})
                        lang = assignment.get("language", "English")
                        try:
                            if model_type == "custom_voice":
                                sr, audio = generate_with_timeout(
                                    engine.generate_custom_voice,
                                    line.text,
                                    assignment.get("speaker", DEFAULT_SPEAKERS[0]),
                                    lang,
                                    assignment.get("instruct", ""),
                                    timeout_seconds=app_settings["timeout"],
                                    **_gen_kwargs(),
                                )
                            else:
                                sr, audio = generate_with_timeout(
                                    engine.generate_voice_design,
                                    line.text,
                                    lang,
                                    assignment.get("instruct", ""),
                                    timeout_seconds=app_settings["timeout"],
                                    **_gen_kwargs(),
                                )
                            audio_by_line_number[line.line_number] = (sr, audio)
                            succeeded += 1
                        except Exception:
                            audio_by_line_number[line.line_number] = None
                            failed += 1
                        done += 1
                        progress(done / total_lines)
        else:
            # Sequential generation (Voice Clone, or batch_size == 1)
            for line in lines:
                assignment = assignments_state.get(line.speaker, {})
                mode = assignment.get("mode", "custom_voice")
                lang = assignment.get("language", "English")

                try:
                    if mode == "custom_voice":
                        sr, audio = generate_with_timeout(
                            engine.generate_custom_voice,
                            line.text,
                            assignment.get("speaker", DEFAULT_SPEAKERS[0]),
                            lang,
                            assignment.get("instruct", ""),
                            timeout_seconds=app_settings["timeout"],
                            **_gen_kwargs(),
                        )
                    elif mode == "voice_design":
                        sr, audio = generate_with_timeout(
                            engine.generate_voice_design,
                            line.text,
                            lang,
                            assignment.get("instruct", ""),
                            timeout_seconds=app_settings["timeout"],
                            **_gen_kwargs(),
                        )
                    elif mode == "voice_clone":
                        lib_voice = assignment.get("library_voice", "")
                        if not lib_voice or lib_voice == "None":
                            raise ValueError("No library voice selected for clone mode")
                        voice = library.load_voice(lib_voice)
                        ref_audio_path = library.get_ref_audio_path(lib_voice)
                        ref_text = voice["ref_text"]
                        sr, audio = generate_with_timeout(
                            engine.generate_voice_clone,
                            line.text, ref_audio_path, ref_text, lang,
                            denoise_ref=app_settings["denoise_ref"],
                            timeout_seconds=app_settings["timeout"],
                            **_gen_kwargs(),
                        )
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    audio_by_line_number[line.line_number] = (sr, audio)
                    succeeded += 1
                except Exception:
                    audio_by_line_number[line.line_number] = None
                    failed += 1

                done += 1
                progress(done / total_lines)

    # Reassemble in script order and build results table
    audio_segments = []
    for line in parsed.lines:
        preview = line.text[:40] + "..." if len(line.text) > 40 else line.text
        result = audio_by_line_number.get(line.line_number)
        if result is not None:
            sr, audio = result
            duration = len(audio) / sr
            audio_segments.append((sr, audio))
            table_rows.append([str(line.line_number), line.speaker, preview, f"{duration:.1f}s"])
        else:
            table_rows.append([str(line.line_number), line.speaker, preview, "Failed"])

    if not audio_segments:
        return None, _format_table_md(["Line", "Speaker", "Text", "Status"], table_rows, "*No results.*"), "All lines failed"

    combined = concatenate_audio(audio_segments, silence_ms=int(silence_ms))
    # Record to history
    speakers_used = ", ".join(parsed.speakers[:4])
    if len(parsed.speakers) > 4:
        speakers_used += "..."
    history.add(
        mode="custom_voice",
        text=f"[Script: {succeeded} lines, speakers: {speakers_used}]",
        language="Multi", audio=combined,
        voice_params="script mode",
    )

    status_msg = f"Generated {succeeded}/{total_lines} lines"
    if failed:
        status_msg += f" ({failed} failed)"
    if app_settings["denoise_ref"]:
        status_msg += " | Noise reduction applied"
    return gr.update(value=combined), _format_table_md(["Line", "Speaker", "Text", "Status"], table_rows, "*No results.*"), status_msg


# ---------------------------------------------------------------------------
# History handlers
# ---------------------------------------------------------------------------
def history_preview(entry_id):
    """Load audio for a history entry."""
    if not entry_id or entry_id == "(empty)":
        return None
    audio = history.get_audio(entry_id)
    if audio is None:
        gr.Warning("Audio not found for this entry.")
        return None
    return audio


def history_delete(entry_id):
    """Delete a single history entry."""
    if not entry_id or entry_id == "(empty)":
        return _history_table_md(), "Select an entry first"
    history.delete_entry(entry_id)
    return _history_table_md(), f"Deleted entry {entry_id}"


def history_clear():
    """Clear all history."""
    history.clear()
    return _history_table_md(), "History cleared"


def history_save_audio(entry_id):
    """Save a history entry's audio to the output directory."""
    if not entry_id or entry_id == "(empty)":
        return "Select an entry first"
    audio = history.get_audio(entry_id)
    if audio is None:
        return "Audio not found"
    return save_audio(audio, "history")


def history_regenerate(entry_id):
    """Get params from a history entry for regeneration."""
    if not entry_id or entry_id == "(empty)":
        return "Select an entry first"
    entry = history.get_entry(entry_id)
    if entry is None:
        return "Entry not found"
    parts = [
        f"Mode: {entry.mode.replace('_', ' ').title()}",
        f"Language: {entry.language}",
    ]
    if entry.speaker:
        parts.append(f"Speaker: {entry.speaker}")
    if entry.voice_params:
        parts.append(f"Params: {entry.voice_params}")
    parts.append(f"Text: {entry.text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Save-to-library handlers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# YT Voice Clone handlers
# ---------------------------------------------------------------------------
def fetch_yt_info(url):
    if not url or not url.strip():
        return gr.update(), "_Enter a URL above._", {}, "Enter a YouTube URL"
    try:
        info = yt_extractor.fetch_info(url.strip())
        dur = info.get("duration") or 0
        h, rem = divmod(int(dur), 3600)
        m, s = divmod(rem, 60)
        dur_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        if info.get("has_manual_subs"):
            sub_note = "Manual subtitles"
        elif info.get("has_auto_subs"):
            sub_note = "Auto-generated subtitles (may need editing)"
        else:
            sub_note = "No subtitles — enter transcript manually"
        md = (
            f"**{info['title']}**\n\n"
            f"Duration: {dur_str}  |  {sub_note}"
            + (f"\n\nChannel: {info['uploader']}" if info.get("uploader") else "")
        )
        state = {
            "id": info["id"],
            "title": info["title"],
            "duration": dur,
            "language": info.get("language"),
        }
        return (
            gr.update(value=info.get("thumbnail") or None),
            md,
            state,
            f"✓ Fetched: {info['title'][:60]}",
        )
    except (ValueError, RuntimeError) as e:
        gr.Warning(str(e))
        return gr.update(), f"**Error:** {e}", {}, str(e)
    except Exception as e:
        gr.Warning(f"Failed to fetch video info: {e}")
        return gr.update(), f"**Error:** {e}", {}, f"Error: {e}"


def extract_yt_clip(url, start_str, end_str, video_state, progress=gr.Progress()):
    if not url or not url.strip():
        return gr.update(), gr.update(), "Enter a YouTube URL first"
    if not video_state or not video_state.get("id"):
        return gr.update(), gr.update(), "Fetch video info first (Step 1)"

    vid_dur = video_state.get("duration")

    # Resolve start — blank defaults to beginning of video
    try:
        start_sec = yt_extractor.parse_timestamp(start_str or "0")
    except ValueError as e:
        gr.Warning(str(e))
        return gr.update(), gr.update(), f"Bad start time: {e}"

    # Resolve end — blank defaults to video duration (or full video)
    if not end_str or not end_str.strip():
        if not vid_dur:
            gr.Warning("Enter an end time — video duration unknown.")
            return gr.update(), gr.update(), "Enter an end time"
        end_sec = float(vid_dur)
    else:
        try:
            end_sec = yt_extractor.parse_timestamp(end_str)
        except ValueError:
            gr.Warning("Invalid end time — use mm:ss")
            return gr.update(), gr.update(), "Enter a valid end time"

    if end_sec <= start_sec:
        gr.Warning("End time must be after start time.")
        return gr.update(), gr.update(), "End must be after start"

    clip_dur = end_sec - start_sec
    if clip_dur < 3.0:
        gr.Warning("Clip must be at least 3 seconds.")
        return gr.update(), gr.update(), "Clip too short (min 3 s)"
    if clip_dur > 60.0:
        gr.Warning("Clip capped at 60 seconds.")
        end_sec = start_sec + 60.0
        clip_dur = 60.0
    elif clip_dur > 30.0:
        gr.Warning(f"Clip is {clip_dur:.0f}s — 5-20 s gives best clone quality.")

    if vid_dur and end_sec > vid_dur + 1:
        gr.Warning(f"End time exceeds video duration ({vid_dur:.0f}s).")
        return gr.update(), gr.update(), "End time beyond video end"

    video_id = video_state["id"]

    try:
        progress(0.0, desc="Starting…")
        wav_path, subs_ok = yt_extractor.download_clip(
            url.strip(), video_id, start_sec, end_sec,
            progress_cb=lambda f, d: progress(f, desc=d),
        )
        progress(0.90, desc="Extracting transcript…")
        transcript = yt_extractor.extract_transcript(video_id, start_sec, end_sec)
        progress(1.0, desc="Done")

        note = "transcript auto-filled" if transcript else "no subtitles — enter transcript manually"
        return (
            gr.update(value=wav_path),
            gr.update(value=transcript),
            f"✓ {clip_dur:.1f}s clip extracted — {note}",
        )
    except Exception as e:
        gr.Warning(f"Extraction failed: {e}")
        return gr.update(), gr.update(), f"Error: {e}"


def clone_yt_voice(text, ref_audio, transcript, language, voice_name):
    errors = []
    if not text or not text.strip():
        errors.append("text to synthesize")
    if not ref_audio:
        errors.append("reference clip (extract one first)")
    if not transcript or not transcript.strip():
        errors.append("reference transcript")
    if not voice_name or not voice_name.strip():
        errors.append("voice name")
    if errors:
        msg = "Required: " + ", ".join(errors)
        gr.Warning(msg)
        return gr.update(), msg, msg, gr.update(), gr.update()

    try:
        t0 = time.time()
        result = generate_with_timeout(
            engine.generate_voice_clone,
            text.strip(), ref_audio, transcript.strip(), language,
            denoise_ref=app_settings["denoise_ref"],
            timeout_seconds=app_settings["timeout"],
            **_gen_kwargs(),
        )
        elapsed = time.time() - t0
        lib_msg = vc_tab.save_clone_to_library(
            ctx, ref_audio, transcript.strip(), voice_name.strip(), language
        )
        history.add(
            mode="voice_clone",
            text=text.strip(),
            language=language,
            audio=result,
            voice_params=f"yt: {voice_name.strip()}",
        )
        extra = ""
        if app_settings["autosave"]:
            extra = " | " + save_audio(result, "yt_clone")
        denoise_msg = " | Noise reduction applied" if app_settings["denoise_ref"] else ""
        status_msg = f"Generated in {elapsed:.1f}s | {lib_msg}{denoise_msg}{extra}"
        return (
            gr.update(value=result),
            status_msg,
            f"✓ {lib_msg}",
            gr.update(choices=_voice_choices()),
            gr.update(value=_voice_table()),
        )
    except GenerationTimeout as e:
        gr.Warning(str(e))
        return gr.update(), str(e), str(e), gr.update(), gr.update()
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        return gr.update(), f"Error: {e}", f"Error: {e}", gr.update(), gr.update()


def clear_yt_cache():
    n = yt_extractor.clear_cache()
    return f"YT cache cleared — {n} entr{'y' if n == 1 else 'ies'} removed"


# ---------------------------------------------------------------------------
# Library management handlers
# ---------------------------------------------------------------------------
def preview_voice(voice_name):
    if not voice_name or voice_name == "(empty)":
        return None
    try:
        path = library.get_ref_audio_path(voice_name)
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


def delete_voice(voice_name):
    if not voice_name or voice_name == "(empty)":
        return _voice_table(), "Select a voice first"
    if not library.delete_voice(voice_name):
        return _voice_table(), f"Voice '{voice_name}' not found"
    return _voice_table(), f"Deleted '{voice_name}'"


def rename_voice(old_name, new_name):
    if not old_name or old_name == "(empty)":
        return _voice_table(), "Select a voice first"
    if not new_name.strip():
        return _voice_table(), "Enter a new name"
    ok = library.rename_voice(old_name, new_name.strip())
    if ok:
        return _voice_table(), f"Renamed '{old_name}' to '{new_name.strip()}'"
    return _voice_table(), f"Rename failed (name may already exist)"


def import_voice(audio_path, transcript, name, language):
    if not audio_path:
        gr.Warning("Upload audio to import.")
        return _voice_table(), "Upload audio first"
    if not name.strip():
        gr.Warning("Enter a name for the imported voice.")
        return _voice_table(), "Enter a name"
    if not transcript or not transcript.strip():
        gr.Warning("Transcript required for imported voice.")
        return _voice_table(), "Enter transcript"
    library.save_voice(
        name=name.strip(),
        ref_audio_path=audio_path,
        ref_text=transcript.strip(),
        language=language,
        description="Imported voice",
        source="import",
    )
    return _voice_table(), f"Imported '{name.strip()}'"


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

        # =================================================================
        # Tab 4: YT Voice Clone
        # =================================================================
        with gr.Tab("YT Voice Clone"):
            gr.HTML(
                "<div class='info-notice'>"
                "Clone any voice from YouTube: fetch a video, pick a timestamp range, "
                "auto-extract the clip with aligned transcript, then generate."
                "</div>"
            )
            yt_video_state = gr.State({"id": None, "title": None, "duration": None})

            with gr.Row():
                # Left column — step-by-step controls
                with gr.Column(scale=3):
                    gr.Markdown("**Step 1 — Video URL**", elem_classes=["yt-step"])
                    yt_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        show_label=False,
                    )
                    yt_fetch_btn = gr.Button("Fetch Video Info", variant="secondary")

                    gr.Markdown("**Step 2 — Select Clip**", elem_classes=["yt-step"])
                    with gr.Row():
                        yt_start = gr.Textbox(
                            label="Start Time (mm:ss)", placeholder="blank = beginning", scale=1
                        )
                        yt_end = gr.Textbox(
                            label="End Time (mm:ss)", placeholder="blank = end of video", scale=1
                        )
                    yt_extract_btn = gr.Button("Extract Clip", variant="primary")

                    gr.Markdown("**Step 3 — Review Transcript**", elem_classes=["yt-step"])
                    yt_transcript = gr.Textbox(
                        label="Reference Transcript (auto-filled from subtitles — edit if needed)",
                        lines=3,
                        placeholder="Transcript appears here after extraction, or enter manually…",
                    )
                    yt_transcribe_btn = gr.Button("Transcribe Clip", variant="secondary")
                    gr.HTML("<div class='text-hint'>Use ASR when subtitles are unavailable or inaccurate</div>")

                    gr.Markdown("**Step 4 — Generate & Save**", elem_classes=["yt-step"])
                    yt_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter text to speak in the cloned voice…",
                    )
                    with gr.Row():
                        yt_language = gr.Dropdown(
                            choices=LANGUAGES, value="English", label="Language", scale=1
                        )
                        yt_voice_name = gr.Textbox(
                            label="Voice Name (saved to library)",
                            placeholder="yt_speaker",
                            scale=2,
                        )
                    yt_clone_btn = gr.Button("Clone & Save to Library", variant="primary")

                # Right column — preview + output panel
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    yt_thumbnail = gr.Image(
                        label="Video Thumbnail",
                        height=180,
                    )
                    yt_video_info = gr.Markdown(
                        "_Video info appears here after fetching._"
                    )
                    yt_clip_audio = gr.Audio(
                        label="Reference Clip Preview",
                        type="filepath",
                        interactive=False,
                        buttons=["download"],
                    )
                    yt_audio_out = gr.Audio(
                        label="Generated Audio", type="numpy", interactive=False, buttons=["download"]
                    )
                    yt_status = gr.Textbox(
                        label="Status", interactive=False, elem_classes=["save-status-text"]
                    )

        # =================================================================
        # Tab 5: Script Mode
        # =================================================================
        with gr.Tab("Script Mode") as script_tab:
            gr.HTML(
                "<div class='info-notice'>"
                "<strong>Multi-Speaker Script</strong> &nbsp;—&nbsp; "
                "Each line: <code>SPEAKER: Dialogue text</code> &nbsp; Lines without a label are narration."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    sm_script = gr.Textbox(
                        label="Script",
                        lines=10,
                        placeholder=(
                            "NARRATOR: Once upon a time, there lived a curious inventor.\n"
                            "EMMA: Father, look what I found in the attic!\n"
                            "FATHER: That is something I built a long time ago."
                        ),
                        elem_classes=["script-editor"],
                    )
                    with gr.Row():
                        sm_parse_btn = gr.Button("Parse Script", variant="primary", scale=1)
                        sm_silence = gr.Slider(
                            0, 2000, value=DEFAULT_SCRIPT_SILENCE_MS, step=50,
                            label="Silence Between Lines (ms)", scale=2,
                        )
                    sm_parse_status = gr.Textbox(label="Parse Result", interactive=False, lines=2)

                    # Voice assignment state
                    sm_assignments = gr.State({})

                    gr.Markdown("### Voice Assignments")
                    # Pre-allocate speaker slots (show/hide based on parse)
                    sm_speaker_groups = []
                    sm_speaker_modes = []
                    sm_speaker_speakers = []
                    sm_speaker_instructs = []
                    sm_speaker_languages = []
                    sm_speaker_lib_voices = []

                    for i in range(MAX_SCRIPT_SPEAKERS):
                        with gr.Group(visible=False, elem_classes=[f"speaker-slot-{i}"]) as grp:
                            sm_speaker_groups.append(grp)
                            with gr.Row():
                                mode = gr.Radio(
                                    ["Custom Voice", "Voice Design", "Voice Clone"],
                                    value="Custom Voice",
                                    label=f"Speaker {i+1} Mode",
                                    scale=2,
                                )
                                sm_speaker_modes.append(mode)
                                lang = gr.Dropdown(
                                    choices=LANGUAGES, value="English",
                                    label="Language", scale=1,
                                )
                                sm_speaker_languages.append(lang)
                            with gr.Row():
                                spk = gr.Dropdown(
                                    choices=DEFAULT_SPEAKERS,
                                    value=DEFAULT_SPEAKERS[0],
                                    label="Speaker",
                                    scale=1,
                                )
                                sm_speaker_speakers.append(spk)
                                inst = gr.Textbox(
                                    label="Instruct / Description",
                                    placeholder="Style instruction or voice description",
                                    scale=2,
                                )
                                sm_speaker_instructs.append(inst)
                                lib_v = gr.Dropdown(
                                    choices=_voice_choices(),
                                    value="None",
                                    label="Library Voice",
                                    scale=1,
                                )
                                sm_speaker_lib_voices.append(lib_v)

                    with gr.Row():
                        sm_generate_btn = gr.Button("Generate Script", variant="primary")
                        sm_save_btn = gr.Button("Save Combined Audio")
                    with gr.Accordion("Per-line breakdown", open=False):
                        sm_table = gr.Markdown(value="*Results will appear after generation.*")

                with gr.Column(scale=1):
                    sm_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False, buttons=["download"])
                    sm_status = gr.Textbox(label="Status", interactive=False)

        ui_ns.asr = asr_tab.build(ctx)

        # =================================================================
        # Tab 7: Voice Library
        # =================================================================
        with gr.Tab("Voice Library"):
            with gr.Row():
                with gr.Column(scale=2):
                    lib_table = gr.Markdown(value=_voice_table(), label="Saved Voices")
                    with gr.Row():
                        lib_selected = gr.Textbox(
                            label="Voice Name",
                            placeholder="Type or paste voice name",
                            scale=2,
                        )
                        lib_preview_btn = gr.Button("Preview", scale=1)
                        lib_delete_btn = gr.Button("Delete", scale=1)
                    with gr.Row():
                        lib_new_name = gr.Textbox(
                            label="Rename To", placeholder="new_name", scale=2
                        )
                        lib_rename_btn = gr.Button("Rename", scale=1)
                    lib_preview_audio = gr.Audio(label="Reference Audio Preview", buttons=["download"])
                    lib_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Status…",
                        elem_classes=["save-status-text"],
                    )
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    gr.Markdown("### Import Voice")
                    lib_import_audio = gr.Audio(
                        label="Audio File", type="filepath", buttons=["download"]
                    )
                    lib_import_transcript = gr.Textbox(
                        label="Transcript", lines=3
                    )
                    lib_import_name = gr.Textbox(
                        label="Name", placeholder="imported_voice"
                    )
                    lib_import_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="Language"
                    )
                    lib_import_btn = gr.Button("Import Voice", variant="primary")

        # =================================================================
        # Tab 8: History
        # =================================================================
        with gr.Tab("History"):
            hist_table = gr.Markdown(value=_history_table_md(), label="Generation History", elem_classes=["history-table"])
            with gr.Row():
                hist_selected = gr.Textbox(
                    label="Entry ID",
                    placeholder="Paste entry ID to preview/manage",
                    scale=3,
                )
                hist_preview_btn = gr.Button("Preview", scale=1)
                hist_delete_btn = gr.Button("Delete Entry", scale=1)
            with gr.Row():
                hist_save_btn = gr.Button("Save Audio", scale=1)
                hist_regen_btn = gr.Button("Show Params", scale=1)
                hist_clear_btn = gr.Button("Clear All History", scale=1)
            hist_status = gr.Textbox(
                show_label=False, interactive=False,
                placeholder="Status…",
                elem_classes=["save-status-text"],
            )
            with gr.Row():
                hist_audio = gr.Audio(
                    label="Audio Preview", type="numpy", interactive=False, scale=1, buttons=["download"]
                )
                hist_regen_info = gr.Textbox(
                    label="Regeneration Params", interactive=False, lines=3, scale=1
                )

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

    # --- YT Voice Clone ---
    yt_fetch_btn.click(
        fn=fetch_yt_info,
        inputs=[yt_url],
        outputs=[yt_thumbnail, yt_video_info, yt_video_state, yt_status],
        show_progress="minimal",
    )
    yt_extract_btn.click(
        fn=extract_yt_clip,
        inputs=[yt_url, yt_start, yt_end, yt_video_state],
        outputs=[yt_clip_audio, yt_transcript, yt_status],
        show_progress="full",
    )
    yt_transcribe_btn.click(
        fn=transcribe_yt_clip,
        inputs=[yt_clip_audio],
        outputs=[yt_transcript, yt_status],
        show_progress="minimal",
    )

    yt_clone_btn.click(
        fn=clone_yt_voice,
        inputs=[yt_text, yt_clip_audio, yt_transcript, yt_language, yt_voice_name],
        outputs=[yt_audio_out, status, yt_status, ui_ns.vc.vc_library_voice, lib_table],
        show_progress="minimal",
    )

    asr_tab.wire(ctx, ui_ns)

    # --- Script Mode ---
    def _parse_and_update_slots(raw_text):
        """Parse script and update speaker slot visibility."""
        if not raw_text.strip():
            gr.Warning("Enter a script first.")
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, "Enter a script first", {}

        parsed = parse_script(raw_text)

        if parsed.errors:
            gr.Warning(parsed.errors[0])
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, "; ".join(parsed.errors), {}

        # Build summary
        summary_parts = [f"Found {len(parsed.speakers)} speakers, {len(parsed.lines)} lines:"]
        lines_per_speaker = {}
        for line in parsed.lines:
            lines_per_speaker.setdefault(line.speaker, 0)
            lines_per_speaker[line.speaker] += 1
        for spk in parsed.speakers:
            count = lines_per_speaker.get(spk, 0)
            summary_parts.append(f"  {spk}: {count} lines")

        # Build visibility updates for speaker slots
        updates = []
        for i in range(MAX_SCRIPT_SPEAKERS):
            if i < len(parsed.speakers):
                updates.append(gr.update(visible=True))
            else:
                updates.append(gr.update(visible=False))

        # Initial assignments state
        assignments = {}
        for spk in parsed.speakers:
            assignments[spk] = {
                "mode": "custom_voice",
                "speaker": DEFAULT_SPEAKERS[0],
                "language": "English",
                "instruct": "",
                "library_voice": "None",
            }

        return *updates, "\n".join(summary_parts), assignments

    sm_parse_btn.click(
        fn=_parse_and_update_slots,
        inputs=[sm_script],
        outputs=[*sm_speaker_groups, sm_parse_status, sm_assignments],
    )

    # Update assignments state when speaker slot controls change
    def _build_assignments_from_slots(current_assignments, script_text,
                                      *slot_values):
        """Rebuild assignments dict from all speaker slot values."""
        if not current_assignments or not script_text.strip():
            return current_assignments

        parsed = parse_script(script_text)
        if parsed.errors or not parsed.speakers:
            return current_assignments

        # slot_values: for each of MAX_SCRIPT_SPEAKERS slots:
        #   mode, speaker, instruct, language, library_voice
        values_per_slot = 5
        assignments = {}
        mode_map = {
            "Custom Voice": "custom_voice",
            "Voice Design": "voice_design",
            "Voice Clone": "voice_clone",
        }
        for i, spk in enumerate(parsed.speakers):
            if i >= MAX_SCRIPT_SPEAKERS:
                break
            base = i * values_per_slot
            mode_label = slot_values[base] if base < len(slot_values) else "Custom Voice"
            assignments[spk] = {
                "mode": mode_map.get(mode_label, "custom_voice"),
                "speaker": slot_values[base + 1] if base + 1 < len(slot_values) else DEFAULT_SPEAKERS[0],
                "instruct": slot_values[base + 2] if base + 2 < len(slot_values) else "",
                "language": slot_values[base + 3] if base + 3 < len(slot_values) else "English",
                "library_voice": slot_values[base + 4] if base + 4 < len(slot_values) else "None",
            }

        return assignments

    # Collect all slot control components in order
    all_slot_controls = []
    for i in range(MAX_SCRIPT_SPEAKERS):
        all_slot_controls.extend([
            sm_speaker_modes[i],
            sm_speaker_speakers[i],
            sm_speaker_instructs[i],
            sm_speaker_languages[i],
            sm_speaker_lib_voices[i],
        ])

    # Wire generate button
    def _generate_script_with_assignments(raw_text, assignments, silence_ms, *slot_values, progress=gr.Progress()):
        """Build fresh assignments from slot values, then generate."""
        # Rebuild assignments from current slot values
        fresh = _build_assignments_from_slots(assignments, raw_text, *slot_values)
        return generate_script_handler(raw_text, fresh, silence_ms, progress)

    sm_generate_btn.click(
        fn=_generate_script_with_assignments,
        inputs=[sm_script, sm_assignments, sm_silence, *all_slot_controls],
        outputs=[sm_audio, sm_table, sm_status],
        show_progress="full",
    )
    sm_save_btn.click(
        fn=lambda audio: save_audio(audio, "script"),
        inputs=[sm_audio],
        outputs=[sm_status],
    )

    # Refresh library voice dropdowns in all speaker slots when the tab is selected
    def _refresh_script_lib_voices():
        choices = _voice_choices()
        return [gr.update(choices=choices) for _ in range(MAX_SCRIPT_SPEAKERS)]

    script_tab.select(
        fn=_refresh_script_lib_voices,
        outputs=sm_speaker_lib_voices,
    )

    # --- Voice Library ---
    lib_preview_btn.click(
        fn=preview_voice,
        inputs=[lib_selected],
        outputs=[lib_preview_audio],
    )
    lib_delete_btn.click(
        fn=delete_voice,
        inputs=[lib_selected],
        outputs=[lib_table, lib_status],
    )
    lib_rename_btn.click(
        fn=rename_voice,
        inputs=[lib_selected, lib_new_name],
        outputs=[lib_table, lib_status],
    )
    lib_import_btn.click(
        fn=import_voice,
        inputs=[lib_import_audio, lib_import_transcript, lib_import_name, lib_import_language],
        outputs=[lib_table, lib_status],
    )

    # --- History ---
    hist_preview_btn.click(
        fn=history_preview,
        inputs=[hist_selected],
        outputs=[hist_audio],
    )
    hist_delete_btn.click(
        fn=history_delete,
        inputs=[hist_selected],
        outputs=[hist_table, hist_status],
    )
    hist_clear_btn.click(
        fn=history_clear,
        outputs=[hist_table, hist_status],
    )
    hist_save_btn.click(
        fn=history_save_audio,
        inputs=[hist_selected],
        outputs=[hist_status],
    )
    hist_regen_btn.click(
        fn=history_regenerate,
        inputs=[hist_selected],
        outputs=[hist_regen_info],
    )

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
            ui_ns.cv.cv_language, ui_ns.vd.vd_language, ui_ns.vc.vc_language, yt_language, ui_ns.asr.asr_language, lib_import_language,
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
