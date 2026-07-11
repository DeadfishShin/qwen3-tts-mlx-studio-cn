"""YT Voice Clone tab: clone a voice straight from a YouTube clip."""
import time
import types

import gradio as gr

from config import LANGUAGES
from generation import (
    GenerationTimeout, generate_with_timeout, save_audio,
)
from ui.components import voice_choices, voice_table
from ui.tabs.voice_clone import save_clone_to_library


def build(ctx):
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
    return types.SimpleNamespace(
        yt_video_state=yt_video_state, yt_url=yt_url, yt_fetch_btn=yt_fetch_btn,
        yt_start=yt_start, yt_end=yt_end, yt_extract_btn=yt_extract_btn,
        yt_transcript=yt_transcript, yt_transcribe_btn=yt_transcribe_btn,
        yt_text=yt_text, yt_language=yt_language, yt_voice_name=yt_voice_name,
        yt_clone_btn=yt_clone_btn, yt_thumbnail=yt_thumbnail,
        yt_video_info=yt_video_info, yt_clip_audio=yt_clip_audio,
        yt_audio_out=yt_audio_out, yt_status=yt_status,
    )


def fetch_yt_info(ctx, url):
    if not url or not url.strip():
        return gr.update(), "_Enter a URL above._", {}, "Enter a YouTube URL"
    try:
        info = ctx.yt.fetch_info(url.strip())
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


def extract_yt_clip(ctx, url, start_str, end_str, video_state, progress=gr.Progress()):
    if not url or not url.strip():
        return gr.update(), gr.update(), "Enter a YouTube URL first"
    if not video_state or not video_state.get("id"):
        return gr.update(), gr.update(), "Fetch video info first (Step 1)"

    vid_dur = video_state.get("duration")

    # Resolve start — blank defaults to beginning of video
    try:
        start_sec = ctx.yt.parse_timestamp(start_str or "0")
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
            end_sec = ctx.yt.parse_timestamp(end_str)
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
        wav_path, _ = ctx.yt.download_clip(
            url.strip(), video_id, start_sec, end_sec,
            progress_cb=lambda f, d: progress(f, desc=d),
        )
        progress(0.90, desc="Extracting transcript…")
        transcript = ctx.yt.extract_transcript(video_id, start_sec, end_sec)
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


def transcribe_yt_clip(ctx, clip_audio):
    """Transcribe extracted YT clip audio."""
    if not clip_audio:
        gr.Warning("Extract a clip first (Step 2).")
        return gr.update(), "No clip to transcribe"
    yield gr.update(), "Loading ASR model..."
    try:
        text = ctx.engine.transcribe(clip_audio, language="auto")
        if not text or not text.strip():
            yield gr.update(), "Transcription returned empty"
            return
        yield gr.update(value=text.strip()), f"Transcribed ({len(text.split())} words)"
    except Exception as e:
        gr.Warning(f"Transcription failed: {e}")
        yield gr.update(), f"Error: {e}"


def clone_yt_voice(ctx, text, ref_audio, transcript, language, voice_name):
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
            ctx.engine.generate_voice_clone,
            text.strip(), ref_audio, transcript.strip(), language,
            denoise_ref=ctx.settings.denoise_ref,
            timeout_seconds=ctx.settings.timeout,
            **ctx.settings.gen_kwargs(),
        )
        elapsed = time.time() - t0
        lib_msg = save_clone_to_library(
            ctx, ref_audio, transcript.strip(), voice_name.strip(), language
        )
        ctx.history.add(
            mode="voice_clone",
            text=text.strip(),
            language=language,
            audio=result,
            voice_params=f"yt: {voice_name.strip()}",
        )
        extra = ""
        if ctx.settings.autosave:
            extra = " | " + save_audio(ctx, result, "yt_clone")
        denoise_msg = " | Noise reduction applied" if ctx.settings.denoise_ref else ""
        status_msg = f"Generated in {elapsed:.1f}s | {lib_msg}{denoise_msg}{extra}"
        return (
            gr.update(value=result),
            status_msg,
            f"✓ {lib_msg}",
            gr.update(choices=voice_choices(ctx)),
            gr.update(value=voice_table(ctx)),
        )
    except GenerationTimeout as e:
        gr.Warning(str(e))
        return gr.update(), str(e), str(e), gr.update(), gr.update()
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        return gr.update(), f"Error: {e}", f"Error: {e}", gr.update(), gr.update()


def wire(ctx, ui):
    t = ui.yt

    def on_fetch(url):
        return fetch_yt_info(ctx, url)

    def on_extract(url, start_str, end_str, video_state, progress=gr.Progress()):
        return extract_yt_clip(ctx, url, start_str, end_str, video_state, progress)

    def on_transcribe(clip_audio):
        yield from transcribe_yt_clip(ctx, clip_audio)

    def on_clone(text, ref_audio, transcript, language, voice_name):
        return clone_yt_voice(ctx, text, ref_audio, transcript, language, voice_name)

    t.yt_fetch_btn.click(
        fn=on_fetch,
        inputs=[t.yt_url],
        outputs=[t.yt_thumbnail, t.yt_video_info, t.yt_video_state, t.yt_status],
        show_progress="minimal",
    )
    t.yt_extract_btn.click(
        fn=on_extract,
        inputs=[t.yt_url, t.yt_start, t.yt_end, t.yt_video_state],
        outputs=[t.yt_clip_audio, t.yt_transcript, t.yt_status],
        show_progress="full",
    )
    t.yt_transcribe_btn.click(
        fn=on_transcribe,
        inputs=[t.yt_clip_audio],
        outputs=[t.yt_transcript, t.yt_status],
        show_progress="minimal",
    )
    t.yt_clone_btn.click(
        fn=on_clone,
        inputs=[t.yt_text, t.yt_clip_audio, t.yt_transcript, t.yt_language, t.yt_voice_name],
        outputs=[t.yt_audio_out, ui.status, t.yt_status,
                 ui.vc.vc_library_voice, ui.lib.lib_table],
        show_progress="minimal",
    )
