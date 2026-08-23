"""YT Voice Clone tab: clone a voice straight from a YouTube clip."""
import time
import types

import gradio as gr

from generation import (
    GenerationCancelled, GenerationTimeout, GenRequest, generate_once, save_audio,
    stream_transcription,
)
from ui import strings as S
from ui.components import voice_choices, voice_table, wire_run_lifecycle, wire_stop
from ui.tabs.voice_clone import save_clone_to_library


def build(ctx):
    with gr.Tab(S.TAB_YT):
        gr.HTML(S.YT_NOTICE_HTML)
        yt_video_state = gr.State({"id": None, "title": None, "duration": None})

        with gr.Row():
            # Left column — step-by-step controls
            with gr.Column(scale=3):
                gr.Markdown(S.YT_STEP_1, elem_classes=["yt-step"])
                yt_url = gr.Textbox(
                    placeholder=S.YT_URL_PLACEHOLDER,
                    show_label=False,
                )
                yt_fetch_btn = gr.Button(S.YT_FETCH, variant="secondary")

                gr.Markdown(S.YT_STEP_2, elem_classes=["yt-step"])
                with gr.Row():
                    yt_start = gr.Textbox(
                        label=S.YT_START, placeholder=S.YT_START_PLACEHOLDER, scale=1
                    )
                    yt_end = gr.Textbox(
                        label=S.YT_END, placeholder=S.YT_END_PLACEHOLDER, scale=1
                    )
                yt_extract_btn = gr.Button(S.YT_EXTRACT, variant="primary")
                yt_trim_ref = gr.Checkbox(
                    value=True, label=S.TRIM_REF_LABEL
                )

                gr.Markdown(S.YT_STEP_3, elem_classes=["yt-step"])
                yt_transcript = gr.Textbox(
                    label=S.YT_TRANSCRIPT,
                    info=S.YT_TRANSCRIPT_INFO,
                    lines=3,
                    placeholder=S.YT_TRANSCRIPT_PLACEHOLDER,
                )
                with gr.Row():
                    yt_transcribe_btn = gr.Button(S.YT_TRANSCRIBE, variant="secondary")
                    yt_stop_btn = gr.Button(S.STOP, variant="stop", visible=False)
                gr.HTML(S.YT_TRANSCRIBE_HINT_HTML)

                gr.Markdown(S.YT_STEP_4, elem_classes=["yt-step"])
                yt_text = gr.Textbox(
                    label=S.TEXT_TO_SPEAK,
                    lines=4,
                    placeholder=S.TEXT_PLACEHOLDER,
                )
                with gr.Row():
                    yt_language = gr.Dropdown(
                        choices=S.LANGUAGE_CHOICES,
                        value=S.LANGUAGE_AUTO_VALUE, label=S.LANGUAGE, scale=1
                    )
                    yt_voice_name = gr.Textbox(
                        label=S.YT_VOICE_NAME,
                        placeholder=S.YT_VOICE_NAME_PLACEHOLDER,
                        scale=2,
                    )
                yt_clone_btn = gr.Button(S.YT_CLONE, variant="primary")

            # Right column — preview + output panel
            with gr.Column(scale=1, elem_classes=["output-col"]):
                yt_thumbnail = gr.Image(
                    label=S.YT_THUMBNAIL,
                    height=180,
                )
                yt_video_info = gr.Markdown(S.YT_INFO_EMPTY)
                yt_clip_audio = gr.Audio(
                    label=S.YT_CLIP_PREVIEW,
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                )
                yt_audio_out = gr.Audio(
                    label=S.OUTPUT, type="numpy", interactive=False, buttons=["download"]
                )
                yt_status = gr.Textbox(
                    label=S.YT_STATUS, interactive=False, elem_classes=["save-status-text"]
                )
    return types.SimpleNamespace(
        yt_video_state=yt_video_state, yt_url=yt_url, yt_fetch_btn=yt_fetch_btn,
        yt_start=yt_start, yt_end=yt_end, yt_extract_btn=yt_extract_btn,
        yt_trim_ref=yt_trim_ref,
        yt_transcript=yt_transcript, yt_transcribe_btn=yt_transcribe_btn,
        yt_stop_btn=yt_stop_btn,
        yt_text=yt_text, yt_language=yt_language, yt_voice_name=yt_voice_name,
        yt_clone_btn=yt_clone_btn, yt_thumbnail=yt_thumbnail,
        yt_video_info=yt_video_info, yt_clip_audio=yt_clip_audio,
        yt_audio_out=yt_audio_out, yt_status=yt_status,
    )


def fetch_yt_info(ctx, url):
    if not url or not url.strip():
        return gr.update(), S.YT_ENTER_URL_MD, {}, S.YT_ENTER_URL
    try:
        info = ctx.yt.fetch_info(url.strip())
        dur = info.get("duration") or 0
        h, rem = divmod(int(dur), 3600)
        m, s = divmod(rem, 60)
        dur_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        if info.get("has_manual_subs"):
            sub_note = S.YT_SUBS_MANUAL
        elif info.get("has_auto_subs"):
            sub_note = S.YT_SUBS_AUTO
        else:
            sub_note = S.YT_SUBS_NONE
        md = (
            f"**{info['title']}**\n\n"
            + S.YT_DURATION.format(duration=dur_str, subtitle=sub_note)
            + (f"\n\n{S.YT_CHANNEL.format(channel=info['uploader'])}"
               if info.get("uploader") else "")
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
            S.YT_FETCHED.format(title=info["title"][:60]),
        )
    except (ValueError, RuntimeError) as e:
        gr.Warning(S.YT_ERROR.format(err=e))
        return gr.update(), f"**{S.YT_ERROR.format(err=e)}**", {}, S.YT_ERROR.format(err=e)
    except Exception as e:
        gr.Warning(S.YT_FETCH_ERROR.format(err=e))
        return (gr.update(), f"**{S.YT_ERROR.format(err=e)}**", {},
                S.YT_ERROR.format(err=e))


def extract_yt_clip(ctx, url, start_str, end_str, video_state, progress=gr.Progress()):
    if not url or not url.strip():
        return gr.update(), gr.update(), S.YT_ENTER_URL
    if not video_state or not video_state.get("id"):
        return gr.update(), gr.update(), S.YT_FETCH_FIRST

    vid_dur = video_state.get("duration")

    # Resolve start — blank defaults to beginning of video
    try:
        start_sec = ctx.yt.parse_timestamp(start_str or "0")
    except ValueError as e:
        gr.Warning(S.YT_ERROR.format(err=e))
        return gr.update(), gr.update(), S.YT_BAD_START.format(err=e)

    # Resolve end — blank defaults to video duration (or full video)
    if not end_str or not end_str.strip():
        if not vid_dur:
            gr.Warning(S.YT_ENTER_END_WARN)
            return gr.update(), gr.update(), S.YT_ENTER_END
        end_sec = float(vid_dur)
    else:
        try:
            end_sec = ctx.yt.parse_timestamp(end_str)
        except ValueError:
            gr.Warning(S.YT_BAD_END_WARN)
            return gr.update(), gr.update(), S.YT_BAD_END

    if end_sec <= start_sec:
        gr.Warning(S.YT_END_BEFORE_START_WARN)
        return gr.update(), gr.update(), S.YT_END_BEFORE_START

    clip_dur = end_sec - start_sec
    if clip_dur < 3.0:
        gr.Warning(S.YT_CLIP_TOO_SHORT_WARN)
        return gr.update(), gr.update(), S.YT_CLIP_TOO_SHORT
    if clip_dur > 60.0:
        gr.Warning(S.YT_CLIP_CAPPED_WARN)
        end_sec = start_sec + 60.0
        clip_dur = 60.0
    elif clip_dur > 30.0:
        gr.Warning(S.YT_CLIP_LONG_WARN.format(secs=clip_dur))

    if vid_dur and end_sec > vid_dur + 1:
        gr.Warning(S.YT_END_BEYOND_WARN.format(secs=vid_dur))
        return gr.update(), gr.update(), S.YT_END_BEYOND

    video_id = video_state["id"]

    try:
        progress(0.0, desc=S.YT_EXTRACT_PROGRESS_START)
        wav_path, _ = ctx.yt.download_clip(
            url.strip(), video_id, start_sec, end_sec,
            progress_cb=lambda f, d: progress(f, desc={
                "Using cached clip": S.YT_PROGRESS_CACHED,
                "Downloading audio section…": S.YT_PROGRESS_AUDIO,
                "Converting to WAV…": S.YT_PROGRESS_WAV,
                "Downloading subtitles…": S.YT_PROGRESS_SUBTITLES,
                "Done": S.YT_EXTRACT_PROGRESS_DONE,
            }.get(d, d)),
        )
        progress(0.90, desc=S.YT_EXTRACT_PROGRESS_TRANSCRIPT)
        transcript = ctx.yt.extract_transcript(video_id, start_sec, end_sec)
        progress(1.0, desc=S.YT_EXTRACT_PROGRESS_DONE)

        note = S.YT_TRANSCRIPT_FILLED if transcript else S.YT_SUBS_NONE
        return (
            gr.update(value=wav_path),
            gr.update(value=transcript),
            S.YT_EXTRACTED.format(secs=clip_dur, note=note),
        )
    except Exception as e:
        gr.Warning(S.YT_EXTRACT_FAILED.format(err=e))
        return gr.update(), gr.update(), S.YT_ERROR.format(err=e)


def transcribe_yt_clip(ctx, clip_audio):
    """Transcribe extracted YT clip audio (streams live)."""
    if not clip_audio:
        gr.Warning(S.YT_NO_CLIP_WARN)
        yield gr.update(), S.YT_NO_CLIP
        return
    yield from stream_transcription(ctx, clip_audio, "auto")


def clone_yt_voice(ctx, text, ref_audio, transcript, language, voice_name, trim_ref=True):
    errors = []
    if not text or not text.strip():
        errors.append(S.YT_REQ_TEXT)
    if not ref_audio:
        errors.append(S.YT_REQ_CLIP)
    if not transcript or not transcript.strip():
        errors.append(S.YT_REQ_TRANSCRIPT)
    if not voice_name or not voice_name.strip():
        errors.append(S.YT_REQ_NAME)
    if errors:
        msg = S.YT_REQUIRED.format(items=", ".join(errors))
        gr.Warning(msg)
        return gr.update(), msg, msg, gr.update(), gr.update()

    try:
        t0 = time.time()
        ctx.cancel_event.clear()
        result = generate_once(ctx, GenRequest(
            mode="voice_clone", text=text.strip(), language=language,
            ref_audio=ref_audio, ref_text=transcript.strip(), trim_ref=trim_ref))
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
            extra = "｜" + save_audio(ctx, result, "yt_clone")
        denoise_msg = S.NOISE_REDUCTION_SUFFIX if ctx.settings.denoise_ref else ""
        status_msg = S.YT_GENERATED_STATUS.format(
            secs=elapsed, library=lib_msg, denoise=denoise_msg, save=extra)
        return (
            gr.update(value=result),
            status_msg,
            S.YT_SAVED_OK.format(msg=lib_msg),
            gr.update(choices=voice_choices(ctx)),
            gr.update(value=voice_table(ctx)),
        )
    except GenerationCancelled:
        return gr.update(), S.YT_STOPPED, S.YT_STOPPED, gr.update(), gr.update()
    except GenerationTimeout as e:
        gr.Warning(S.ERROR.format(err=e))
        return gr.update(), S.ERROR.format(err=e), S.ERROR.format(err=e), gr.update(), gr.update()
    except Exception as e:
        gr.Warning(S.YT_GENERATION_FAILED.format(err=e))
        return (gr.update(), S.YT_ERROR.format(err=e), S.YT_ERROR.format(err=e),
                gr.update(), gr.update())


def wire(ctx, ui):
    t = ui.yt

    def on_fetch(url):
        return fetch_yt_info(ctx, url)

    def on_extract(url, start_str, end_str, video_state, progress=gr.Progress()):
        return extract_yt_clip(ctx, url, start_str, end_str, video_state, progress)

    def on_transcribe(clip_audio):
        yield from transcribe_yt_clip(ctx, clip_audio)

    def on_clone(text, ref_audio, transcript, language, voice_name, trim_ref):
        return clone_yt_voice(ctx, text, ref_audio, transcript, language, voice_name,
                              trim_ref=trim_ref)

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
    wire_stop(ctx, t.yt_stop_btn, t.yt_status)
    wire_run_lifecycle(
        t.yt_transcribe_btn, t.yt_stop_btn, on_transcribe,
        inputs=[t.yt_clip_audio],
        outputs=[t.yt_transcript, t.yt_status],
    )
    wire_run_lifecycle(
        t.yt_clone_btn, t.yt_stop_btn, on_clone,
        inputs=[t.yt_text, t.yt_clip_audio, t.yt_transcript, t.yt_language, t.yt_voice_name,
                t.yt_trim_ref],
        outputs=[t.yt_audio_out, ui.status, t.yt_status,
                 ui.vc.vc_library_voice, ui.lib.lib_table],
    )
