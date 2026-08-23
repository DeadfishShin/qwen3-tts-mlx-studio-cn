"""Script Mode tab: multi-speaker scripts with per-speaker voice assignment."""
import types

import gradio as gr

from config import DEFAULT_SCRIPT_SILENCE_MS, DEFAULT_SPEAKERS, LANGUAGE_AUTO, MAX_SCRIPT_SPEAKERS
from audio_utils import concatenate_audio
from generation import (
    GenerationCancelled, GenRequest, api_language, generate_once, save_audio,
)
from script_parser import group_by_model_type, parse_script
from ui import strings as S
from ui.components import format_table_md, voice_choices, wire_run_lifecycle, wire_stop


def build(ctx):
    with gr.Tab(S.TAB_SCRIPT) as script_tab:
        gr.HTML(S.SM_NOTICE_HTML)
        with gr.Row():
            with gr.Column(scale=2):
                sm_script = gr.Textbox(
                    label=S.SM_SCRIPT,
                    lines=10,
                    placeholder=S.SM_SCRIPT_PLACEHOLDER,
                    elem_classes=["script-editor"],
                )
                with gr.Row():
                    sm_parse_btn = gr.Button(S.SM_PARSE, variant="primary", scale=1)
                    sm_silence = gr.Slider(
                        0, 2000, value=DEFAULT_SCRIPT_SILENCE_MS, step=50,
                        label=S.SM_SILENCE, scale=2,
                    )
                sm_parse_status = gr.Textbox(label=S.SM_PARSE_RESULT, interactive=False, lines=2)

                # Voice assignment state
                sm_assignments = gr.State({})

                gr.Markdown(S.SM_ASSIGNMENTS_HEADER)
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
                                S.SM_MODE_CHOICES,
                                value="custom_voice",
                                label=S.SM_SLOT_MODE.format(n=i + 1),
                                scale=2,
                            )
                            sm_speaker_modes.append(mode)
                            lang = gr.Dropdown(
                                choices=S.LANGUAGE_CHOICES,
                                value=LANGUAGE_AUTO,
                                label=S.LANGUAGE, scale=1,
                            )
                            sm_speaker_languages.append(lang)
                        with gr.Row():
                            spk = gr.Dropdown(
                                choices=DEFAULT_SPEAKERS,
                                value=DEFAULT_SPEAKERS[0],
                                label=S.CV_SPEAKER,
                                scale=1,
                            )
                            sm_speaker_speakers.append(spk)
                            inst = gr.Textbox(
                                label=S.SM_SLOT_INSTRUCT,
                                placeholder=S.SM_SLOT_INSTRUCT_PLACEHOLDER,
                                scale=2,
                            )
                            sm_speaker_instructs.append(inst)
                            lib_v = gr.Dropdown(
                                choices=voice_choices(ctx),
                                value="None",
                                label=S.SM_SLOT_LIBRARY,
                                scale=1,
                            )
                            sm_speaker_lib_voices.append(lib_v)

                with gr.Row():
                    sm_generate_btn = gr.Button(S.SM_GENERATE, variant="primary")
                    sm_stop_btn = gr.Button(S.STOP, variant="stop", visible=False)
                    sm_save_btn = gr.Button(S.SAVE_COMBINED)
                with gr.Accordion(S.SM_BREAKDOWN, open=False):
                    sm_table = gr.Markdown(value=S.SM_TABLE_EMPTY)

            with gr.Column(scale=1):
                sm_audio = gr.Audio(label=S.COMBINED_OUTPUT, type="numpy", interactive=False, buttons=["download"])
                sm_status = gr.Textbox(label=S.SM_STATUS, interactive=False)
    return types.SimpleNamespace(
        script_tab=script_tab, sm_script=sm_script, sm_parse_btn=sm_parse_btn,
        sm_silence=sm_silence, sm_parse_status=sm_parse_status,
        sm_assignments=sm_assignments,
        sm_speaker_groups=sm_speaker_groups, sm_speaker_modes=sm_speaker_modes,
        sm_speaker_speakers=sm_speaker_speakers,
        sm_speaker_instructs=sm_speaker_instructs,
        sm_speaker_languages=sm_speaker_languages,
        sm_speaker_lib_voices=sm_speaker_lib_voices,
        sm_generate_btn=sm_generate_btn, sm_stop_btn=sm_stop_btn,
        sm_save_btn=sm_save_btn,
        sm_table=sm_table, sm_audio=sm_audio, sm_status=sm_status,
    )


def _script_secs(audio_by_line_number):
    """Total seconds of audio generated so far across the whole script."""
    return sum(len(a) / s for (s, a) in
               [r for r in audio_by_line_number.values() if r is not None])


def _localize_parse_error(error):
    if error.startswith("No valid lines found in script"):
        return S.SM_NO_VALID_LINES
    if error.startswith("Too many speakers ("):
        count = error.split("(", 1)[1].split(")", 1)[0]
        max_count = error.rsplit("Maximum is ", 1)[-1].rstrip(".")
        return S.SM_TOO_MANY_SPEAKERS.format(count=count, max_count=max_count)
    return S.SM_PARSE_ERROR.format(err=error)


def _generate_clone_lines_batched(ctx, lines, assignments_state, audio_by_line_number,
                                  done, total_lines, progress):
    """Generate Voice Clone script lines, batching lines that share a library voice.

    Script clone lines always use library refs as saved (no trim). Generator:
    yields a status string after each completed chunk/line; returns
    (succeeded_delta, failed_delta, done, cancelled).
    """
    succeeded = failed = 0
    cancelled = False

    def line_status():
        return S.SCRIPT_LINE_PROGRESS.format(
            done=done, total=total_lines, secs=_script_secs(audio_by_line_number))

    def fail_line(line):
        nonlocal failed, done
        audio_by_line_number[line.line_number] = None
        failed += 1
        done += 1
        progress(done / total_lines)

    def run_line_single(line, ref_audio_path, ref_text, lang):
        nonlocal succeeded, done, cancelled
        try:
            sr, audio = generate_once(ctx, GenRequest(
                mode="voice_clone", text=line.text, language=lang,
                ref_audio=ref_audio_path, ref_text=ref_text, trim_ref=False))
            audio_by_line_number[line.line_number] = (sr, audio)
            succeeded += 1
            done += 1
            progress(done / total_lines, desc=S.SM_CLONE_PROGRESS)
        except GenerationCancelled:
            cancelled = True
        except Exception:
            fail_line(line)

    # Group by (library voice, language) — one shared reference per batched call
    groups = {}
    for line in lines:
        assignment = assignments_state.get(line.speaker, {})
        key = (assignment.get("library_voice", ""), assignment.get("language", "English"))
        groups.setdefault(key, []).append(line)

    batch_size = ctx.settings.batch_size
    for (lib_voice, lang), group_lines in groups.items():
        if cancelled or ctx.cancel_event.is_set():
            cancelled = True
            break
        if not lib_voice or lib_voice == "None":
            for line in group_lines:
                fail_line(line)
            yield line_status()
            continue
        try:
            voice = ctx.library.load_voice(lib_voice)
            ref_audio_path = ctx.library.get_ref_audio_path(lib_voice)
            ref_text = voice["ref_text"]
        except FileNotFoundError:
            for line in group_lines:
                fail_line(line)
            yield line_status()
            continue

        if batch_size > 1 and len(group_lines) > 1:
            for chunk_start in range(0, len(group_lines), batch_size):
                if cancelled or ctx.cancel_event.is_set():
                    cancelled = True
                    break
                chunk = group_lines[chunk_start:chunk_start + batch_size]
                try:
                    results = ctx.engine.batch_generate_voice_clone(
                        [l.text for l in chunk], ref_audio_path, ref_text,
                        api_language(lang),
                        denoise_ref=ctx.settings.denoise_ref,
                        **ctx.settings.gen_kwargs(),
                    )
                    for j, (sr, audio) in enumerate(results):
                        audio_by_line_number[chunk[j].line_number] = (sr, audio)
                        succeeded += 1
                        done += 1
                        progress(done / total_lines, desc=S.SM_CLONE_PROGRESS)
                except Exception:
                    # Batch failed — retry each line individually
                    for line in chunk:
                        if cancelled or ctx.cancel_event.is_set():
                            cancelled = True
                            break
                        run_line_single(line, ref_audio_path, ref_text, lang)
                yield line_status()
        else:
            for line in group_lines:
                if cancelled or ctx.cancel_event.is_set():
                    cancelled = True
                    break
                run_line_single(line, ref_audio_path, ref_text, lang)
                yield line_status()

    return succeeded, failed, done, cancelled


def generate_script_handler(ctx, raw_text, assignments_state, silence_ms, progress=gr.Progress()):
    """Generate audio for a parsed multi-speaker script.

    Generator yielding (audio_update, table_md, status) with live per-line
    progress. Cancel is checked between lines/batched calls; on Stop the
    completed lines are combined but NOT recorded to history.

    assignments_state is a dict mapping speaker name to voice config:
      {speaker: {"mode": ..., "speaker": ..., "language": ..., "instruct": ..., "library_voice": ...}}
    """
    if not raw_text.strip():
        gr.Warning(S.SM_ENTER_SCRIPT_WARN)
        yield None, S.SM_ENTER_SCRIPT_MD, S.SM_ENTER_SCRIPT
        return

    parsed = parse_script(raw_text)
    if parsed.errors:
        parse_error = _localize_parse_error(parsed.errors[0])
        gr.Warning(parse_error)
        yield None, f"*{parse_error}*", parse_error
        return

    if not assignments_state:
        gr.Warning(S.SM_PARSE_FIRST_WARN)
        yield None, S.SM_PARSE_FIRST_MD, S.SM_PARSE_FIRST
        return

    ctx.cancel_event.clear()

    # Group lines by model type for efficient model swapping
    groups = group_by_model_type(parsed.lines, assignments_state)

    audio_by_line_number = {}  # line_number -> (sr, audio)
    table_rows = []
    succeeded, failed = 0, 0

    total_lines = len(parsed.lines)
    done = 0
    cancelled = False

    def line_status():
        return S.SCRIPT_LINE_PROGRESS.format(
            done=done, total=total_lines, secs=_script_secs(audio_by_line_number))

    for model_type, lines in groups.items():
        if cancelled or ctx.cancel_event.is_set():
            cancelled = True
            break
        label = S.SM_MODE_LABELS.get(model_type, model_type)
        batch_size = ctx.settings.batch_size

        if model_type == "base":
            inner = _generate_clone_lines_batched(
                ctx, lines, assignments_state, audio_by_line_number,
                done, total_lines, progress)
            while True:
                try:
                    status = next(inner)
                    yield gr.skip(), gr.skip(), status
                except StopIteration as stop:
                    s, f, done, was_cancelled = stop.value
                    break
            succeeded += s
            failed += f
            cancelled = cancelled or was_cancelled
        elif model_type in ("custom_voice", "voice_design") and batch_size > 1:
            # Batch generation for Custom Voice and Voice Design
            for batch_start in range(0, len(lines), batch_size):
                if cancelled or ctx.cancel_event.is_set():
                    cancelled = True
                    break
                batch_lines = lines[batch_start:batch_start + batch_size]
                texts = [l.text for l in batch_lines]

                try:
                    if model_type == "custom_voice":
                        speakers = []
                        instructs = []
                        lang = None
                        for line in batch_lines:
                            assignment = assignments_state.get(line.speaker, {})
                            lang = api_language(assignment.get("language", "English"))
                            speakers.append(assignment.get("speaker", DEFAULT_SPEAKERS[0]))
                            instructs.append(assignment.get("instruct", ""))

                        results = ctx.engine.batch_generate_custom_voice(
                            texts, speakers, lang, instructs,
                            **ctx.settings.gen_kwargs(),
                        )
                    else:  # voice_design
                        instructs = []
                        lang = None
                        for line in batch_lines:
                            assignment = assignments_state.get(line.speaker, {})
                            lang = api_language(assignment.get("language", "English"))
                            instructs.append(assignment.get("instruct", ""))

                        results = ctx.engine.batch_generate_voice_design(
                            texts, lang, instructs,
                            **ctx.settings.gen_kwargs(),
                        )

                    for j, (sr, audio) in enumerate(results):
                        audio_by_line_number[batch_lines[j].line_number] = (sr, audio)
                        succeeded += 1
                        done += 1
                        progress(done / total_lines, desc=S.SM_PROGRESS.format(label=label))

                except Exception:
                    # Batch failed — retry each line individually
                    for line in batch_lines:
                        if cancelled or ctx.cancel_event.is_set():
                            cancelled = True
                            break
                        assignment = assignments_state.get(line.speaker, {})
                        try:
                            sr, audio = generate_once(ctx, GenRequest(
                                mode=model_type, text=line.text,
                                language=assignment.get("language", "English"),
                                speaker=assignment.get("speaker", DEFAULT_SPEAKERS[0]),
                                instruct=assignment.get("instruct", "")))
                            audio_by_line_number[line.line_number] = (sr, audio)
                            succeeded += 1
                        except GenerationCancelled:
                            cancelled = True
                            break
                        except Exception:
                            audio_by_line_number[line.line_number] = None
                            failed += 1
                        done += 1
                        progress(done / total_lines)
                yield gr.skip(), gr.skip(), line_status()
        else:
            # Sequential generation (Custom Voice / Voice Design at batch_size == 1)
            for line in lines:
                if cancelled or ctx.cancel_event.is_set():
                    cancelled = True
                    break
                assignment = assignments_state.get(line.speaker, {})
                mode = assignment.get("mode", "custom_voice")

                try:
                    if mode not in ("custom_voice", "voice_design"):
                        raise ValueError(S.SM_UNKNOWN_SPEAKER_MODE.format(mode=mode))
                    sr, audio = generate_once(ctx, GenRequest(
                        mode=mode, text=line.text,
                        language=assignment.get("language", "English"),
                        speaker=assignment.get("speaker", DEFAULT_SPEAKERS[0]),
                        instruct=assignment.get("instruct", "")))
                    audio_by_line_number[line.line_number] = (sr, audio)
                    succeeded += 1
                except GenerationCancelled:
                    cancelled = True
                    break
                except Exception:
                    audio_by_line_number[line.line_number] = None
                    failed += 1

                done += 1
                progress(done / total_lines)
                yield gr.skip(), gr.skip(), line_status()

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
        elif line.line_number in audio_by_line_number:
            table_rows.append([str(line.line_number), line.speaker, preview, S.SM_FAILED])
        else:
            # never attempted — only happens when the run was stopped
            table_rows.append([str(line.line_number), line.speaker, preview,
                               S.SM_STOPPED if cancelled else S.SM_FAILED])

    table_md = format_table_md(S.SM_TABLE_HEADERS, table_rows, S.SM_NO_RESULTS)

    if not audio_segments:
        yield None, table_md, (S.SM_STOPPED if cancelled else S.SM_ALL_FAILED)
        return

    combined = concatenate_audio(audio_segments, silence_ms=int(silence_ms))
    if cancelled:
        # Partial script kept for listening/manual save; not recorded to history.
        yield gr.update(value=combined), table_md, S.SCRIPT_STOPPED.format(
            done=succeeded, total=total_lines)
        return

    # Record to history
    speakers_used = ", ".join(parsed.speakers[:4])
    if len(parsed.speakers) > 4:
        speakers_used += "..."
    ctx.history.add(
        mode="custom_voice",
        text=f"[Script: {succeeded} lines, speakers: {speakers_used}]",
        language="Multi", audio=combined,
        voice_params=S.SM_HISTORY_PARAMS,
    )

    status_msg = S.SM_GENERATED.format(done=succeeded, total=total_lines)
    if failed:
        status_msg += S.SM_FAILURE_SUFFIX.format(count=failed)
    if ctx.settings.denoise_ref:
        status_msg += S.SM_NOISE_SUFFIX
    yield gr.update(value=combined), table_md, status_msg


def wire(ctx, ui):
    t = ui.sm

    def _parse_and_update_slots(raw_text):
        """Parse script and update speaker slot visibility."""
        if not raw_text.strip():
            gr.Warning(S.SM_ENTER_SCRIPT_WARN)
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, S.SM_ENTER_SCRIPT, {}

        parsed = parse_script(raw_text)

        if parsed.errors:
            parse_error = _localize_parse_error(parsed.errors[0])
            gr.Warning(parse_error)
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, parse_error, {}

        # Build summary
        summary_parts = [S.SM_SUMMARY_HEAD.format(
            speakers=len(parsed.speakers), lines=len(parsed.lines))]
        lines_per_speaker = {}
        for line in parsed.lines:
            lines_per_speaker.setdefault(line.speaker, 0)
            lines_per_speaker[line.speaker] += 1
        for spk in parsed.speakers:
            count = lines_per_speaker.get(spk, 0)
            summary_parts.append(S.SM_SUMMARY_LINE.format(speaker=spk, count=count))

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
                "language": LANGUAGE_AUTO,
                "instruct": "",
                "library_voice": "None",
            }

        return *updates, "\n".join(summary_parts), assignments

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
            "custom_voice": "custom_voice",
            "voice_design": "voice_design",
            "voice_clone": "voice_clone",
        }
        for i, spk in enumerate(parsed.speakers):
            if i >= MAX_SCRIPT_SPEAKERS:
                break
            base = i * values_per_slot
            mode_label = slot_values[base] if base < len(slot_values) else "custom_voice"
            assignments[spk] = {
                "mode": mode_map.get(mode_label, "custom_voice"),
                "speaker": slot_values[base + 1] if base + 1 < len(slot_values) else DEFAULT_SPEAKERS[0],
                "instruct": slot_values[base + 2] if base + 2 < len(slot_values) else "",
                "language": slot_values[base + 3] if base + 3 < len(slot_values) else LANGUAGE_AUTO,
                "library_voice": slot_values[base + 4] if base + 4 < len(slot_values) else "None",
            }

        return assignments

    def _generate_script_with_assignments(raw_text, assignments, silence_ms, *slot_values, progress=gr.Progress()):
        """Build fresh assignments from slot values, then generate."""
        fresh = _build_assignments_from_slots(assignments, raw_text, *slot_values)
        yield from generate_script_handler(ctx, raw_text, fresh, silence_ms, progress)

    def _refresh_script_lib_voices():
        choices = voice_choices(ctx)
        return [gr.update(choices=choices) for _ in range(MAX_SCRIPT_SPEAKERS)]

    # Collect all slot control components in order
    all_slot_controls = []
    for i in range(MAX_SCRIPT_SPEAKERS):
        all_slot_controls.extend([
            t.sm_speaker_modes[i],
            t.sm_speaker_speakers[i],
            t.sm_speaker_instructs[i],
            t.sm_speaker_languages[i],
            t.sm_speaker_lib_voices[i],
        ])

    t.sm_parse_btn.click(
        fn=_parse_and_update_slots,
        inputs=[t.sm_script],
        outputs=[*t.sm_speaker_groups, t.sm_parse_status, t.sm_assignments],
    )
    wire_stop(ctx, t.sm_stop_btn, t.sm_status)
    wire_run_lifecycle(
        t.sm_generate_btn, t.sm_stop_btn, _generate_script_with_assignments,
        inputs=[t.sm_script, t.sm_assignments, t.sm_silence, *all_slot_controls],
        outputs=[t.sm_audio, t.sm_table, t.sm_status],
        show_progress="full",
    )
    t.sm_save_btn.click(
        fn=lambda audio: save_audio(ctx, audio, "script"),
        inputs=[t.sm_audio],
        outputs=[t.sm_status],
    )
    t.script_tab.select(
        fn=_refresh_script_lib_voices,
        outputs=t.sm_speaker_lib_voices,
    )
