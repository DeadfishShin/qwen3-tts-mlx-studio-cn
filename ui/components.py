"""Shared UI builders and table formatters used by multiple tabs."""
import types

import gradio as gr

from config import DEFAULT_BATCH_SPLIT_MODE, DEFAULT_SILENCE_GAP_MS
from ui import strings as S


def format_table_md(headers, rows, empty_msg=None):
    """Format rows as a Markdown table. Avoids gr.Dataframe AG Grid recursion bug."""
    if empty_msg is None:
        empty_msg = S.NO_ENTRIES
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


def voice_choices(ctx):
    """Return list of saved voice names for dropdowns."""
    return ["None"] + [v["name"] for v in ctx.library.list_voices()]


def voice_table(ctx):
    """Return voice data as a Markdown table string."""
    voices = ctx.library.list_voices()
    rows = [
        [v["name"], v.get("source", ""), v.get("language", ""), v.get("description", "")]
        for v in voices
    ]
    return format_table_md(
        ["Name", "Source", "Language", "Description"],
        rows,
        S.NO_VOICES,
    )


def history_table_md(ctx):
    """Return history data formatted as a Markdown table string."""
    return format_table_md(
        ["ID", "Time", "Mode", "Text", "Duration"],
        ctx.history.table_data(),
        S.NO_HISTORY,
    )


def build_batch_accordion():
    """The Batch Mode accordion shared by Custom Voice / Voice Design / Voice Clone."""
    with gr.Accordion(S.BATCH_ACCORDION, open=False, elem_classes=["batch-accordion"]):
        with gr.Row():
            batch_split = gr.Radio(
                ["paragraph", "sentence", "line"],
                value=DEFAULT_BATCH_SPLIT_MODE,
                label=S.SPLIT_MODE,
            )
            batch_silence = gr.Slider(
                0, 2000, value=DEFAULT_SILENCE_GAP_MS, step=50,
                label=S.SILENCE_GAP,
            )
        batch_generate = gr.Button(S.GENERATE_BATCH, variant="primary")
        batch_table = gr.Dataframe(
            headers=["#", "Text", "Status"],
            value=[["", "", ""]],
            label=S.BATCH_RESULTS,
            interactive=False,
        )
        batch_audio = gr.Audio(label=S.COMBINED_OUTPUT, type="numpy", interactive=False, buttons=["download"])
        with gr.Row():
            batch_save = gr.Button(S.SAVE_COMBINED)
            batch_status = gr.Textbox(label=S.BATCH_STATUS, interactive=False)
    return types.SimpleNamespace(
        batch_split=batch_split, batch_silence=batch_silence,
        batch_generate=batch_generate, batch_table=batch_table,
        batch_audio=batch_audio, batch_save=batch_save, batch_status=batch_status,
    )


def build_output_column():
    """The right-hand output column shared by the three generation tabs.

    A regular (non-streaming) player: waveform view, smooth playback. The
    complete or Stop-kept-partial waveform lands here when the run ends.
    """
    with gr.Column(scale=1, elem_classes=["output-col"]):
        audio = gr.Audio(label=S.OUTPUT, type="numpy", interactive=False, buttons=["download"])
        stop = gr.Button(S.STOP, variant="stop", visible=False)
        save = gr.Button(S.SAVE_AUDIO)
        save_status = gr.Textbox(
            show_label=False, interactive=False,
            placeholder=S.SAVE_PATH_PLACEHOLDER,
            elem_classes=["save-status-text"],
        )
    return types.SimpleNamespace(audio=audio, stop=stop, save=save, save_status=save_status)


def wire_run_lifecycle(start_btn, stop_btn, fn, inputs, outputs, show_progress="minimal"):
    """start → (disable start, show stop) → run fn → restore. Stop wiring is separate."""
    begin = start_btn.click(
        fn=lambda: (gr.update(interactive=False), gr.update(visible=True)),
        outputs=[start_btn, stop_btn], queue=False)
    run = begin.then(fn=fn, inputs=inputs, outputs=outputs, show_progress=show_progress)
    run.then(fn=lambda: (gr.update(interactive=True), gr.update(visible=False)),
             outputs=[start_btn, stop_btn], queue=False)
    return run


def wire_stop(ctx, stop_btn, status_out):
    """A Stop click sets the shared cancel event; runners notice between chunks."""
    stop_btn.click(fn=lambda: (ctx.cancel_event.set(), S.STOPPING)[1],
                   outputs=[status_out], queue=False)


def build_lib_save_accordion(name_placeholder):
    """The Save to Voice Library accordion shared by Voice Design / Voice Clone."""
    with gr.Accordion(S.LIB_SAVE_ACCORDION, open=False, elem_classes=["lib-save-accordion"]):
        with gr.Row():
            lib_name = gr.Textbox(
                label=S.VOICE_NAME, placeholder=name_placeholder, scale=2
            )
            lib_save = gr.Button(S.SAVE_VOICE_TO_LIBRARY, scale=1)
        lib_status = gr.Textbox(
            show_label=False, interactive=False,
            placeholder=S.LIB_STATUS_PLACEHOLDER,
            elem_classes=["save-status-text"],
        )
    return types.SimpleNamespace(lib_name=lib_name, lib_save=lib_save, lib_status=lib_status)
