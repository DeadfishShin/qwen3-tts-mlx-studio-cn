"""History tab: browse, replay, and manage past generations."""
import types

import gradio as gr

from generation import save_audio
from ui.components import history_table_md


def build(ctx):
    with gr.Tab("History"):
        hist_table = gr.Markdown(value=history_table_md(ctx), label="Generation History", elem_classes=["history-table"])
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
    return types.SimpleNamespace(
        hist_table=hist_table, hist_selected=hist_selected,
        hist_preview_btn=hist_preview_btn, hist_delete_btn=hist_delete_btn,
        hist_save_btn=hist_save_btn, hist_regen_btn=hist_regen_btn,
        hist_clear_btn=hist_clear_btn, hist_status=hist_status,
        hist_audio=hist_audio, hist_regen_info=hist_regen_info,
    )


def history_preview(ctx, entry_id):
    """Load audio for a history entry."""
    if not entry_id or entry_id == "(empty)":
        return None
    audio = ctx.history.get_audio(entry_id)
    if audio is None:
        gr.Warning("Audio not found for this entry.")
        return None
    return audio


def history_delete(ctx, entry_id):
    """Delete a single history entry."""
    if not entry_id or entry_id == "(empty)":
        return history_table_md(ctx), "Select an entry first"
    ctx.history.delete_entry(entry_id)
    return history_table_md(ctx), f"Deleted entry {entry_id}"


def history_clear(ctx):
    """Clear all history."""
    ctx.history.clear()
    return history_table_md(ctx), "History cleared"


def history_save_audio(ctx, entry_id):
    """Save a history entry's audio to the output directory."""
    if not entry_id or entry_id == "(empty)":
        return "Select an entry first"
    audio = ctx.history.get_audio(entry_id)
    if audio is None:
        return "Audio not found"
    return save_audio(ctx, audio, "history")


def history_regenerate(ctx, entry_id):
    """Get params from a history entry for regeneration."""
    if not entry_id or entry_id == "(empty)":
        return "Select an entry first"
    entry = ctx.history.get_entry(entry_id)
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


def wire(ctx, ui):
    t = ui.hist

    t.hist_preview_btn.click(
        fn=lambda entry_id: history_preview(ctx, entry_id),
        inputs=[t.hist_selected],
        outputs=[t.hist_audio],
    )
    t.hist_delete_btn.click(
        fn=lambda entry_id: history_delete(ctx, entry_id),
        inputs=[t.hist_selected],
        outputs=[t.hist_table, t.hist_status],
    )
    t.hist_clear_btn.click(
        fn=lambda: history_clear(ctx),
        outputs=[t.hist_table, t.hist_status],
    )
    t.hist_save_btn.click(
        fn=lambda entry_id: history_save_audio(ctx, entry_id),
        inputs=[t.hist_selected],
        outputs=[t.hist_status],
    )
    t.hist_regen_btn.click(
        fn=lambda entry_id: history_regenerate(ctx, entry_id),
        inputs=[t.hist_selected],
        outputs=[t.hist_regen_info],
    )
