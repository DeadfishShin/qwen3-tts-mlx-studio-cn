"""History tab: browse, replay, and manage past generations."""
import types

import gradio as gr

from generation import save_audio
from ui import strings as S
from ui.components import history_table_md


def build(ctx):
    with gr.Tab(S.TAB_HISTORY):
        hist_table = gr.Markdown(value=history_table_md(ctx), label=S.HIST_TABLE_LABEL, elem_classes=["history-table"])
        with gr.Row():
            hist_selected = gr.Textbox(
                label=S.HIST_ENTRY_ID,
                placeholder=S.HIST_ENTRY_PLACEHOLDER,
                scale=3,
            )
            hist_preview_btn = gr.Button(S.HIST_PREVIEW, scale=1)
            hist_delete_btn = gr.Button(S.HIST_DELETE, scale=1)
        with gr.Row():
            hist_save_btn = gr.Button(S.SAVE_AUDIO, scale=1)
            hist_regen_btn = gr.Button(S.HIST_VIEW_SETTINGS, scale=1)
            hist_clear_btn = gr.Button(S.HIST_CLEAR, scale=1)
        hist_status = gr.Textbox(
            show_label=False, interactive=False,
            placeholder=S.HIST_STATUS_PLACEHOLDER,
            elem_classes=["save-status-text"],
        )
        with gr.Row():
            hist_audio = gr.Audio(
                label=S.OUTPUT, type="numpy", interactive=False, scale=1, buttons=["download"]
            )
            hist_regen_info = gr.Textbox(
                label=S.HIST_SETTINGS_LABEL, interactive=False, lines=3, scale=1
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
        gr.Warning(S.HIST_AUDIO_MISSING_WARN)
        return None
    return audio


def history_delete(ctx, entry_id):
    """Delete a single history entry."""
    if not entry_id or entry_id == "(empty)":
        return history_table_md(ctx), S.HIST_SELECT_FIRST
    ctx.history.delete_entry(entry_id)
    return history_table_md(ctx), S.HIST_DELETED.format(entry_id=entry_id)


def history_clear(ctx):
    """Clear all history."""
    ctx.history.clear()
    return history_table_md(ctx), S.HIST_CLEARED


def history_save_audio(ctx, entry_id):
    """Save a history entry's audio to the output directory."""
    if not entry_id or entry_id == "(empty)":
        return S.HIST_SELECT_FIRST
    audio = ctx.history.get_audio(entry_id)
    if audio is None:
        return S.HIST_AUDIO_NOT_FOUND
    return save_audio(ctx, audio, "history")


def history_regenerate(ctx, entry_id):
    """Get params from a history entry for regeneration."""
    if not entry_id or entry_id == "(empty)":
        return S.HIST_SELECT_FIRST
    entry = ctx.history.get_entry(entry_id)
    if entry is None:
        return S.HIST_NOT_FOUND
    parts = [
        S.HIST_MODE.format(mode=S.SM_MODE_LABELS.get(entry.mode, entry.mode)),
        S.HIST_LANGUAGE.format(language=entry.language),
    ]
    if entry.speaker:
        parts.append(S.HIST_VOICE.format(speaker=entry.speaker))
    if getattr(entry, "voice_description", ""):
        parts.append(S.HIST_VOICE_DESCRIPTION.format(description=entry.voice_description))
    if getattr(entry, "style_instruction", ""):
        parts.append(S.HIST_STYLE_INSTRUCTION.format(style=entry.style_instruction))
    if entry.voice_params and not getattr(entry, "voice_description", ""):
        parts.append(S.HIST_PARAMS.format(params=entry.voice_params))
    parts.append(S.HIST_TEXT.format(text=entry.text))
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
