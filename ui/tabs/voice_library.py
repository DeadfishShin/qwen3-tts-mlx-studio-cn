"""Voice Library tab: browse, preview, rename, delete, and import saved voices."""
import os
import types

import gradio as gr

from config import LANGUAGES
from ui.components import voice_table


def build(ctx):
    with gr.Tab("Voice Library"):
        with gr.Row():
            with gr.Column(scale=2):
                lib_table = gr.Markdown(value=voice_table(ctx), label="Saved Voices")
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
    return types.SimpleNamespace(
        lib_table=lib_table, lib_selected=lib_selected,
        lib_preview_btn=lib_preview_btn, lib_delete_btn=lib_delete_btn,
        lib_new_name=lib_new_name, lib_rename_btn=lib_rename_btn,
        lib_preview_audio=lib_preview_audio, lib_status=lib_status,
        lib_import_audio=lib_import_audio,
        lib_import_transcript=lib_import_transcript,
        lib_import_name=lib_import_name,
        lib_import_language=lib_import_language,
        lib_import_btn=lib_import_btn,
    )


def preview_voice(ctx, voice_name):
    if not voice_name or voice_name == "(empty)":
        return None
    try:
        path = ctx.library.get_ref_audio_path(voice_name)
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


def delete_voice(ctx, voice_name):
    if not voice_name or voice_name == "(empty)":
        return voice_table(ctx), "Select a voice first"
    if not ctx.library.delete_voice(voice_name):
        return voice_table(ctx), f"Voice '{voice_name}' not found"
    return voice_table(ctx), f"Deleted '{voice_name}'"


def rename_voice(ctx, old_name, new_name):
    if not old_name or old_name == "(empty)":
        return voice_table(ctx), "Select a voice first"
    if not new_name.strip():
        return voice_table(ctx), "Enter a new name"
    ok = ctx.library.rename_voice(old_name, new_name.strip())
    if ok:
        return voice_table(ctx), f"Renamed '{old_name}' to '{new_name.strip()}'"
    return voice_table(ctx), f"Rename failed (name may already exist)"


def import_voice(ctx, audio_path, transcript, name, language):
    if not audio_path:
        gr.Warning("Upload audio to import.")
        return voice_table(ctx), "Upload audio first"
    if not name.strip():
        gr.Warning("Enter a name for the imported voice.")
        return voice_table(ctx), "Enter a name"
    if not transcript or not transcript.strip():
        gr.Warning("Transcript required for imported voice.")
        return voice_table(ctx), "Enter transcript"
    ctx.library.save_voice(
        name=name.strip(),
        ref_audio_path=audio_path,
        ref_text=transcript.strip(),
        language=language,
        description="Imported voice",
        source="import",
    )
    return voice_table(ctx), f"Imported '{name.strip()}'"


def wire(ctx, ui):
    t = ui.lib

    t.lib_preview_btn.click(
        fn=lambda name: preview_voice(ctx, name),
        inputs=[t.lib_selected],
        outputs=[t.lib_preview_audio],
    )
    t.lib_delete_btn.click(
        fn=lambda name: delete_voice(ctx, name),
        inputs=[t.lib_selected],
        outputs=[t.lib_table, t.lib_status],
    )
    t.lib_rename_btn.click(
        fn=lambda old, new: rename_voice(ctx, old, new),
        inputs=[t.lib_selected, t.lib_new_name],
        outputs=[t.lib_table, t.lib_status],
    )
    t.lib_import_btn.click(
        fn=lambda audio, transcript, name, lang: import_voice(ctx, audio, transcript, name, lang),
        inputs=[t.lib_import_audio, t.lib_import_transcript,
                t.lib_import_name, t.lib_import_language],
        outputs=[t.lib_table, t.lib_status],
    )
