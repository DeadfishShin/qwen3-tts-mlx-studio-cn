"""Voice Library tab: browse, preview, rename, delete, and import saved voices."""
import os
import types

import gradio as gr

from ui import strings as S
from ui.components import voice_table


def build(ctx):
    with gr.Tab(S.TAB_LIBRARY):
        with gr.Row():
            with gr.Column(scale=2):
                lib_table = gr.Markdown(value=voice_table(ctx), label=S.LIB_TABLE_LABEL)
                with gr.Row():
                    lib_selected = gr.Textbox(
                        label=S.LIB_SELECTED,
                        placeholder=S.LIB_SELECTED_PLACEHOLDER,
                        scale=2,
                    )
                    lib_preview_btn = gr.Button(S.LIB_PREVIEW, scale=1)
                    lib_delete_btn = gr.Button(S.LIB_DELETE, scale=1)
                with gr.Row():
                    lib_new_name = gr.Textbox(
                        label=S.LIB_RENAME_TO, placeholder=S.LIB_RENAME_PLACEHOLDER, scale=2
                    )
                    lib_rename_btn = gr.Button(S.LIB_RENAME, scale=1)
                lib_preview_audio = gr.Audio(label=S.LIB_PREVIEW_AUDIO, buttons=["download"])
                lib_status = gr.Textbox(
                    show_label=False, interactive=False,
                    placeholder=S.LIB_STATUS_TEXT_PLACEHOLDER,
                    elem_classes=["save-status-text"],
                )
            with gr.Column(scale=1, elem_classes=["output-col"]):
                gr.Markdown(S.LIB_IMPORT_HEADER)
                lib_import_audio = gr.Audio(
                    label=S.VC_REF_AUDIO, type="filepath", buttons=["download"]
                )
                lib_import_transcript = gr.Textbox(
                    label=S.LIB_IMPORT_TRANSCRIPT, lines=3
                )
                lib_import_name = gr.Textbox(
                    label=S.LIB_IMPORT_NAME, placeholder=S.LIB_IMPORT_NAME_PLACEHOLDER
                )
                lib_import_language = gr.Dropdown(
                    choices=S.LANGUAGE_CHOICES[1:], value="English", label=S.LANGUAGE
                )
                lib_import_btn = gr.Button(S.LIB_IMPORT, variant="primary")
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
        return voice_table(ctx), S.LIB_SELECT_FIRST
    if not ctx.library.delete_voice(voice_name):
        return voice_table(ctx), S.LIB_VOICE_NOT_FOUND.format(name=voice_name)
    return voice_table(ctx), S.LIB_DELETED.format(name=voice_name)


def rename_voice(ctx, old_name, new_name):
    if not old_name or old_name == "(empty)":
        return voice_table(ctx), S.LIB_SELECT_FIRST
    if not new_name.strip():
        return voice_table(ctx), S.LIB_ENTER_NEW_NAME
    ok = ctx.library.rename_voice(old_name, new_name.strip())
    if ok:
        return voice_table(ctx), S.LIB_RENAMED.format(old=old_name, new=new_name.strip())
    return voice_table(ctx), S.LIB_RENAME_FAILED


def import_voice(ctx, audio_path, transcript, name, language):
    if not audio_path:
        gr.Warning(S.LIB_IMPORT_NO_AUDIO_WARN)
        return voice_table(ctx), S.LIB_IMPORT_NO_AUDIO
    if not name.strip():
        gr.Warning(S.LIB_IMPORT_NO_NAME_WARN)
        return voice_table(ctx), S.LIB_IMPORT_NO_NAME
    if not transcript or not transcript.strip():
        gr.Warning(S.LIB_IMPORT_NO_TRANSCRIPT_WARN)
        return voice_table(ctx), S.LIB_IMPORT_NO_TRANSCRIPT
    ctx.library.save_voice(
        name=name.strip(),
        ref_audio_path=audio_path,
        ref_text=transcript.strip(),
        language=language,
        description=S.LIB_IMPORTED_DESCRIPTION,
        source="import",
    )
    return voice_table(ctx), S.LIB_IMPORTED.format(name=name.strip())


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
