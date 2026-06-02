# Hide Script Mode results table behind an accordion

**Date:** 2026-06-02
**Scope:** Script Mode tab only. Batch tables (Custom Voice, Voice Design, Voice Clone) deferred until pain is confirmed there.

## Problem

The `sm_table` (Markdown results table) in Script Mode has two UX issues:

1. **Redundant content.** The table re-displays each line's text, which is already visible in the input script box above it. Most users won't reference the table at all.
2. **Stale overlay on re-generate.** When the user clicks Generate a second time, the `show_progress="full"` progress overlay paints on top of the prior table, leaving the old results visible behind the spinner (screenshotted in user testing).

## Solution

Wrap the existing `sm_table` (`gr.Markdown`) inside a collapsed-by-default `gr.Accordion`. The audio output and status textbox stay where they are.

- Default state: collapsed (`open=False`)
- Label: `"Per-line breakdown"`
- Position: right column of Script Mode, between `sm_audio` and `sm_status` (unchanged location)
- Behavior on re-generate: accordion stays in whatever state the user left it. No auto-collapse, no auto-expand.

## Why this fixes both problems

- The duplicate text is no longer visible by default — users who don't need the breakdown never see it.
- When Generate fires, the progress overlay paints over an empty region (the collapsed accordion header). No stale table behind it.

## Considered alternative

**Clear-on-generate** — reset `sm_table` to an empty string before each generation runs. Solves problem #2 but not #1. Dropped.

## Implementation outline

Single change in `app.py` around the existing `sm_table` definition (currently at line 1813):

```python
sm_audio = gr.Audio(...)                                     # unchanged
with gr.Accordion("Per-line breakdown", open=False):
    sm_table = gr.Markdown(value="*Results will appear after generation.*")
sm_status = gr.Textbox(...)                                  # unchanged
```

- No handler changes — `sm_table` keeps its name and component identity, so the existing return paths in `generate_script_handler` and `_parse_and_update_slots` still target it correctly.
- No CSS/theme changes.

## Out of scope

- Batch result tables (`cv_batch_table`, `vd_batch_table`, `vc_batch_table`). Same pattern may apply but will be evaluated separately after user gauges pain on those tabs.
- Auto-expand-on-completion or remember-state-across-sessions. Not requested.

## Testing

After change:
1. Restart app, open Script Mode in Safari (the more sensitive browser).
2. Verify accordion is collapsed on tab load; clicking it expands and shows the placeholder text.
3. Parse a script, generate audio, confirm audio plays.
4. Expand the accordion → verify the Markdown results table renders correctly.
5. Click Generate again → verify the progress overlay no longer covers stale data, and accordion preserves its open/closed state.
