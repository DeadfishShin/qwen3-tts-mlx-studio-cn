"""All user-facing text, one string per concept.

Stage 1 rule: every value is copied VERBATIM from the original app.py.
Rewording happens in Stage 4 (clarity pass) — in this file only.
"""

# --- App shell ---
APP_TITLE = "Qwen3-TTS MLX Studio"
APP_HEADER_HTML = (
    "<div class='app-header'>"
    "<h1>Qwen3-TTS MLX Studio</h1>"
    "<p class='subtitle'>Local AI Text-to-Speech &middot; MLX &middot; Apple Silicon</p>"
    "</div>"
)
STATUS_READY = "Ready"

# --- Shared generation inputs ---
TEXT_PLACEHOLDER = "Enter text to speak..."
TIP_TEXT_LENGTH = "_Tip: 1–4 sentences work best. Very long text may hit the 120 s timeout._"
LANGUAGE = "Language"
GENERATE = "Generate"

# --- Shared output column ---
OUTPUT = "Output"
SAVE_AUDIO = "Save Audio"
SAVE_PATH_PLACEHOLDER = "Save path appears here…"

# --- Batch accordion ---
BATCH_ACCORDION = "Batch Mode"
SPLIT_MODE = "Split Mode"
SILENCE_GAP = "Silence Gap (ms)"
GENERATE_BATCH = "Generate Batch"
BATCH_RESULTS = "Batch Results"
COMBINED_OUTPUT = "Combined Output"
SAVE_COMBINED = "Save Combined Audio"
BATCH_STATUS = "Batch Status"

# --- Save-to-library accordion ---
LIB_SAVE_ACCORDION = "Save to Voice Library"
VOICE_NAME = "Voice Name"
SAVE_VOICE_TO_LIBRARY = "Save Voice to Library"
LIB_STATUS_PLACEHOLDER = "Library status…"

# --- Custom Voice tab ---
TAB_CUSTOM_VOICE = "Custom Voice"
CV_TEXT_LABEL = "Text"
CV_SPEAKER = "Speaker"
CV_INSTRUCT = "Style Instruction (optional)"
CV_INSTRUCT_PLACEHOLDER = "e.g. Speak warmly, Sound excited..."

# --- Voice Design tab ---
TAB_VOICE_DESIGN = "Voice Design"
VD_TEXT_LABEL = "Text"
VD_INSTRUCT = "Voice Description"
VD_INSTRUCT_PLACEHOLDER = "e.g. A deep, calm male narrator with a British accent"
VD_LIB_NAME_PLACEHOLDER = "my_narrator"

# --- Table empty states ---
NO_VOICES = "*No voices saved.*"
NO_HISTORY = "*No history entries.*"
NO_ENTRIES = "*No entries.*"
