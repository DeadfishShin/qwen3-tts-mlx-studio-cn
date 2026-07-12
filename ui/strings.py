"""All user-facing text, one string per concept.

Stage 1 rule: every value is copied VERBATIM from the original app.py.
Rewording happens in Stage 4 (clarity pass) — in this file only.
"""
from config import LANGUAGE_AUTO as CONFIG_LANGUAGE_AUTO

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
LANGUAGE_AUTO = CONFIG_LANGUAGE_AUTO  # UI label for automatic detection
GENERATE = "Generate"

# --- Voice cloning extras (Stage 2) ---
TRIM_REF_LABEL = "Trim silence from reference (recommended)"
REP_PENALTY_CLONE_INFO = "Voice cloning always uses at least 1.5"

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

# --- Voice Cloning tab ---
TAB_VOICE_CLONE = "Voice Cloning"
VC_NOTICE_HTML = (
    "<div class='info-notice'>"
    "<strong>Reference transcript must exactly match what is spoken in the audio.</strong> "
    "Use a clean 3–30 second clip for best results."
    "</div>"
)
VC_LIBRARY_VOICE = "Load from Library"
VC_REF_AUDIO = "Reference Audio"
VC_TRANSCRIBE = "Transcribe Reference"
VC_TRANSCRIBE_HINT_HTML = "<div class='text-hint'>Auto-fills transcript using Qwen3-ASR-1.7B-8bit</div>"
VC_REF_TEXT = "Reference Transcript (required)"
VC_REF_TEXT_PLACEHOLDER = "Exact text spoken in reference audio"
VC_TEXT_LABEL = "Text to Speak"
VC_TEXT_PLACEHOLDER = "Enter text to speak in the cloned voice..."
VC_LIB_NAME_PLACEHOLDER = "my_clone"

# --- Transcription tab ---
TAB_TRANSCRIPTION = "Transcription"

# --- Streaming / cancel (Stage 3) ---
STOP = "Stop"
STOPPING = "Stopping…"
GENERATING_STATUS = "Generating… {secs:.1f}s"
STOPPED_KEPT = "Stopped — kept {secs:.1f}s of partial audio"
TIMED_OUT_KEPT = "Timed out after {timeout}s — kept {secs:.1f}s of partial audio"
TIMEOUT_MSG = "Generation timed out — lower Max Tokens or shorten the text"
BATCH_SEGMENT_PROGRESS = "Segment {done}/{total} · {secs:.1f}s audio"
BATCH_STOPPED = "Stopped — completed {done}/{total} segments"
SCRIPT_LINE_PROGRESS = "Line {done}/{total} · {secs:.1f}s audio"
SCRIPT_STOPPED = "Stopped — completed {done}/{total} lines"
ASR_LOADING = "Loading ASR model..."
TRANSCRIBING = "Transcribing… {words} words"
TRANSCRIBE_STOPPED = "Stopped — partial transcript kept"
STREAM_PLAYBACK_LABEL = "Play audio while it generates"
STREAM_PLAYBACK_INFO = "Streams sound to the player as it is generated (single generations)"
TIMEOUT_SLIDER_INFO = "Auto-stops a run after this many seconds, keeping partial audio"

# --- Table empty states ---
NO_VOICES = "*No voices saved.*"
NO_HISTORY = "*No history entries.*"
NO_ENTRIES = "*No entries.*"
