"""All user-facing text, one string per concept.

Stage 4 canonical terms (change wording HERE, not in tabs):
"Text to speak" · "Voice" (preset speaker) · "Reference audio" /
"Reference transcript" · "Style instructions (optional)" ·
"Voice description" · "Generated audio" · "Voice Clone" (mode name).
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
TEXT_TO_SPEAK = "Text to speak"
TEXT_PLACEHOLDER = "Enter the text you want spoken…"
TIP_TEXT_LENGTH = (
    "_Tip: 1–4 sentences work best. Long or number-heavy text can run away — "
    "the Stop button or the auto-stop timeout will end it, keeping partial audio._"
)
LANGUAGE = "Language"
LANGUAGE_AUTO = CONFIG_LANGUAGE_AUTO  # UI label for automatic detection
GENERATE = "Generate"

# --- Voice cloning extras (Stage 2) ---
TRIM_REF_LABEL = "Trim silence from reference (recommended)"
REP_PENALTY_CLONE_INFO = "Voice cloning always uses at least 1.5"

# --- Shared output column ---
OUTPUT = "Generated audio"
SAVE_AUDIO = "Save Audio"
SAVE_PATH_PLACEHOLDER = "Save path appears here…"

# --- Batch accordion ---
BATCH_ACCORDION = "Batch Mode"
SPLIT_MODE = "Split text by"
SILENCE_GAP = "Silence between segments (ms)"
GENERATE_BATCH = "Generate Batch"
BATCH_RESULTS = "Segments"
COMBINED_OUTPUT = "Generated audio (combined)"
SAVE_COMBINED = "Save Combined Audio"
BATCH_STATUS = "Batch Status"

# --- Save-to-library accordion ---
LIB_SAVE_ACCORDION = "Save to Voice Library"
VOICE_NAME = "Voice Name"
SAVE_VOICE_TO_LIBRARY = "Save Voice to Library"
LIB_STATUS_PLACEHOLDER = "Library status…"

# --- Custom Voice tab ---
TAB_CUSTOM_VOICE = "Custom Voice"
CV_SPEAKER = "Voice"
CV_INSTRUCT = "Style instructions (optional)"
CV_INSTRUCT_PLACEHOLDER = "e.g. Speak warmly, Sound excited..."

# --- Voice Design tab ---
TAB_VOICE_DESIGN = "Voice Design"
VD_INSTRUCT = "Voice description"
VD_INSTRUCT_PLACEHOLDER = "e.g. A deep, calm male narrator with a British accent"
VD_LIB_NAME_PLACEHOLDER = "my_narrator"

# --- Voice Clone tab ---
TAB_VOICE_CLONE = "Voice Clone"
VC_NOTICE_HTML = (
    "<div class='info-notice'>"
    "<strong>The reference transcript must exactly match what is spoken in the audio.</strong> "
    "Use a clean 3–30 second clip for best results."
    "</div>"
)
VC_LIBRARY_VOICE = "Load from Library"
VC_REF_AUDIO = "Reference audio"
VC_TRANSCRIBE = "Transcribe Reference"
VC_TRANSCRIBE_HINT_HTML = "<div class='text-hint'>Auto-fills the transcript using on-device speech recognition</div>"
VC_REF_TEXT = "Reference transcript (required)"
VC_REF_TEXT_PLACEHOLDER = "The exact words spoken in the reference audio"
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
TIMEOUT_SLIDER_INFO = "Auto-stops a run after this many seconds, keeping partial audio"

# --- Settings tab ---
TAB_SETTINGS = "Settings"
SET_MODEL_HEADER = "### Model"
SET_MODEL_SIZE = "Model size"
SET_QUANT = "Quantization"
SET_QUANT_INFO = "Smaller = less memory, slightly lower quality"
SET_LOADED_MODEL = "Loaded model"
SET_UNLOAD = "Unload Model / Free RAM"
SET_REF_HEADER = "### Reference Audio"
SET_DENOISE = "Reduce reference background noise"
SET_DENOISE_INFO = "Uses the DeepFilterNet model (8 MB, downloads on first use) on Voice Clone references"
SET_LANGUAGE_HEADER = "### Language"
SET_DEFAULT_LANGUAGE = "Default language"
SET_JIT = "Speed up repeat runs"
SET_JIT_INFO = "Compiles the model after the first generation; changing this reloads the model"
SET_CACHE_ACCORDION = "Model Cache & Speech Recognition"
SET_CACHE_HEADER = "### Model Cache"
SET_CACHE_DIR = "Model download folder"
SET_DELETE_MODELS = "Delete Downloaded Models"
SET_DELETE_PLACEHOLDER = "Models will be re-downloaded on next use."
SET_ASR_HEADER = "### Speech Recognition"
SET_ASR_STATUS = "Speech recognition model"
SET_ASR_NOT_LOADED = "Not loaded (loads on demand)"
SET_ASR_UNLOAD = "Unload Speech Recognition Model"
SET_GENERATION_HEADER = "### Generation"
SET_PRESET = "Generation presets"
SET_PRESET_INFO = "Presets fill the sliders below. Adjust freely afterward."
SET_TEMP = "Temperature"
SET_TEMP_INFO = "Higher = more varied delivery, lower = more consistent"
SET_TOP_K = "Top-K"
SET_TOP_K_INFO = "How many candidate sounds are considered each step — lower = safer"
SET_TOP_P = "Top-P"
SET_TOP_P_INFO = "Keeps only the most likely sounds — lower = more predictable"
SET_REP_PENALTY = "Repetition penalty"
SET_REP_PENALTY_INFO = "Discourages repeating sounds. Voice Clone always uses at least 1.5"
SET_MAX_TOKENS = "Max length (tokens)"
SET_MAX_TOKENS_INFO = "Upper limit on how long one generation may run — lower it if runs go on too long"
SET_TIMEOUT = "Auto-stop after (seconds)"
SET_BATCH_SIZE = "Batch size"
SET_BATCH_SIZE_INFO = "Segments generated in parallel (batch and Script Mode)"
SET_RESET = "Reset to Defaults"
SET_OUTPUT_HEADER = "### Output"
SET_OUTPUT_DIR = "Output folder"
SET_AUTOSAVE = "Auto-save generated audio"
SET_EXPORT_HEADER = "### Export Format"
SET_EXPORT_FORMAT = "Audio format"
SET_MP3_BITRATE = "MP3 bitrate (kbps)"
SET_POST_HEADER = "### Post-Processing"
SET_LOUDNORM = "Normalize loudness"
SET_LOUDNORM_INFO = "EBU R128 broadcast standard"
SET_TRIM_SILENCE = "Trim leading/trailing silence"
SET_STORAGE_ACCORDION = "Storage & Cache"
SET_YT_CACHE_HEADER = "### YT Cache"
SET_YT_CACHE_CLEAR = "Clear YT Cache"
SET_YT_CACHE_PLACEHOLDER = "Cache: {cache_dir}/"
SET_STORAGE_HEADER = "### Storage Paths"
SET_STORAGE_LIBRARY = "Voice Library"
SET_STORAGE_HISTORY = "History"
SET_APPLY = "Apply Settings"
SET_NO_MODEL = "No model loaded"
SET_MODEL_LOADED = "Loaded: {repo}"
SET_APPLIED = "Settings applied — {details}."
SET_APPLIED_SIZE_QUANT = "size: {size}, quant: {quant}"
SET_APPLIED_UNLOADED = "model unloaded"
SET_UNLOADED_MSG = "Model unloaded. RAM freed."
SET_ASR_UNLOADED_MSG = "Speech recognition model unloaded. RAM freed."
SET_YT_CACHE_CLEARED = "YT cache cleared — {n} entr{plural} removed"
SET_CACHE_DIR_MISSING = "Model download folder not found"
SET_DELETED_MODELS = "Deleted {n} model(s): {names}"
SET_DELETE_FAILED = "Failed to delete {n}: {details}"
SET_NO_MODELS_FOUND = "No Qwen3-TTS models found in cache"

# --- Table empty states ---
NO_VOICES = "*No voices saved.*"
NO_HISTORY = "*No history entries.*"
NO_ENTRIES = "*No entries.*"
