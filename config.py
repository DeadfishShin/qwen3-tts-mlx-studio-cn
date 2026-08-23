# Model defaults
DEFAULT_MODEL_SIZE = "1.7B"
DEFAULT_QUANTIZATION = "bf16"

# HuggingFace repo template
REPO_TEMPLATE = "mlx-community/Qwen3-TTS-12Hz-{size}-{variant}-{quant}"

# Model variant mapping
MODEL_VARIANTS = {
    "custom_voice": "CustomVoice",
    "voice_design": "VoiceDesign",
    "base": "Base",
}

# ASR model (fixed — do not change size/quant)
ASR_REPO_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"

# UI value for automatic language detection (engine receives "auto")
LANGUAGE_AUTO = "Auto-detect"

# Supported languages
LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
]

# Default speakers — must match model's actual speaker IDs exactly (case-sensitive)
DEFAULT_SPEAKERS = [
    "serena", "vivian", "uncle_fu", "ryan", "aiden",
    "ono_anna", "sohee", "eric", "dylan",
]

# Audio settings
DEFAULT_SAMPLE_RATE = 24000
OUTPUT_DIR = "./outputs"
VOICE_LIBRARY_DIR = "./voices"
HISTORY_DIR = "./outputs/history"
YT_CACHE_DIR = ".yt_cache"

# History settings
MAX_HISTORY_ENTRIES = 50
MAX_HISTORY_AUDIO_CACHE = 10

# Batch settings
DEFAULT_BATCH_SPLIT_MODE = "paragraph"
DEFAULT_SILENCE_GAP_MS = 300
MAX_BATCH_SEGMENTS = 50
DEFAULT_BATCH_SIZE = 4
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8

# Script settings
MAX_SCRIPT_SPEAKERS = 8
DEFAULT_SCRIPT_SILENCE_MS = 500

# Generation defaults
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120

# Voice Design reproducibility.  MLX accepts a Python integer seed; this
# conservative uint32-sized range is also exactly representable by the UI's
# JavaScript number transport.  The seed is applied only on TTSEngine's MLX
# owner thread, never from a Gradio/AnyIO caller.
VOICE_DESIGN_SEED_MIN = 0
VOICE_DESIGN_SEED_MAX = 2**32 - 1
DEFAULT_VOICE_DESIGN_SEED = 123456

# Streaming decode (Stage 3) — generation is consumed in chunks internally so
# Stop/timeout can land between chunks; this sets the cancel latency (seconds
# of audio per chunk). Live playback of these chunks was removed: it needs
# faster-than-real-time generation, which this app's low-end-hardware target
# doesn't have.
STREAMING_INTERVAL_S = 1.0

# Owner-controlled Apple Silicon testing found a severe startup timbre
# transient in the first ~0.5s of the Qwen3-TTS Base Voice Clone stream at
# 1.0s.  A Clone-only 2.0s interval removed that artifact in the same test;
# the longer chunk slightly reduces Stop/timeout granularity.  Keep this
# scoped to Clone while future low-latency work investigates decoder
# first-chunk context instead of treating 2.0s as universally optimal.
VOICE_CLONE_STREAMING_INTERVAL_S = 2.0

# Output defaults
DEFAULT_AUTOSAVE = False
DEFAULT_EXPORT_FORMAT = "wav"    # "wav", "mp3", "ogg"
DEFAULT_MP3_BITRATE = 192       # kbps
DEFAULT_LOUDNORM = False        # EBU R128 loudness normalization
DEFAULT_TRIM_SILENCE = False    # Strip leading/trailing silence
DEFAULT_DENOISE_REF = False     # DeepFilterNet ref audio denoising

# DeepFilterNet model (8MB, cached after first download)
DEEPFILTER_REPO = "mlx-community/DeepFilterNet-mlx"

# Silero VAD model (~2MB) for trimming reference-clip silence
VAD_REPO = "mlx-community/silero-vad"

# JIT compilation — opt in only after the worker-thread path is proven stable.
# MLX compile-cache cleanup has had interpreter-shutdown regressions; the
# user-facing Settings toggle remains available for an explicit opt-in.
ENABLE_JIT_COMPILE = False

# Gradio settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
