import threading
from dataclasses import dataclass, field
from typing import Any, List

from config import (
    DEFAULT_MODEL_SIZE, DEFAULT_QUANTIZATION,
    DEFAULT_AUTOSAVE, DEFAULT_BATCH_SIZE, DEFAULT_DENOISE_REF,
    DEFAULT_EXPORT_FORMAT, DEFAULT_LOUDNORM, DEFAULT_MAX_TOKENS,
    DEFAULT_MP3_BITRATE, DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_TRIM_SILENCE,
    ENABLE_JIT_COMPILE,
    LANGUAGE_AUTO,
    OUTPUT_DIR,
)


@dataclass
class AppSettings:
    """Runtime settings, mutated by the Settings tab."""
    model_size: str = DEFAULT_MODEL_SIZE
    quantization: str = DEFAULT_QUANTIZATION
    jit_compile: bool = ENABLE_JIT_COMPILE
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = DEFAULT_TIMEOUT
    output_dir: str = OUTPUT_DIR
    autosave: bool = DEFAULT_AUTOSAVE
    export_format: str = DEFAULT_EXPORT_FORMAT
    mp3_bitrate: int = DEFAULT_MP3_BITRATE
    loudnorm: bool = DEFAULT_LOUDNORM
    trim_silence: bool = DEFAULT_TRIM_SILENCE
    denoise_ref: bool = DEFAULT_DENOISE_REF
    batch_size: int = DEFAULT_BATCH_SIZE
    default_language: str = LANGUAGE_AUTO

    def gen_kwargs(self) -> dict:
        """Sampler kwargs passed to every engine generate call."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
        }


@dataclass
class AppContext:
    """Everything handlers need, passed explicitly instead of module globals."""
    engine: Any
    library: Any
    history: Any
    yt: Any
    settings: AppSettings
    startup_warnings: List[str] = field(default_factory=list)
    # Successful Voice Design provenance used by the optional Voice Library
    # save action and the "use last seed" UI control.
    last_voice_design_seed: int | None = None
    last_voice_design_style_instruction: str = ""
    # One shared cooperative-cancel flag: the engine lock serializes runs, so a
    # single event is enough. Runners clear it at start; Stop buttons set it.
    cancel_event: threading.Event = field(default_factory=threading.Event)
