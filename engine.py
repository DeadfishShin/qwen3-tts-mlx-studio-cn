import gc
import os
import threading

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_audio.stt.utils import load_model as load_stt_model

from config import REPO_TEMPLATE, MODEL_VARIANTS, DEFAULT_MODEL_SIZE, DEFAULT_QUANTIZATION, ENABLE_JIT_COMPILE, ASR_REPO_ID

LOCK_TIMEOUT = 10  # seconds to wait for lock before raising


class TTSEngine:
    """Manages model loading, unloading, and inference."""

    def __init__(self):
        self.current_model = None
        self.current_model_type = None  # "custom_voice" | "voice_design" | "base"
        self.model_size = DEFAULT_MODEL_SIZE
        self.quantization = DEFAULT_QUANTIZATION
        self.jit_compile = ENABLE_JIT_COMPILE
        self.asr_model = None
        self._lock = threading.Lock()

    def _acquire_lock(self):
        """Acquire lock with timeout so callers don't hang forever."""
        if not self._lock.acquire(timeout=LOCK_TIMEOUT):
            raise RuntimeError(
                "Engine is busy with a previous generation. Please wait and try again."
            )

    def _release_lock(self):
        """Return cached GPU buffers to the OS, then release the lock.

        MLX's buffer cache can grow to several GB after a long generation and
        is never returned while idle — enough to push a 16 GB machine into
        swap. Clearing per-generation trades a little re-allocation time for
        a low idle footprint.
        """
        mx.clear_cache()
        self._lock.release()

    def get_repo_id(self, model_type: str) -> str:
        """Build HuggingFace repo ID from model_type + size + quant."""
        variant = MODEL_VARIANTS[model_type]
        return REPO_TEMPLATE.format(
            size=self.model_size, variant=variant, quant=self.quantization
        )

    def _load_model(self, model_type: str):
        """Load model, unloading previous if different. Caller must hold lock."""
        self._unload_asr_unlocked()
        if self.current_model_type == model_type:
            return
        self._unload_model_unlocked()
        repo_id = self.get_repo_id(model_type)
        self.current_model = load_model(repo_id)
        self.current_model_type = model_type
        if self.jit_compile:
            self._apply_compile()
        mx.eval(self.current_model.parameters())

    def _apply_compile(self):
        """Wrap talker forward passes with mx.compile for faster inference."""
        talker = getattr(self.current_model, "talker", None)
        if talker is None:
            return
        talker.__call__ = mx.compile(talker.__call__, shapeless=True)
        code_pred = getattr(talker, "code_predictor", None)
        if code_pred is not None:
            code_pred.__call__ = mx.compile(code_pred.__call__, shapeless=True)

    def is_model_loaded(self, model_type: str) -> bool:
        """Return True if this model type is already in memory (no swap needed)."""
        return self.current_model_type == model_type

    def unload_model(self):
        """Free memory from current model (thread-safe)."""
        self._acquire_lock()
        try:
            self._unload_model_unlocked()
        finally:
            self._release_lock()

    def _unload_model_unlocked(self):
        """Free memory from current model (and ASR if loaded). Caller must hold lock."""
        self._unload_asr_unlocked()
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            gc.collect()
            mx.clear_cache()

    def generate_custom_voice(self, text, speaker, language, instruct="", **kwargs) -> tuple:
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        try:
            self._load_model("custom_voice")
            results = list(
                self.current_model.generate_custom_voice(
                    text=text, speaker=speaker, language=language, instruct=instruct,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._release_lock()

    def generate_voice_design(self, text, language, instruct, **kwargs) -> tuple:
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        try:
            self._load_model("voice_design")
            results = list(
                self.current_model.generate_voice_design(
                    text=text, language=language, instruct=instruct,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._release_lock()

    def _prepare_ref(self, ref_audio_path, trim_ref, denoise_ref):
        """Trim then denoise a reference clip. Returns (path, temp files to delete)."""
        from audio_utils import denoise_ref_audio, trim_ref_silence
        temps = []
        if trim_ref:
            trimmed = trim_ref_silence(ref_audio_path)
            if trimmed != ref_audio_path:
                temps.append(trimmed)
                ref_audio_path = trimmed
        if denoise_ref:
            denoised = denoise_ref_audio(ref_audio_path)
            temps.append(denoised)
            ref_audio_path = denoised
        return ref_audio_path, temps

    @staticmethod
    def _cleanup_temps(temps):
        for path in temps:
            if os.path.isfile(path):
                os.remove(path)

    def generate_voice_clone(self, text, ref_audio_path, ref_text, language="English",
                             denoise_ref=False, trim_ref=False, **kwargs) -> tuple:
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        temps = []
        try:
            ref_audio_path, temps = self._prepare_ref(ref_audio_path, trim_ref, denoise_ref)
            self._load_model("base")
            results = list(
                self.current_model.generate(
                    text=text, ref_audio=ref_audio_path, ref_text=ref_text,
                    language=language,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._cleanup_temps(temps)
            self._release_lock()

    def stream_generate_custom_voice(self, text, speaker, language, instruct="",
                                     streaming_interval=2.0, **kwargs):
        """Yield (sample_rate, numpy_chunk) as audio is decoded.

        Holds the engine lock until the generator is exhausted or closed —
        consumers must close() it promptly when abandoning mid-stream.
        """
        self._acquire_lock()
        try:
            self._load_model("custom_voice")
            inner = self.current_model.generate_custom_voice(
                text=text, speaker=speaker, language=language, instruct=instruct,
                stream=True, streaming_interval=streaming_interval, **kwargs,
            )
            try:
                for result in inner:
                    yield self._to_numpy(result)
            finally:
                inner.close()
                gc.collect()
        finally:
            self._release_lock()

    def stream_generate_voice_design(self, text, language, instruct,
                                     streaming_interval=2.0, **kwargs):
        """Yield (sample_rate, numpy_chunk) as audio is decoded (see stream_generate_custom_voice)."""
        self._acquire_lock()
        try:
            self._load_model("voice_design")
            inner = self.current_model.generate_voice_design(
                text=text, language=language, instruct=instruct,
                stream=True, streaming_interval=streaming_interval, **kwargs,
            )
            try:
                for result in inner:
                    yield self._to_numpy(result)
            finally:
                inner.close()
                gc.collect()
        finally:
            self._release_lock()

    def stream_generate_voice_clone(self, text, ref_audio_path, ref_text,
                                    language="English", denoise_ref=False,
                                    trim_ref=False, streaming_interval=2.0, **kwargs):
        """Yield (sample_rate, numpy_chunk) as audio is decoded (see stream_generate_custom_voice)."""
        self._acquire_lock()
        temps = []
        try:
            ref_audio_path, temps = self._prepare_ref(ref_audio_path, trim_ref, denoise_ref)
            self._load_model("base")
            inner = self.current_model.generate(
                text=text, ref_audio=ref_audio_path, ref_text=ref_text,
                language=language,
                stream=True, streaming_interval=streaming_interval, **kwargs,
            )
            try:
                for result in inner:
                    yield self._to_numpy(result)
            finally:
                inner.close()
                gc.collect()
        finally:
            self._cleanup_temps(temps)
            self._release_lock()

    def batch_generate_voice_clone(self, texts, ref_audio_path, ref_text,
                                   language="English", denoise_ref=False,
                                   trim_ref=False, **kwargs):
        """Clone-generate multiple texts sharing ONE reference in a batched pass.

        Upstream batch_generate supports a single shared reference per call
        (and silently enforces repetition_penalty >= 1.5 on the ICL path).
        Returns list of (sample_rate, numpy_audio_array) in input order.
        """
        self._acquire_lock()
        temps = []
        try:
            ref_audio_path, temps = self._prepare_ref(ref_audio_path, trim_ref, denoise_ref)
            self._load_model("base")
            results = list(
                self.current_model.batch_generate(
                    texts=texts,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    lang_code=language,
                    **kwargs,
                )
            )
            results.sort(key=lambda r: r.sequence_idx)
            return [self._to_numpy(r) for r in results]
        finally:
            self._cleanup_temps(temps)
            self._release_lock()

    def batch_generate_custom_voice(self, texts, speaker, language, instruct="", **kwargs):
        """Generate audio for multiple texts in one batched forward pass.

        speaker and instruct may each be a single value (applied to every text)
        or a per-text list. Returns list of (sample_rate, numpy_audio_array)
        in input order.
        """
        self._acquire_lock()
        try:
            self._load_model("custom_voice")
            batch_size = len(texts)
            speakers = (list(speaker) if isinstance(speaker, (list, tuple))
                        else [speaker] * batch_size)
            instructs = (list(instruct) if isinstance(instruct, (list, tuple))
                         else [instruct] * batch_size)
            results = list(
                self.current_model.batch_generate(
                    texts=texts,
                    voices=speakers,
                    instructs=instructs,
                    lang_code=language,
                    **kwargs,
                )
            )
            # Sort by sequence_idx to guarantee input order
            results.sort(key=lambda r: r.sequence_idx)
            return [self._to_numpy(r) for r in results]
        finally:
            self._release_lock()

    def batch_generate_voice_design(self, texts, language, instruct, **kwargs):
        """Generate audio for multiple texts in one batched forward pass.

        instruct may be a single value (applied to every text) or a per-text
        list. Returns list of (sample_rate, numpy_audio_array) in input order.
        """
        self._acquire_lock()
        try:
            self._load_model("voice_design")
            batch_size = len(texts)
            instructs = (list(instruct) if isinstance(instruct, (list, tuple))
                         else [instruct] * batch_size)
            results = list(
                self.current_model.batch_generate(
                    texts=texts,
                    instructs=instructs,
                    lang_code=language,
                    **kwargs,
                )
            )
            results.sort(key=lambda r: r.sequence_idx)
            return [self._to_numpy(r) for r in results]
        finally:
            self._release_lock()

    # ----- ASR -----

    def _load_asr(self):
        """Load ASR model, unloading TTS first. Caller must hold lock."""
        if self.asr_model is not None:
            return  # already loaded
        self._unload_model_unlocked()  # free TTS
        self.asr_model = load_stt_model(ASR_REPO_ID)
        mx.eval(self.asr_model.parameters())

    def _unload_asr_unlocked(self):
        """Free ASR model. Caller must hold lock."""
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None
            gc.collect()

    def unload_asr(self):
        """Free ASR model (thread-safe)."""
        self._acquire_lock()
        try:
            self._unload_asr_unlocked()
        finally:
            self._release_lock()

    def transcribe(self, audio_path, language="auto") -> str:
        """Transcribe audio file, returns text. Loads/unloads ASR automatically."""
        self._acquire_lock()
        try:
            self._load_asr()
            result = self.asr_model.generate(audio_path, language=language)
            return result.text
        finally:
            self._unload_asr_unlocked()
            self._release_lock()

    def stream_transcribe(self, audio_path, language="auto"):
        """Yield transcript text deltas. Loads/unloads ASR automatically.

        Holds the engine lock until the generator is exhausted or closed.
        Upstream auto-detects only when language is None ("auto" would be
        injected into the prompt as a literal language name).
        """
        self._acquire_lock()
        try:
            self._load_asr()
            lang = None if not language or language.lower() == "auto" else language
            inner = self.asr_model.generate(audio_path, language=lang, stream=True)
            try:
                for piece in inner:
                    text = getattr(piece, "text", None)
                    if text is None:
                        text = str(piece)
                    if text:
                        yield text
            finally:
                if hasattr(inner, "close"):
                    inner.close()
        finally:
            self._unload_asr_unlocked()
            self._release_lock()

    def _to_numpy(self, result) -> tuple:
        """Convert mlx array result to numpy for Gradio Audio component."""
        audio = np.array(result.audio, dtype=np.float32)
        sr = getattr(result, "sample_rate", 24000)
        return (sr, audio)
