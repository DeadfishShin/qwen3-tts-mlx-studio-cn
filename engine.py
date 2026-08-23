"""Thread-affine MLX inference engine for Qwen3-TTS Studio.

Gradio executes synchronous handlers on reusable AnyIO worker threads. MLX
streams and resident model state are instead owned by one dedicated,
long-lived thread. Public methods below are thread-safe request proxies; all
methods ending in ``_unlocked`` or ``_impl`` run only on the owner thread.
"""

import atexit
import gc
import logging
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_audio.stt.utils import load_model as load_stt_model

from config import (
    ASR_REPO_ID,
    DEFAULT_MODEL_SIZE,
    DEFAULT_QUANTIZATION,
    ENABLE_JIT_COMPILE,
    MODEL_VARIANTS,
    REPO_TEMPLATE,
)


OWNER_THREAD_NAME = "qwen3-tts-mlx-owner"
REQUEST_QUEUE_SIZE = 32
REQUEST_WAIT_S = 0.1
RUNTIME_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "runtime.log")

_runtime_logger = logging.getLogger("qwen3_tts.runtime")
_runtime_logger.setLevel(logging.INFO)
_runtime_logger.propagate = False
_runtime_logger_lock = threading.Lock()


def _log_runtime_exception(
    *,
    operation: str,
    owner_thread_id: int | None,
    caller_thread_id: int | None,
    exc: BaseException,
    traceback_text: str | None = None,
):
    """Persist a concise, non-content-bearing MLX failure diagnostic."""
    os.makedirs(os.path.dirname(RUNTIME_LOG_PATH), exist_ok=True)
    with _runtime_logger_lock:
        has_runtime_file = any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(handler.baseFilename) == os.path.abspath(RUNTIME_LOG_PATH)
            for handler in _runtime_logger.handlers
        )
        if not has_runtime_file:
            handler = logging.FileHandler(
                RUNTIME_LOG_PATH, encoding="utf-8", delay=True
            )
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s"
            ))
            _runtime_logger.addHandler(handler)
    _runtime_logger.error(
        "operation=%s owner_thread_id=%s caller_thread_id=%s "
        "exception=%s: %s\n%s",
        operation,
        owner_thread_id,
        caller_thread_id,
        type(exc).__name__,
        str(exc),
        traceback_text or "",
    )


@dataclass
class _InferenceRequest:
    """One synchronous or streaming item submitted to the owner thread."""

    operation: str
    function: Callable[[], Any]
    transform: Callable[[Any], Any] = lambda value: value
    stream: bool = False
    cleanup: bool = True
    caller_thread_id: int = field(default_factory=threading.get_ident)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))
    result: Any = None
    error: BaseException | None = None
    owner_thread_id: int | None = None


class _StopOwner:
    pass


STOP_OWNER = _StopOwner()


class TTSEngine:
    """Proxy requests to one persistent, thread-affine MLX owner."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        quantization: str = DEFAULT_QUANTIZATION,
        jit_compile: bool | None = None,
    ):
        # These configuration values are ordinary Python state. The model and
        # every MLX object below are created/read only by _owner_loop().
        self.model_size = model_size
        self.quantization = quantization
        self.jit_compile = ENABLE_JIT_COMPILE if jit_compile is None else bool(jit_compile)

        self.current_model = None
        self.current_model_type = None  # owner-thread state
        self.asr_model = None            # owner-thread state

        self._requests = queue.Queue(maxsize=REQUEST_QUEUE_SIZE)
        self._owner_ready = threading.Event()
        self._owner_stopped = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._closing = False
        self._shutdown_error: BaseException | None = None
        self._active_request: _InferenceRequest | None = None
        self._owner_thread_id: int | None = None
        self._owner_thread = threading.Thread(
            target=self._owner_loop,
            name=OWNER_THREAD_NAME,
            daemon=False,
        )
        self._owner_thread.start()
        if not self._owner_ready.wait(timeout=10):
            raise RuntimeError("MLX owner thread did not start")

        # Fallback for abnormal exits. app.py also calls shutdown() in a
        # launch finally block for normal exits.
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Owner-thread request plumbing
    # ------------------------------------------------------------------

    @property
    def owner_thread_id(self) -> int | None:
        """Thread identifier used for diagnostics and deterministic tests."""
        return self._owner_thread_id

    @property
    def owner_thread_alive(self) -> bool:
        return self._owner_thread.is_alive()

    def _on_owner_thread(self) -> bool:
        return threading.get_ident() == self._owner_thread_id

    def _ensure_open(self):
        if self._closing or self._owner_stopped.is_set():
            raise RuntimeError("MLX owner thread is shutting down")

    def _wait_for_request(self, request: _InferenceRequest):
        while not request.done.wait(REQUEST_WAIT_S):
            if not self._owner_thread.is_alive():
                raise RuntimeError("MLX owner thread stopped unexpectedly")

    def _raise_request_error(self, request: _InferenceRequest):
        if request.error is not None:
            raise request.error

    def _submit_sync(
        self,
        operation: str,
        function: Callable[[], Any],
        *,
        cleanup: bool = True,
    ):
        if self._on_owner_thread():
            return function()
        self._ensure_open()
        request = _InferenceRequest(
            operation=operation,
            function=function,
            cleanup=cleanup,
            stream=False,
            events=queue.Queue(maxsize=1),
        )
        self._requests.put(request)
        self._wait_for_request(request)
        self._raise_request_error(request)
        return request.result

    def _put_stream_event(self, request: _InferenceRequest, event) -> bool:
        while not request.cancel_event.is_set():
            try:
                request.events.put(event, timeout=REQUEST_WAIT_S)
                return True
            except queue.Full:
                continue
        return False

    def _submit_stream(
        self,
        operation: str,
        function: Callable[[], Any],
        transform: Callable[[Any], Any],
    ):
        """Yield transformed stream items while the owner consumes MLX."""
        if self._on_owner_thread():
            inner = function()
            try:
                for item in inner:
                    yield transform(item)
            finally:
                if hasattr(inner, "close"):
                    inner.close()
                gc.collect()
            return

        self._ensure_open()
        request = _InferenceRequest(
            operation=operation,
            function=function,
            transform=transform,
            stream=True,
            cleanup=True,
        )
        self._requests.put(request)
        try:
            while True:
                try:
                    kind, value = request.events.get(timeout=REQUEST_WAIT_S)
                except queue.Empty:
                    if request.done.is_set():
                        break
                    if not self._owner_thread.is_alive():
                        raise RuntimeError("MLX owner thread stopped unexpectedly")
                    continue
                if kind == "item":
                    yield value
                elif kind == "error":
                    raise value
                elif kind == "done":
                    break
        finally:
            # Stop/timeout closes this proxy. The owner notices between MLX
            # chunks, closes the underlying generator, and remains alive.
            request.cancel_event.set()
            self._wait_for_request(request)
        self._raise_request_error(request)

    def _cleanup_request_on_owner(self, request: _InferenceRequest):
        """Per-request cleanup that preserves the owner's live MLX streams."""
        if not request.cleanup:
            return
        try:
            gc.collect()
        except BaseException as exc:
            if request.error is None:
                request.error = exc
            _log_runtime_exception(
                operation=f"{request.operation}:gc_collect",
                owner_thread_id=request.owner_thread_id,
                caller_thread_id=request.caller_thread_id,
                exc=exc,
                traceback_text=traceback.format_exc(),
            )
        try:
            mx.clear_cache()
        except BaseException as exc:
            if request.error is None:
                request.error = exc
            _log_runtime_exception(
                operation=f"{request.operation}:clear_cache",
                owner_thread_id=request.owner_thread_id,
                caller_thread_id=request.caller_thread_id,
                exc=exc,
                traceback_text=traceback.format_exc(),
            )

    def _execute_sync_request(self, request: _InferenceRequest):
        try:
            request.result = request.function()
        except BaseException as exc:
            request.error = exc
            _log_runtime_exception(
                operation=request.operation,
                owner_thread_id=request.owner_thread_id,
                caller_thread_id=request.caller_thread_id,
                exc=exc,
                traceback_text=traceback.format_exc(),
            )
        finally:
            self._cleanup_request_on_owner(request)
            request.done.set()

    def _execute_stream_request(self, request: _InferenceRequest):
        inner = None
        try:
            inner = request.function()
            while not request.cancel_event.is_set():
                try:
                    raw = next(inner)
                except StopIteration:
                    break
                if not self._put_stream_event(request, ("item", request.transform(raw))):
                    break
        except BaseException as exc:
            request.error = exc
            _log_runtime_exception(
                operation=request.operation,
                owner_thread_id=request.owner_thread_id,
                caller_thread_id=request.caller_thread_id,
                exc=exc,
                traceback_text=traceback.format_exc(),
            )
        finally:
            try:
                if inner is not None and hasattr(inner, "close"):
                    inner.close()
            except BaseException as exc:
                if request.error is None and not request.cancel_event.is_set():
                    request.error = exc
                _log_runtime_exception(
                    operation=f"{request.operation}:close",
                    owner_thread_id=request.owner_thread_id,
                    caller_thread_id=request.caller_thread_id,
                    exc=exc,
                    traceback_text=traceback.format_exc(),
                )
            gc.collect()
            self._cleanup_request_on_owner(request)
            if request.error is not None and not request.cancel_event.is_set():
                self._put_stream_event(request, ("error", request.error))
            self._put_stream_event(request, ("done", None))
            request.done.set()

    def _owner_loop(self):
        self._owner_thread_id = threading.get_ident()
        self._owner_ready.set()
        try:
            while True:
                request = self._requests.get()
                if request is STOP_OWNER:
                    break
                request.owner_thread_id = self._owner_thread_id
                self._active_request = request
                if request.stream:
                    self._execute_stream_request(request)
                else:
                    self._execute_sync_request(request)
                self._active_request = None
        except BaseException as exc:
            self._shutdown_error = exc
            _log_runtime_exception(
                operation="owner_thread",
                owner_thread_id=self._owner_thread_id,
                caller_thread_id=None,
                exc=exc,
                traceback_text=traceback.format_exc(),
            )
        finally:
            try:
                self._shutdown_on_owner()
            except BaseException as exc:
                self._shutdown_error = exc
                _log_runtime_exception(
                    operation="owner_thread_shutdown",
                    owner_thread_id=self._owner_thread_id,
                    caller_thread_id=None,
                    exc=exc,
                    traceback_text=traceback.format_exc(),
                )
            self._owner_stopped.set()

    # ------------------------------------------------------------------
    # Configuration and lifecycle
    # ------------------------------------------------------------------

    def get_repo_id(self, model_type: str) -> str:
        """Build Hugging Face repo ID from model_type + size + quant."""
        variant = MODEL_VARIANTS[model_type]
        return REPO_TEMPLATE.format(
            size=self.model_size, variant=variant, quant=self.quantization
        )

    def get_model_state(self) -> tuple[str | None, str, str, bool]:
        """Return owner-safe model state for the Settings/status UI."""
        return self._submit_sync(
            "get_model_state",
            lambda: (
                self.current_model_type,
                self.model_size,
                self.quantization,
                self.jit_compile,
            ),
            cleanup=False,
        )

    def configure(self, model_size: str, quantization: str, jit_compile: bool) -> bool:
        """Apply model/JIT settings and unload on the owner thread if needed."""
        def apply():
            changed = (
                model_size != self.model_size
                or quantization != self.quantization
                or bool(jit_compile) != self.jit_compile
            )
            self.model_size = model_size
            self.quantization = quantization
            self.jit_compile = bool(jit_compile)
            if changed:
                self._unload_model_unlocked()
            return changed

        return self._submit_sync("configure", apply)

    def is_model_loaded(self, model_type: str) -> bool:
        """Return whether the owner currently has this model in memory."""
        return self._submit_sync(
            "is_model_loaded",
            lambda: self.current_model_type == model_type,
            cleanup=False,
        )

    def _shutdown_on_owner(self):
        """Unload all MLX state, then clear streams as the owner exits."""
        try:
            self._unload_model_unlocked()
            self._unload_audio_preprocessors_unlocked()
            gc.collect()
            mx.clear_cache()
        finally:
            # This is intentionally the only normal Studio call site. The
            # owner thread is actually terminating immediately after this.
            mx.clear_streams()

    def shutdown(self):
        """Stop the owner thread after queued work finishes/cancels safely."""
        with self._shutdown_lock:
            if self._owner_stopped.is_set():
                return
            if self._closing:
                already_closing = True
            else:
                already_closing = False
                self._closing = True
                if self._active_request is not None:
                    self._active_request.cancel_event.set()
                self._requests.put(STOP_OWNER)
        if already_closing:
            self._owner_stopped.wait()
        elif not self._on_owner_thread():
            self._owner_stopped.wait()
            self._owner_thread.join(timeout=10)
        if self._shutdown_error is not None:
            raise self._shutdown_error

    # ------------------------------------------------------------------
    # Owner-only model lifecycle and MLX operations
    # ------------------------------------------------------------------

    def _load_model(self, model_type: str):
        """Load model on the owner thread, unloading previous state first."""
        self._unload_asr_unlocked()
        if self.current_model_type == model_type:
            return
        self._unload_model_unlocked()
        mx.clear_cache()
        repo_id = self.get_repo_id(model_type)
        self.current_model = load_model(repo_id)
        self.current_model_type = model_type
        if self.jit_compile:
            self._apply_compile()
        mx.eval(self.current_model.parameters())

    def _apply_compile(self):
        """Wrap talker forward passes with mx.compile on the owner thread."""
        talker = getattr(self.current_model, "talker", None)
        if talker is None:
            return
        talker.__call__ = mx.compile(talker.__call__, shapeless=True)
        code_pred = getattr(talker, "code_predictor", None)
        if code_pred is not None:
            code_pred.__call__ = mx.compile(code_pred.__call__, shapeless=True)

    def _unload_model_unlocked(self):
        """Owner-only release of TTS and ASR model references."""
        self._unload_asr_unlocked()
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            gc.collect()

    def _unload_audio_preprocessors_unlocked(self):
        """Release optional MLX VAD/DeepFilter models on the owner thread."""
        from audio_utils import unload_deepfilter, unload_vad
        unload_deepfilter()
        unload_vad()

    # ------------------------------------------------------------------
    # Public TTS/ASR request proxies
    # ------------------------------------------------------------------

    def unload_model(self):
        return self._submit_sync("unload_model", self._unload_model_unlocked)

    def unload_audio_preprocessors(self):
        return self._submit_sync(
            "unload_audio_preprocessors", self._unload_audio_preprocessors_unlocked
        )

    def _generate_custom_voice_impl(self, text, speaker, language, instruct="", **kwargs):
        self._load_model("custom_voice")
        results = list(self.current_model.generate_custom_voice(
            text=text, speaker=speaker, language=language, instruct=instruct, **kwargs
        ))
        return self._to_numpy(results[0])

    def generate_custom_voice(self, text, speaker, language, instruct="", **kwargs) -> tuple:
        return self._submit_sync(
            "custom_voice", lambda: self._generate_custom_voice_impl(
                text, speaker, language, instruct, **kwargs
            )
        )

    def _generate_voice_design_impl(self, text, language, instruct, **kwargs):
        self._load_model("voice_design")
        seed = kwargs.pop("seed", None)
        if seed is not None:
            # This method is executed by _owner_loop.  Do not move RNG
            # mutation into the Gradio/AnyIO caller or into a reusable pool
            # thread: MLX random state is thread-affine here.
            self._seed_mlx_rng(seed)
        results = list(self.current_model.generate_voice_design(
            text=text, language=language, instruct=instruct, **kwargs
        ))
        return self._to_numpy(results[0])

    def generate_voice_design(self, text, language, instruct, **kwargs) -> tuple:
        return self._submit_sync(
            "voice_design", lambda: self._generate_voice_design_impl(
                text, language, instruct, **kwargs
            )
        )

    def _seed_mlx_rng(self, seed: int):
        """Seed MLX global RNG, enforcing owner-thread affinity."""
        if not self._on_owner_thread():
            raise RuntimeError("MLX RNG must be seeded on the owner thread")
        mx.random.seed(int(seed))

    def _prepare_ref(self, ref_audio_path, trim_ref, denoise_ref):
        """Owner-only reference preprocessing; VAD/DeepFilter stay affine."""
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

    def _generate_voice_clone_impl(
        self, text, ref_audio_path, ref_text, language="English",
        denoise_ref=False, trim_ref=False, **kwargs
    ):
        temps = []
        try:
            ref_audio_path, temps = self._prepare_ref(
                ref_audio_path, trim_ref, denoise_ref
            )
            self._load_model("base")
            results = list(self.current_model.generate(
                text=text, ref_audio=ref_audio_path, ref_text=ref_text,
                language=language, **kwargs
            ))
            return self._to_numpy(results[0])
        finally:
            self._cleanup_temps(temps)

    def generate_voice_clone(self, text, ref_audio_path, ref_text, language="English",
                             denoise_ref=False, trim_ref=False, **kwargs) -> tuple:
        return self._submit_sync(
            "voice_clone", lambda: self._generate_voice_clone_impl(
                text, ref_audio_path, ref_text, language,
                denoise_ref, trim_ref, **kwargs
            )
        )

    def _stream_custom_voice_impl(self, text, speaker, language, instruct="",
                                  streaming_interval=2.0, **kwargs):
        self._load_model("custom_voice")
        return self.current_model.generate_custom_voice(
            text=text, speaker=speaker, language=language, instruct=instruct,
            stream=True, streaming_interval=streaming_interval, **kwargs
        )

    def stream_generate_custom_voice(self, text, speaker, language, instruct="",
                                     streaming_interval=2.0, **kwargs):
        yield from self._submit_stream(
            "custom_voice_stream",
            lambda: self._stream_custom_voice_impl(
                text, speaker, language, instruct, streaming_interval, **kwargs
            ),
            self._to_numpy,
        )

    def _stream_voice_design_impl(self, text, language, instruct,
                                  streaming_interval=2.0, seed=None, **kwargs):
        self._load_model("voice_design")
        if seed is not None:
            # Qwen3-TTS exposes no seed keyword. Its sampler consumes MLX's
            # global PRNG state, so seed it immediately before inference on
            # this persistent owner thread.
            self._seed_mlx_rng(seed)
        return self.current_model.generate_voice_design(
            text=text, language=language, instruct=instruct,
            stream=True, streaming_interval=streaming_interval, **kwargs
        )

    def stream_generate_voice_design(self, text, language, instruct,
                                     streaming_interval=2.0, **kwargs):
        yield from self._submit_stream(
            "voice_design_stream",
            lambda: self._stream_voice_design_impl(
                text, language, instruct, streaming_interval, **kwargs
            ),
            self._to_numpy,
        )

    def stream_generate_voice_clone(self, text, ref_audio_path, ref_text,
                                    language="English", denoise_ref=False,
                                    trim_ref=False, streaming_interval=2.0, **kwargs):
        def make_stream():
            ref_audio_path_local, temps = self._prepare_ref(
                ref_audio_path, trim_ref, denoise_ref
            )
            try:
                self._load_model("base")
                inner = self.current_model.generate(
                    text=text, ref_audio=ref_audio_path_local, ref_text=ref_text,
                    language=language, stream=True,
                    streaming_interval=streaming_interval, **kwargs
                )
            except BaseException:
                self._cleanup_temps(temps)
                raise

            class TempCleanupStream:
                def __iter__(self):
                    return self

                def __next__(self):
                    return next(inner)

                def close(self):
                    try:
                        if hasattr(inner, "close"):
                            inner.close()
                    finally:
                        TTSEngine._cleanup_temps(temps)

            return TempCleanupStream()

        yield from self._submit_stream(
            "voice_clone_stream", make_stream, self._to_numpy
        )

    def _batch_generate_voice_clone_impl(self, texts, ref_audio_path, ref_text,
                                         language="English", denoise_ref=False,
                                         trim_ref=False, **kwargs):
        temps = []
        try:
            ref_audio_path, temps = self._prepare_ref(
                ref_audio_path, trim_ref, denoise_ref
            )
            self._load_model("base")
            results = list(self.current_model.batch_generate(
                texts=texts, ref_audio=ref_audio_path, ref_text=ref_text,
                lang_code=language, **kwargs
            ))
            results.sort(key=lambda r: r.sequence_idx)
            return [self._to_numpy(r) for r in results]
        finally:
            self._cleanup_temps(temps)

    def batch_generate_voice_clone(self, texts, ref_audio_path, ref_text,
                                   language="English", denoise_ref=False,
                                   trim_ref=False, **kwargs):
        return self._submit_sync(
            "voice_clone_batch", lambda: self._batch_generate_voice_clone_impl(
                texts, ref_audio_path, ref_text, language,
                denoise_ref, trim_ref, **kwargs
            )
        )

    def _batch_generate_custom_voice_impl(self, texts, speaker, language,
                                           instruct="", **kwargs):
        self._load_model("custom_voice")
        batch_size = len(texts)
        speakers = (list(speaker) if isinstance(speaker, (list, tuple))
                    else [speaker] * batch_size)
        instructs = (list(instruct) if isinstance(instruct, (list, tuple))
                     else [instruct] * batch_size)
        results = list(self.current_model.batch_generate(
            texts=texts, voices=speakers, instructs=instructs,
            lang_code=language, **kwargs
        ))
        results.sort(key=lambda r: r.sequence_idx)
        return [self._to_numpy(r) for r in results]

    def batch_generate_custom_voice(self, texts, speaker, language, instruct="", **kwargs):
        return self._submit_sync(
            "custom_voice_batch", lambda: self._batch_generate_custom_voice_impl(
                texts, speaker, language, instruct, **kwargs
            )
        )

    def _batch_generate_voice_design_impl(self, texts, language, instruct, **kwargs):
        self._load_model("voice_design")
        batch_size = len(texts)
        instructs = (list(instruct) if isinstance(instruct, (list, tuple))
                     else [instruct] * batch_size)
        results = list(self.current_model.batch_generate(
            texts=texts, instructs=instructs, lang_code=language, **kwargs
        ))
        results.sort(key=lambda r: r.sequence_idx)
        return [self._to_numpy(r) for r in results]

    def batch_generate_voice_design(self, texts, language, instruct, **kwargs):
        return self._submit_sync(
            "voice_design_batch", lambda: self._batch_generate_voice_design_impl(
                texts, language, instruct, **kwargs
            )
        )

    # ----- ASR -----

    def _load_asr(self):
        if self.asr_model is not None:
            return
        self._unload_model_unlocked()
        self.asr_model = load_stt_model(ASR_REPO_ID)
        mx.eval(self.asr_model.parameters())

    def _unload_asr_unlocked(self):
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None
            gc.collect()

    def unload_asr(self):
        return self._submit_sync("unload_asr", self._unload_asr_unlocked)

    def _transcribe_impl(self, audio_path, language="auto") -> str:
        self._load_asr()
        result = self.asr_model.generate(audio_path, language=language)
        return result.text

    def transcribe(self, audio_path, language="auto") -> str:
        def run():
            try:
                return self._transcribe_impl(audio_path, language)
            finally:
                self._unload_asr_unlocked()

        return self._submit_sync("asr", run)

    def stream_transcribe(self, audio_path, language="auto"):
        def make_stream():
            try:
                self._load_asr()
                lang = None if not language or language.lower() == "auto" else language
                inner = self.asr_model.generate(audio_path, language=lang, stream=True)
            except BaseException:
                self._unload_asr_unlocked()
                raise

            class TranscriptStream:
                def __iter__(self):
                    return self

                def __next__(self):
                    piece = next(inner)
                    text = getattr(piece, "text", None)
                    return text if text is not None else str(piece)

                def close(self):
                    try:
                        if hasattr(inner, "close"):
                            inner.close()
                    finally:
                        self_engine._unload_asr_unlocked()

            self_engine = self
            return TranscriptStream()

        yield from self._submit_stream(
            "asr_stream", make_stream, lambda text: text
        )

    @staticmethod
    def _to_numpy(result) -> tuple:
        """Convert MLX audio to numpy while still on the owner thread."""
        audio = np.array(result.audio, dtype=np.float32)
        sr = getattr(result, "sample_rate", 24000)
        return (sr, audio)
