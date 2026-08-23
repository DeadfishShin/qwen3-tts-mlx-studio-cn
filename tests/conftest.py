import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEngine:
    """Mimics engine.TTSEngine's public surface without loading models."""

    def __init__(self, sr=24000, fail_modes=None, fail_batch=False):
        self.sr = sr
        self.fail_modes = fail_modes or set()   # modes whose single-gen raises
        self.fail_batch = fail_batch            # batch methods raise if True
        self.model_size = "1.7B"
        self.quantization = "bf16"
        self.jit_compile = True
        self.current_model = None
        self.current_model_type = None
        self.calls = []                          # (method, args) log
        self.single_calls = []                   # structured blocking-call log
        self.stream_calls = []                   # structured stream-call log
        self.batch_calls = []                    # structured batch-call log
        self.n_chunks = 3                        # chunks per fake stream
        self.chunk_secs = 0.5                    # audio seconds per chunk
        self.chunk_hook = None                   # callable(i) before yielding chunk i

    def _audio(self, seconds=0.5):
        return (self.sr, np.zeros(int(self.sr * seconds), dtype=np.float32))

    def is_model_loaded(self, model_type):
        return self.current_model_type == model_type

    def get_repo_id(self, model_type):
        return f"fake/{model_type}-{self.model_size}-{self.quantization}"

    def generate_custom_voice(self, text, speaker, language, instruct="", **kw):
        self.calls.append(("generate_custom_voice", text, language))
        self.single_calls.append({
            "method": "generate_custom_voice", "text": text,
            "speaker": speaker, "language": language, "instruct": instruct,
            "kwargs": dict(kw),
        })
        if "custom_voice" in self.fail_modes:
            raise RuntimeError("fake single failure")
        self.current_model_type = "custom_voice"
        return self._audio()

    def generate_voice_design(self, text, language, instruct, **kw):
        self.calls.append(("generate_voice_design", text, language))
        self.single_calls.append({
            "method": "generate_voice_design", "text": text,
            "language": language, "instruct": instruct, "kwargs": dict(kw),
        })
        if "voice_design" in self.fail_modes:
            raise RuntimeError("fake single failure")
        self.current_model_type = "voice_design"
        return self._audio()

    def generate_voice_clone(self, text, ref_audio_path, ref_text, language="English",
                             denoise_ref=False, trim_ref=False, **kw):
        self.calls.append(("generate_voice_clone", text, language))
        self.single_calls.append({
            "method": "generate_voice_clone", "text": text,
            "ref_audio_path": ref_audio_path, "ref_text": ref_text,
            "language": language, "denoise_ref": denoise_ref,
            "trim_ref": trim_ref, "kwargs": dict(kw),
        })
        if "voice_clone" in self.fail_modes:
            raise RuntimeError("fake single failure")
        self.current_model_type = "base"
        return self._audio()

    def _stream(self, mode, model_type):
        if mode in self.fail_modes:
            raise RuntimeError("fake single failure")
        self.current_model_type = model_type
        for i in range(self.n_chunks):
            if self.chunk_hook:
                self.chunk_hook(i)
            yield (self.sr, np.zeros(int(self.sr * self.chunk_secs), dtype=np.float32))

    def stream_generate_custom_voice(self, text, speaker, language, instruct="", **kw):
        self.calls.append(("stream_generate_custom_voice", text, language))
        self.stream_calls.append({
            "method": "stream_generate_custom_voice", "text": text,
            "speaker": speaker, "language": language, "instruct": instruct,
            "kwargs": dict(kw),
        })
        yield from self._stream("custom_voice", "custom_voice")

    def stream_generate_voice_design(self, text, language, instruct, **kw):
        self.calls.append(("stream_generate_voice_design", text, language))
        self.stream_calls.append({
            "method": "stream_generate_voice_design", "text": text,
            "language": language, "instruct": instruct, "kwargs": dict(kw),
        })
        yield from self._stream("voice_design", "voice_design")

    def stream_generate_voice_clone(self, text, ref_audio_path, ref_text,
                                    language="English", denoise_ref=False,
                                    trim_ref=False, **kw):
        self.calls.append(("stream_generate_voice_clone", text, language))
        self.stream_calls.append({
            "method": "stream_generate_voice_clone", "text": text,
            "ref_audio_path": ref_audio_path, "ref_text": ref_text,
            "language": language, "denoise_ref": denoise_ref,
            "trim_ref": trim_ref, "kwargs": dict(kw),
        })
        yield from self._stream("voice_clone", "base")

    def stream_transcribe(self, audio_path, language="auto"):
        self.calls.append(("stream_transcribe", audio_path))
        yield "fake "
        yield "transcript"

    def batch_generate_voice_clone(self, texts, ref_audio_path, ref_text,
                                   language="English", denoise_ref=False,
                                   trim_ref=False, **kw):
        self.calls.append(("batch_generate_voice_clone", tuple(texts), language))
        self.batch_calls.append({
            "method": "batch_generate_voice_clone", "texts": tuple(texts),
            "ref_audio_path": ref_audio_path, "ref_text": ref_text,
            "language": language, "denoise_ref": denoise_ref,
            "trim_ref": trim_ref, "kwargs": dict(kw),
        })
        if self.fail_batch:
            raise RuntimeError("fake batch failure")
        self.current_model_type = "base"
        return [self._audio() for _ in texts]

    def batch_generate_custom_voice(self, texts, speaker, language, instruct="", **kw):
        self.calls.append(("batch_generate_custom_voice", tuple(texts), language))
        self.batch_calls.append({
            "method": "batch_generate_custom_voice", "texts": tuple(texts),
            "speaker": speaker, "language": language, "instruct": instruct,
            "kwargs": dict(kw),
        })
        if self.fail_batch:
            raise RuntimeError("fake batch failure")
        self.current_model_type = "custom_voice"
        return [self._audio() for _ in texts]

    def batch_generate_voice_design(self, texts, language, instruct, **kw):
        self.calls.append(("batch_generate_voice_design", tuple(texts), language))
        self.batch_calls.append({
            "method": "batch_generate_voice_design", "texts": tuple(texts),
            "language": language, "instruct": instruct, "kwargs": dict(kw),
        })
        if self.fail_batch:
            raise RuntimeError("fake batch failure")
        self.current_model_type = "voice_design"
        return [self._audio() for _ in texts]

    def unload_model(self):
        self.current_model_type = None

    def unload_asr(self):
        pass

    def transcribe(self, audio_path, language="auto"):
        self.calls.append(("transcribe", audio_path))
        return "fake transcript"


class FakeHistory:
    def __init__(self):
        self.entries = []

    def add(self, **kwargs):
        self.entries.append(kwargs)


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def fake_history():
    return FakeHistory()
