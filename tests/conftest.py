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

    def _audio(self, seconds=0.5):
        return (self.sr, np.zeros(int(self.sr * seconds), dtype=np.float32))

    def is_model_loaded(self, model_type):
        return self.current_model_type == model_type

    def get_repo_id(self, model_type):
        return f"fake/{model_type}-{self.model_size}-{self.quantization}"

    def generate_custom_voice(self, text, speaker, language, instruct="", **kw):
        self.calls.append(("generate_custom_voice", text))
        if "custom_voice" in self.fail_modes:
            raise RuntimeError("fake single failure")
        self.current_model_type = "custom_voice"
        return self._audio()

    def generate_voice_design(self, text, language, instruct, **kw):
        self.calls.append(("generate_voice_design", text))
        if "voice_design" in self.fail_modes:
            raise RuntimeError("fake single failure")
        self.current_model_type = "voice_design"
        return self._audio()

    def generate_voice_clone(self, text, ref_audio_path, ref_text, language="English",
                             denoise_ref=False, trim_ref=False, **kw):
        self.calls.append(("generate_voice_clone", text))
        if "voice_clone" in self.fail_modes:
            raise RuntimeError("fake single failure")
        self.current_model_type = "base"
        return self._audio()

    def batch_generate_voice_clone(self, texts, ref_audio_path, ref_text,
                                   language="English", denoise_ref=False,
                                   trim_ref=False, **kw):
        self.calls.append(("batch_generate_voice_clone", tuple(texts)))
        if self.fail_batch:
            raise RuntimeError("fake batch failure")
        self.current_model_type = "base"
        return [self._audio() for _ in texts]

    def batch_generate_custom_voice(self, texts, speaker, language, instruct="", **kw):
        self.calls.append(("batch_generate_custom_voice", tuple(texts)))
        if self.fail_batch:
            raise RuntimeError("fake batch failure")
        self.current_model_type = "custom_voice"
        return [self._audio() for _ in texts]

    def batch_generate_voice_design(self, texts, language, instruct, **kw):
        self.calls.append(("batch_generate_voice_design", tuple(texts)))
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
