import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import config
import engine


class FakeResult:
    sample_rate = 24000

    def __init__(self, value=0.0):
        self.audio = np.full(8, value, dtype=np.float32)


class FakeStream:
    def __init__(self, results):
        self.results = iter(results)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.results)

    def close(self):
        self.closed = True


class FakeTalker:
    def __init__(self, thread_ids):
        self.thread_ids = thread_ids
        self.code_predictor = None

    def __call__(self, *args, **kwargs):
        self.thread_ids.append(threading.get_ident())


class FakeModel:
    def __init__(self):
        self.thread_ids = []
        self.load_parameter_thread_ids = []
        self.streams = []
        self.fail_next = False
        self.active = 0
        self.max_active = 0
        self.talker = None

    def parameters(self):
        self.load_parameter_thread_ids.append(threading.get_ident())
        return {}

    def generate_voice_design(self, **kwargs):
        self.thread_ids.append(threading.get_ident())
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("fake inference failure")
            if kwargs.get("stream"):
                stream = FakeStream([FakeResult(1.0), FakeResult(2.0)])
                self.streams.append(stream)
                return stream
            time.sleep(0.002)
            return [FakeResult(1.0)]
        finally:
            self.active -= 1


@pytest.fixture
def owner_engine(monkeypatch):
    events = []
    model = FakeModel()
    monkeypatch.setattr(engine, "load_model", lambda repo_id: model)
    monkeypatch.setattr(engine.mx, "eval", lambda value: events.append(("eval", threading.get_ident())))
    monkeypatch.setattr(engine.mx, "clear_cache", lambda: events.append(("cache", threading.get_ident())))
    monkeypatch.setattr(engine.mx, "clear_streams", lambda: events.append(("streams", threading.get_ident())))
    eng = engine.TTSEngine()
    try:
        yield eng, model, events
    finally:
        eng.shutdown()


def test_owner_thread_is_persistent_across_ten_requests(owner_engine):
    eng, model, events = owner_engine
    caller_ids = set()

    def request(i):
        caller_ids.add(threading.get_ident())
        sr, audio = eng.generate_voice_design(f"hello {i}", "Chinese", "calm")
        return sr, len(audio)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(request, range(10)))

    assert all(sr == 24000 and length == 8 for sr, length in results)
    assert len(caller_ids) >= 2
    assert eng.owner_thread_id not in caller_ids
    assert set(model.thread_ids) == {eng.owner_thread_id}
    assert set(model.load_parameter_thread_ids) == {eng.owner_thread_id}
    assert eng.owner_thread_alive
    assert not any(kind == "streams" for kind, _ in events)
    assert sum(kind == "cache" for kind, _ in events) >= 10


def test_request_serialization_is_owner_thread_serialization(owner_engine):
    eng, model, _ = owner_engine

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda i: eng.generate_voice_design(str(i), "Chinese", "calm"),
            range(20),
        ))

    assert model.max_active == 1


def test_exception_is_logged_owner_survives_and_next_request_runs(owner_engine):
    eng, model, _ = owner_engine
    model.fail_next = True

    with pytest.raises(RuntimeError, match="fake inference failure"):
        eng.generate_voice_design("first", "Chinese", "calm")

    result = eng.generate_voice_design("second", "Chinese", "calm")

    assert result[0] == 24000
    assert eng.owner_thread_alive
    assert set(model.thread_ids) == {eng.owner_thread_id}
    assert engine.os.path.exists(engine.RUNTIME_LOG_PATH)
    with open(engine.RUNTIME_LOG_PATH, encoding="utf-8") as log_file:
        log = log_file.read()
    assert "operation=voice_design" in log
    assert "fake inference failure" in log
    assert f"owner_thread_id={eng.owner_thread_id}" in log


def test_cancellation_closes_owner_stream_without_poisoning_next_request(owner_engine):
    eng, model, events = owner_engine
    stream = eng.stream_generate_voice_design("stream", "Chinese", "calm")

    first_sr, first_audio = next(stream)
    stream.close()
    result = eng.generate_voice_design("after cancel", "Chinese", "calm")

    assert first_sr == 24000 and len(first_audio) == 8
    assert model.streams[0].closed
    assert result[0] == 24000
    assert eng.owner_thread_alive
    assert not any(kind == "streams" for kind, _ in events)


def test_voice_design_seed_is_applied_on_persistent_owner_thread(owner_engine, monkeypatch):
    eng, _, _ = owner_engine
    seed_events = []
    monkeypatch.setattr(
        engine.TTSEngine,
        "_seed_mlx_rng",
        lambda self, seed: seed_events.append((seed, threading.get_ident())),
    )

    stream = eng.stream_generate_voice_design(
        "seeded", "Chinese", "calm", seed=123456
    )
    list(stream)

    assert seed_events == [(123456, eng.owner_thread_id)]


def test_cancellation_does_not_poison_following_seeded_request(owner_engine, monkeypatch):
    eng, _, _ = owner_engine
    seed_events = []
    monkeypatch.setattr(
        engine.TTSEngine,
        "_seed_mlx_rng",
        lambda self, seed: seed_events.append((seed, threading.get_ident())),
    )

    stream = eng.stream_generate_voice_design(
        "cancelled", "Chinese", "calm", seed=1
    )
    next(stream)
    stream.close()

    next_result = eng.generate_voice_design(
        "next", "Chinese", "calm", seed=2
    )
    assert next_result[0] == 24000
    assert seed_events == [(1, eng.owner_thread_id), (2, eng.owner_thread_id)]
    assert eng.owner_thread_alive


def test_clear_cache_is_per_request_and_clear_streams_is_owner_shutdown_only(owner_engine):
    eng, _, events = owner_engine
    eng.generate_voice_design("one", "Chinese", "calm")
    eng.generate_voice_design("two", "Chinese", "calm")

    before_shutdown = list(events)
    assert not any(kind == "streams" for kind, _ in before_shutdown)

    owner_id = eng.owner_thread_id
    eng.shutdown()

    assert events[-2:] == [("cache", owner_id), ("streams", owner_id)]
    assert not eng.owner_thread_alive


def test_shutdown_cancels_active_stream_and_joins_owner(owner_engine):
    eng, model, _ = owner_engine
    stream = eng.stream_generate_voice_design("shutdown", "Chinese", "calm")
    next(stream)

    eng.shutdown()
    stream.close()

    assert model.streams[0].closed
    assert not eng.owner_thread_alive


def test_model_load_eval_and_explicit_jit_compile_stay_on_owner(monkeypatch):
    model = FakeModel()
    model_thread_ids = []
    model.talker = FakeTalker(model_thread_ids)
    events = []
    monkeypatch.setattr(engine, "load_model", lambda repo_id: model)
    monkeypatch.setattr(engine.mx, "eval", lambda value: events.append(("eval", threading.get_ident())))
    monkeypatch.setattr(engine.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(engine.mx, "clear_streams", lambda: None)
    monkeypatch.setattr(
        engine.mx,
        "compile",
        lambda function, **kwargs: (events.append(("compile", threading.get_ident())) or function),
    )
    eng = engine.TTSEngine()
    eng.jit_compile = True
    try:
        eng.generate_voice_design("jit", "Chinese", "calm")
        owner_id = eng.owner_thread_id
        assert model_thread_ids == []
        assert all(thread_id == owner_id for _, thread_id in events)
    finally:
        eng.shutdown()


def test_default_jit_remains_off_and_owner_thread_is_not_a_daemon(owner_engine):
    eng, _, _ = owner_engine
    assert config.ENABLE_JIT_COMPILE is False
    assert eng.jit_compile is False
    assert eng._owner_thread.daemon is False
