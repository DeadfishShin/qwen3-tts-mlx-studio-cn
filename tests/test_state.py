from state import AppSettings, AppContext


def test_settings_defaults_match_config():
    import config
    s = AppSettings()
    assert s.temperature == config.DEFAULT_TEMPERATURE
    assert s.top_k == config.DEFAULT_TOP_K
    assert s.max_tokens == config.DEFAULT_MAX_TOKENS
    assert s.output_dir == config.OUTPUT_DIR
    assert s.default_language == "English"


def test_gen_kwargs_shape():
    s = AppSettings()
    kw = s.gen_kwargs()
    assert set(kw) == {"temperature", "top_k", "top_p", "repetition_penalty", "max_tokens"}


def test_context_holds_parts(fake_engine, fake_history):
    ctx = AppContext(engine=fake_engine, library=None, history=fake_history,
                     yt=None, settings=AppSettings(), startup_warnings=[])
    assert ctx.engine is fake_engine
