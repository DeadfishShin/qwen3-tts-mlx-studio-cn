import generation
from generation import GenRequest, run_batch, run_single
from state import AppContext, AppSettings


class NoProgress:
    def __call__(self, *args, **kwargs):
        pass


def make_ctx(fake_engine, fake_history, tmp_path, **settings):
    app_settings = AppSettings()
    app_settings.output_dir = str(tmp_path / "out")
    app_settings.batch_size = 2
    for key, value in settings.items():
        setattr(app_settings, key, value)
    return AppContext(engine=fake_engine, library=None, history=fake_history,
                      yt=None, settings=app_settings)


def run_single_to_completion(ctx, request):
    return list(run_single(ctx, request))[-1]


def expected_sampler_kwargs(settings):
    return {
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "top_p": settings.top_p,
        "repetition_penalty": settings.repetition_penalty,
        "max_tokens": settings.max_tokens,
    }


def test_clone_uses_two_second_interval_and_preserves_clone_arguments(
        fake_engine, fake_history, tmp_path):
    ctx = make_ctx(
        fake_engine, fake_history, tmp_path,
        temperature=0.73, top_k=37, top_p=0.82,
        repetition_penalty=1.12, max_tokens=2048, denoise_ref=True,
    )
    request = GenRequest(
        mode="voice_clone", text="新的中文测试。", language="Chinese",
        ref_audio="/tmp/reference.wav", ref_text="准确的参考文本",
        trim_ref=False,
    )

    result = run_single_to_completion(ctx, request)

    assert result[0][0] == fake_engine.sr
    call = fake_engine.stream_calls[-1]
    assert call["method"] == "stream_generate_voice_clone"
    assert call["ref_audio_path"] == "/tmp/reference.wav"
    assert call["ref_text"] == "准确的参考文本"
    assert call["language"] == "Chinese"
    assert call["denoise_ref"] is True
    assert call["trim_ref"] is False
    assert call["kwargs"] == {
        **expected_sampler_kwargs(ctx.settings),
        "streaming_interval": 2.0,
    }
    assert not (tmp_path / "out").exists()


def test_voice_design_and_custom_voice_keep_one_second_interval(
        fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)

    run_single_to_completion(
        ctx,
        GenRequest(
            mode="voice_design", text="设计一段声音。", language="Chinese",
            voice_description="稳定、清晰的成年女性声音", random_seed=False,
            seed=123,
        ),
    )
    design_call = fake_engine.stream_calls[-1]
    assert design_call["method"] == "stream_generate_voice_design"
    assert design_call["kwargs"]["streaming_interval"] == 1.0
    assert design_call["kwargs"]["seed"] == 123

    run_single_to_completion(
        ctx,
        GenRequest(
            mode="custom_voice", text="自定义音色测试。", language="Chinese",
            speaker="serena", instruct="自然、平静",
        ),
    )
    custom_call = fake_engine.stream_calls[-1]
    assert custom_call["method"] == "stream_generate_custom_voice"
    assert custom_call["kwargs"] == {
        **expected_sampler_kwargs(ctx.settings),
        "streaming_interval": 1.0,
    }


def test_clone_batch_uses_batch_api_without_streaming_interval(
        fake_engine, fake_history, tmp_path):
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    request = GenRequest(
        mode="voice_clone", text="第一段。\n\n第二段。", language="Chinese",
        ref_audio="/tmp/reference.wav", ref_text="准确的参考文本",
        trim_ref=False,
    )

    outputs = list(run_batch(ctx, request, "paragraph", 300, NoProgress()))

    assert outputs[-1][2].startswith("已生成")
    call = fake_engine.batch_calls[-1]
    assert call["method"] == "batch_generate_voice_clone"
    assert call["ref_audio_path"] == "/tmp/reference.wav"
    assert call["ref_text"] == "准确的参考文本"
    assert call["language"] == "Chinese"
    assert call["denoise_ref"] is False
    assert call["trim_ref"] is False
    assert "streaming_interval" not in call["kwargs"]
    assert call["kwargs"] == expected_sampler_kwargs(ctx.settings)
    assert not (tmp_path / "out").exists()


def test_clone_batch_fallback_keeps_general_one_second_interval(
        fake_engine, fake_history, tmp_path):
    fake_engine.fail_batch = True
    ctx = make_ctx(fake_engine, fake_history, tmp_path)
    request = GenRequest(
        mode="voice_clone", text="第一段。\n\n第二段。", language="Chinese",
        ref_audio="/tmp/reference.wav", ref_text="准确的参考文本",
        trim_ref=False,
    )

    outputs = list(run_batch(ctx, request, "paragraph", 300, NoProgress()))

    assert outputs[-1][2].startswith("已生成")
    fallback_calls = [
        call for call in fake_engine.stream_calls
        if call["method"] == "stream_generate_voice_clone"
    ]
    assert len(fallback_calls) == 2
    assert all(call["kwargs"]["streaming_interval"] == 1.0
               for call in fallback_calls)
