# Voice Clone 生成路径与启动稳定性说明

正常单次 Voice Clone 现在使用已由主人验证的质量优先非流式路径：调用完整的
`generate_voice_clone()`，完成后一次性更新播放器。该路径不支持生成中的 Stop 或
分块 timeout；Voice Clone 的批量回退仍保留可取消的 2.0 秒流式路径。

内部流式路径仍保留通用的 1.0 秒间隔，并仅在 Voice Clone 单段回退时使用 2.0 秒间隔。
这些内部分块不会直接播放；Studio 会先累计完整音频，再一次性更新播放器。

这是基于真实 Apple Silicon 主机上的受控观察得出的工程选择：

- 1.0 秒流式：启动约前 0.5 秒出现严重的音色瞬态，随后才快速收敛到预期克隆女声。
- 非流式：该样本中的严重瞬态和起始呼吸/“ha”均消失。
- 2.0 秒流式：严重瞬态和起始呼吸均消失；声音从开头就稳定，主体音色与 1.0 秒流式结果相当。

因此 `VOICE_CLONE_STREAMING_INTERVAL_S = 2.0` 现在只应用于 Voice Clone 批量单段回退；
Voice Design、Custom Voice 的单段回退仍使用 1.0 秒路径。正常 Voice Clone 批量 API
仍是原有非流式路径。由于 Stop/timeout 只会在内部块之间检查，Clone 回退的 2.0 秒
间隔会略微降低取消和超时响应的时间粒度。

Kurisu 的生产 Clone reference 是本地确认的 7.45 秒 MEDIUM 片段：
`Kurisu_Production_Clone_Reference_A.wav`，SHA-256 为
`b3c4ea03803b3b7226d85c8ddc288e47caab4113b63513e5218abe722f5dbfbe`。
五次非流式主人复测相似度为 85/86/88/85/87，严重起始伪影为 0/5。
这项优势是启动可靠性，不代表所有机器或模型都必须使用非流式，也没有暴露通用
流式/非流式开关。

该结果是本项目当前模型与硬件上的 owner 验证记录，不是对所有机器、模型或 MLX-Audio
版本都适用的普遍缺陷结论。若未来需要真正的低延迟流式输出，应继续调查解码器的首块上下文，
而不是假定 2.0 秒对所有场景都最优。
