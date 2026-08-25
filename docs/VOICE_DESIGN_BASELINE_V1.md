# VOICE_DESIGN_BASELINE_V1

## 阶段身份

`VOICE_DESIGN_BASELINE_V1` 是当前 Qwen3-TTS 1.7B VoiceDesign「克里斯目标音色炼制」阶段的阶段性基准，不是 `FINAL_PRODUCTION_VOICE`，也不是对最终克里斯音色的宣称。

本收口冻结的目的，是为下一阶段 `Christina-TTS-1.5 isolated challenger validation` 提供一份可回退、可试听、可复现条件明确的固定 A/B 对照组。该阶段不再继续扩大 seed 或 prompt 搜索。

## Baseline 身份与证据链

| 字段 | 已核实值 |
|---|---|
| Baseline ID | `VOICE_DESIGN_BASELINE_V1` |
| History entry | `3701ed19fa69` |
| 生成时间 | `2026-08-23T17:33:04` |
| 实际 seed | `955911998` |
| 原始 seed mode | `random`（历史记录中的实际值） |
| History WAV | `/Users/mizukinamachi/Qwen3-TTS/studio/outputs/history/3701ed19fa69.wav` |
| owner 保存副本 | `/Users/mizukinamachi/Qwen3-TTS/output/design_20260823_173304-955911998-m.wav` |
| 稳定 Master 归档 | `/Users/mizukinamachi/Qwen3-TTS/master_voice/Kurisu_Master_Timbre_A.wav` |
| 三份 WAV 的 SHA-256 | `78a38fad92ec24babc43235f55f0ccffd70bcaa64216b787e4474ede901729eb` |
| WAV 属性 | 24,000 Hz、单声道、PCM16、28.32 s、679,680 frames |
| owner 状态 | `confirmed` |
| owner 评分记录 | `95` |

证据来源为 active History index、History 实际 WAV、owner 保存副本，以及 `/Users/mizukinamachi/Qwen3-TTS/master_voice/Kurisu_Master_Timbre_A.json`。三份 WAV 已逐字节 SHA-256 一致；Master 元数据中的 `history_entry_id`、timestamp、seed、文本、prompt、采样参数和模型仓库与 History 记录一致。

同一数值 seed 在历史中还出现于其他文本、语言、风格指令和 seed mode 组合。因此，`955911998` 单独不是稳定的跨文本音色 ID。本 Baseline 必须由 `history_entry_id + exact text + exact prompt + exact parameters + WAV SHA-256` 共同定义。

## 当前目标音色定义

- 年轻成年女性，约 20 岁年龄感；
- 成年女性中高音区；
- 清澈、明亮、干净、通透；
- 声音核心稳定，有自然共鸣、支撑和实体感；
- 少气声，不幼态、不奶声、不甜腻、不软糯；
- 略带锋利感，聪明、敏锐、自信；
- 理性、知性、反应快，有自己的判断；
- 情绪总体克制，外冷内柔；
- 面向信任对象时允许从语气、停顿和句尾自然流露少量温和；
- 不追求机械复刻今井麻美原声；目标是保留牧濑红莉栖式的气质方向，同时形成可用的本地阶段基准。

## 完整输入文本

```text
等一下，你这个结论是怎么得出来的？……不，我不是说一定有问题，只是这里少了一个必要条件。如果前提不成立，后面的推导当然也不会成立。
所以先别急着下结论。把数据给我，我陪你重新看一遍。
……干嘛这样看着我？我只是觉得，既然已经做到这里了，因为这种小问题放弃也太可惜了吧。
```

## 完整 VoiceDesign prompt

以下内容从 `index.json` 与 `Kurisu_Master_Timbre_A.json` 原始记录恢复，作为 VoiceDesign 的完整 identity description。该候选的 `style_instruction` 为空，因此当前链路将 identity description 原样作为唯一 instruction 通道输入。

```text
一位年轻成年女性，声音年龄感约20岁左右。声线清澈、明亮，音高处于成年女性的中高区域，但绝不能幼态、奶声奶气或甜腻。
声音具有清晰稳定的核心和自然的共鸣，发声有支撑、有实体感，虽然明亮但并不轻飘。避免过多气声和软糯感。声音听起来干净、通透、略带一点锋利感，能够自然传递聪明、敏锐和自信。
她给人的第一印象应该是知性、理性、头脑转得很快，并且有自己的判断。说话时带有轻微的戒备感和克制的强势，不习惯刻意讨好别人，也不会通过可爱或撒娇获得亲近感。
她并不冷漠。隐藏在理性和稍显锋利的外表下面，有自然的善意和柔软。面对真正熟悉和信任的人时，这种柔和会偶尔从语气、停顿和句尾自然流露出来，但幅度很小，不要突然变成甜妹或撒娇声。
语速中等略快，体现思考速度快和反应敏捷，但必须保持从容和控制感。不要像赶时间一样连续快速输出。语义重要的位置自然减速，逻辑转折处有短暂停顿，复杂内容表达清晰果断。整体感觉应该是“脑子转得很快”，而不是“嘴巴说得很急”。
发音清楚但保持真实会话感，不要逐字用力，不要播音式字正腔圆。句子可以有自然的轻重音和细微音高变化。陈述事实时冷静直接，反驳别人时略带锋芒，表达关心时则自然降低攻击性。
避免幼女音、妹妹音、奶音、夹子音、甜妹音、软糯音和刻意可爱的动漫腔。避免过分气声、鼻音和持续偏高的娇柔语调。
同时避免成熟御姐音、低沉女主播音、新闻播音腔、职业客服腔以及故意压低嗓音制造成熟感。
整体声音应该呈现：
年轻、清澈、明亮、有芯、知性、敏锐、自信、略带锋芒；不是靠低音表现成熟，而是靠稳定的发声、清晰的逻辑和受控的表达体现可靠感。
```

## 完整生成参数

| 参数 | 原始记录 |
|---|---|
| mode | `voice_design` |
| language | `Auto-detect` |
| temperature | `0.9` |
| top_k | `50` |
| top_p | `1.0` |
| repetition_penalty | `1.05` |
| max_tokens | `4096` |
| style_instruction | 空字符串 |
| actual seed | `955911998` |
| seed_mode | `random` |
| output sample rate | `24000` |
| output channels | `1` |
| output duration | `28.32 s` |

`seed_mode=random` 是该 History 记录的真实值，不能被本收口文档改写为 `fixed`。Studio 当前实现会在随机模式下先解析并保存一个数值 seed；若未来需要做受控复现，应在不改变其他条件的前提下显式使用已保存的数值 `955911998` 作为 fixed seed，并记录这是“使用已解析 seed 的复现”，而非重新进行随机抽样。当前任务无需为了收口再生成音频。

## 当前运行现实

### 启动与 Studio

- Primary launcher: `/Users/mizukinamachi/Qwen3-TTS/启动Qwen3-TTS.command`
- Studio 工作目录: `/Users/mizukinamachi/Qwen3-TTS/studio`
- Launcher 使用的 Python: `/Users/mizukinamachi/Qwen3-TTS/studio/.venv/bin/python`
- Launcher 会先读取 `/Users/mizukinamachi/Qwen3-TTS/huggingface-env.sh`，使用共享 HF cache。
- `HISTORY_DIR` 配置值为 `./outputs/history`，按 Studio 工作目录解析为 `/Users/mizukinamachi/Qwen3-TTS/studio/outputs/history`。
- Active history index: `/Users/mizukinamachi/Qwen3-TTS/studio/outputs/history/index.json`
- 当前 index 有 50 条记录；其中 seed `955911998` 的 VoiceDesign 记录有 12 条。

### 模型

- Model repository: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- Local HF snapshot: `/Users/mizukinamachi/Qwen3-TTS/models/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/snapshots/7d3824abff87e49756bb0f83fb5411de75d160c4`
- Local snapshot revision: `7d3824abff87e49756bb0f83fb5411de75d160c4`
- Snapshot `config.json`: `tts_model_type=voice_design`, `tts_model_size=1b`, architecture `Qwen3TTSForConditionalGeneration`。
- 当前 Studio 配置默认值为 `model_size=1.7B`、`quantization=bf16`；模型仓库映射由 `REPO_TEMPLATE` 与 `MODEL_VARIANTS["voice_design"]` 解析。

### 依赖环境

以下版本从当前 Studio `.venv` 只读查询，未在本任务中安装或升级：

| 组件 | 版本 |
|---|---|
| Python | `3.11.7` |
| mlx | `0.32.1` |
| mlx-metal | `0.32.1` |
| mlx-audio | `0.4.8` |
| transformers | `5.15.1` |
| huggingface-hub | `1.28.0` |
| tokenizers | `0.22.2` |
| sentencepiece | `0.2.2` |
| gradio | `6.25.0` |
| anyio | `4.14.2` |
| numpy | `2.4.6` |
| soundfile | `0.14.0` |
| safetensors | `0.8.0` |
| pyyaml | `6.0.3` |
| tqdm | `4.70.0` |

Launcher 环境变量：

- `HF_HOME=/Users/mizukinamachi/Qwen3-TTS/models/huggingface`
- `HF_HUB_CACHE=/Users/mizukinamachi/Qwen3-TTS/models/huggingface/hub`
- `HUGGINGFACE_HUB_CACHE=/Users/mizukinamachi/Qwen3-TTS/models/huggingface/hub`
- `PYTHONNOUSERSITE=1`

## 当前启动与推理链

静态导入检查通过：`config`、`history`、`engine`、`generation`、`app`、`mlx_audio` 和 `mlx` 均可解析。启动检查通过，Studio 在 scratch port `7897` 返回 `LAUNCH OK`；该检查没有加载 VoiceDesign 权重，也没有执行 TTS。

当前单条 VoiceDesign 路由为：

1. `voice_design` UI 构造 `GenRequest`；
2. `prepare_voice_design_seed` 保存实际 numeric seed；
3. `TTSEngine._load_model("voice_design")` 解析上述 HF repository；
4. `TTSEngine._generate_voice_design_impl` 在 MLX owner thread 设置 seed；
5. `current_model.generate_voice_design(text=..., language=..., instruct=..., ...)` 生成；
6. History 保存 text、description、style、seed、seed mode 和 sampler 参数。

由于该 Baseline 的 `style_instruction` 为空，`compose_voice_design_instruct` 返回完整 description 原文，没有额外风格层包装。

## 可复现步骤

不需要为了本次收口重新生成音频。未来若需复现，必须：

1. 使用同一 primary launcher、Studio 工作目录、`.venv` 和共享 HF cache；
2. 使用同一 VoiceDesign snapshot/repository 与 `1.7B/bf16` 配置；
3. 使用上面完整输入文本与完整 prompt，`style_instruction` 保持为空；
4. 使用 `Auto-detect`、`temperature=0.9`、`top_k=50`、`top_p=1.0`、`repetition_penalty=1.05`、`max_tokens=4096`；
5. 使用已保存 numeric seed `955911998` 的 fixed-seed 受控复现方式，并将其明确标记为基于原始 random-mode 记录的 resolved-seed replay；
6. 将新输出与 Baseline WAV 的 SHA-256、时长和听感分别比较，不覆盖 History 或 Master 归档。

Studio 自身已说明固定 seed 目标是相同 MLX RNG 序列，但未承诺跨未来 Apple GPU/依赖版本逐 bit 一致。因此本 Baseline 的首要复现契约是完整参数、模型快照、原始文本、prompt、resolved seed 和资产 SHA 的证据闭环。

## 已完成工作与资产保护

- 已定位并核实 Baseline 的实际 History WAV、owner 副本和 Master 归档。
- 已验证三份 WAV SHA-256 一致，未复制、移动、重命名或删除音频。
- 已恢复并归档完整 VoiceDesign prompt、完整文本、采样参数、seed mode、模型和 timestamp。
- 已记录当前 launcher、Studio、History、HF snapshot 与依赖环境。
- 已进行静态导入和 Studio 启动可解析性检查；未加载模型权重，未生成新音频。
- 未安装、下载、转换或运行 Christina-TTS。
- 未升级 Python、MLX、mlx-audio、模型或其他依赖。
- 未修改生产 launcher、History/index、WAV、metadata 或 Studio 运行逻辑。

## 已知局限

1. Baseline 是当前阶段最佳已确认候选，不是最终生产声音。
2. 数值 seed 不能脱离 text、prompt、language、style 和 sampler 参数单独代表音色。
3. 原始记录是 `seed_mode=random`；固定 seed 是未来受控复现方法，不应反向改写历史事实。
4. 固定 seed 与当前模型/MLX 只提供受控复现目标，不承诺未来环境 bit-identical。
5. `owner_rating=95` 是当前 owner 记录，不替代后续产品级长期听测。

## 为什么在此停止 VoiceDesign 搜索

当前候选已经有 owner-confirmed 资产、稳定归档、完整原始参数和可审计的 History 对应关系。继续无边界搜索 seed 或 prompt 会改变对照组、扩大实验面，并使下一阶段无法区分是模型路线变化还是 VoiceDesign 基线变化。因此从现在起，`VOICE_DESIGN_BASELINE_V1` 冻结；任何新 VoiceDesign 探索必须另建实验记录，不能覆盖或污染本 Baseline。

## 下一阶段：Christina-TTS-1.5 isolated challenger validation

允许开展的下一阶段是 `Christina-TTS-1.5 isolated challenger validation`。必须：

- 使用独立目录和优先独立 Python/MLX 环境；
- 不覆盖当前 VoiceDesign 模型、HF snapshot、Master WAV 或 History；
- 不修改 `/Users/mizukinamachi/Qwen3-TTS/启动Qwen3-TTS.command`；
- 使用固定测试语料与本 Baseline 做 A/B；
- 在候选路线被接受前，保持 Baseline 可回退、可运行、可试听；
- 不把 challenger 的产物直接写入现有生产输出或 Studio。

本任务不实施 Christina-TTS 部署、安装、下载、转换或运行。

## Git 与收口状态

文档属于现有 Studio Git 仓库；本次只允许新增本收口文档，不修改无关源码。提交前后应验证 Studio working tree 只包含本文件的预期变更，且不 push 第三方或生产远端。
