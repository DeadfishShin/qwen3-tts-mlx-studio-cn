"""All user-facing text, one string per concept.

Keep display labels here and keep backend values in the value side of the
explicit choice mappings below. Internal mode IDs, model IDs, and speaker IDs
are intentionally not translated.
"""
from config import LANGUAGE_AUTO as CONFIG_LANGUAGE_AUTO, LANGUAGES as CONFIG_LANGUAGES


# --- Display-label/value mappings ---
_LANGUAGE_LABELS = {
    "Auto-detect": "自动检测",
    "English": "英语",
    "Chinese": "中文",
    "Japanese": "日语",
    "Korean": "韩语",
    "German": "德语",
    "French": "法语",
    "Russian": "俄语",
    "Portuguese": "葡萄牙语",
    "Spanish": "西班牙语",
    "Italian": "意大利语",
}
LANGUAGE_CHOICES = [
    (_LANGUAGE_LABELS.get(value, value), value)
    for value in [CONFIG_LANGUAGE_AUTO] + CONFIG_LANGUAGES
]
LANGUAGE_AUTO = _LANGUAGE_LABELS.get(CONFIG_LANGUAGE_AUTO, "自动检测")
LANGUAGE_AUTO_VALUE = CONFIG_LANGUAGE_AUTO

BATCH_SPLIT_CHOICES = [("按段落", "paragraph"), ("按句子", "sentence"), ("按行", "line")]
SM_MODE_CHOICES = [
    ("自定义音色", "custom_voice"),
    ("声音设计", "voice_design"),
    ("声音克隆", "voice_clone"),
]
SET_MODEL_SIZE_CHOICES = [("小型（0.6B）", "0.6B"), ("大型（1.7B）", "1.7B")]
SET_QUANT_CHOICES = [
    ("4-bit（低内存）", "4bit"),
    ("6-bit", "6bit"),
    ("8-bit", "8bit"),
    ("BF16（最高质量）", "bf16"),
]
SET_PRESET_CHOICES = [
    ("均衡", "Balanced"),
    ("创意", "Creative"),
    ("精确", "Precise"),
    ("自定义", "Custom"),
]
SET_EXPORT_FORMAT_CHOICES = [("WAV", "wav"), ("MP3", "mp3"), ("OGG", "ogg")]
NO_LIBRARY_VOICE = "不使用声音库"


# --- App shell ---
APP_TITLE = "Qwen3-TTS MLX Studio"
APP_HEADER_HTML = (
    "<div class='app-header'>"
    "<h1>Qwen3-TTS MLX Studio</h1>"
    "<p class='subtitle'>本地 AI 文字转语音 · MLX · Apple Silicon</p>"
    "</div>"
)
STATUS_READY = "就绪"
STATUS_WARNINGS = "警告：{warnings}"

# --- Shared generation inputs ---
TEXT_TO_SPEAK = "要朗读的文本"
TEXT_PLACEHOLDER = "输入要朗读的文本……"
TIP_TEXT_LENGTH = (
    "_提示：1–4 句话效果较好。过长或数字较多的文本可能持续运行；"
    "可以点击“停止”，或等待自动停止超时并保留已生成的音频。_"
)
LANGUAGE = "语言"
GENERATE = "生成"

# --- Voice cloning extras ---
TRIM_REF_LABEL = "裁剪参考音频静音（推荐）"
REP_PENALTY_CLONE_INFO = "声音克隆始终使用不低于 1.5 的重复惩罚"

# --- Shared output column ---
OUTPUT = "生成的音频"
SAVE_AUDIO = "保存音频"
SAVE_PATH_PLACEHOLDER = "保存路径将在此显示……"

# --- Batch accordion ---
BATCH_ACCORDION = "批量模式"
SPLIT_MODE = "文本切分方式"
SILENCE_GAP = "片段之间的静音（毫秒）"
GENERATE_BATCH = "批量生成"
BATCH_RESULTS = "片段"
BATCH_TABLE_HEADERS = ["序号", "文本", "状态"]
COMBINED_OUTPUT = "生成的音频（合并）"
SAVE_COMBINED = "保存合并音频"
BATCH_STATUS = "批量状态"

# --- Save-to-library accordion ---
LIB_SAVE_ACCORDION = "保存到声音库"
VOICE_NAME = "声音名称"
SAVE_VOICE_TO_LIBRARY = "保存声音到声音库"
LIB_STATUS_PLACEHOLDER = "声音库状态……"

# --- Custom Voice tab ---
TAB_CUSTOM_VOICE = "自定义音色"
CV_SPEAKER = "声音"
CV_INSTRUCT = "风格指令（可选）"
CV_INSTRUCT_PLACEHOLDER = "例如：温暖地说话，语气兴奋……"

# --- Voice Design tab ---
TAB_VOICE_DESIGN = "声音设计"
VD_DESCRIPTION = "声音描述"
VD_INSTRUCT = VD_DESCRIPTION  # Compatibility alias for integrations using the old name.
VD_DESCRIPTION_PLACEHOLDER = "例如：低沉、沉稳的男性旁白，带有英式口音"
VD_INSTRUCT_PLACEHOLDER = VD_DESCRIPTION_PLACEHOLDER
VD_STYLE = "风格指令（可选）"
VD_STYLE_PLACEHOLDER = "例如：语速稍慢、语气沉稳自然，逻辑转折处适当停顿"
VD_LIB_NAME_PLACEHOLDER = "我的设计声音"
VD_INFO = (
    "**声音描述**：相对稳定的说话人特征，例如音色、年龄感、性格和口音。\n\n"
    "**风格指令**：本次生成的表达方式，例如语速、情绪、停顿和能量。\n\n"
    "本地 VoiceDesign 模型只有一个 instruction 通道，生成时会将两项合并为同一条模型指令；"
    "风格提示也可能对音色产生轻微影响。留空风格指令时，行为与原 Voice Design 路径保持一致。"
)
VD_IDENTITY_HEADER = "声音身份："
VD_STYLE_HEADER = "本次说话风格："
VD_PROMPT_FORMAT = "{identity}\n{description}\n\n{style}\n{style_instruction}"
VD_SEED_HEADER = "随机种子模式"
VD_RANDOM_EACH = "每次随机"
VD_SEED = "随机种子"
VD_SEED_INFO = (
    "勾选“每次随机”进行抽卡；取消勾选后使用输入的固定整数种子，"
    "范围为 0–4294967295。种子控制适用于单条声音设计；批量模式保持当前连续采样，"
    "不会为每个片段重复重置同一个种子。"
)
VD_USE_LAST_SEED = "使用上次种子"
VD_NO_LAST_SEED = "还没有可复用的声音设计种子。"
VD_SEED_USED = "本次使用的随机种子：{seed}"
VD_SEED_INVALID = "随机种子必须是 0–4294967295 范围内的整数"
VD_SEED_INVALID_WARN = "随机种子无效，请输入 0–4294967295 范围内的整数。"
VD_SAVE_NO_AUDIO_WARN = "请先生成音频，再保存到声音库。"
VD_SAVE_NO_AUDIO = "没有可保存的音频"
VD_SAVE_NO_NAME_WARN = "请输入声音名称。"
VD_SAVE_NO_NAME = "请输入声音名称"
VD_SAVED = "声音“{name}”已保存到声音库"
VD_BATCH_DESCRIPTION_REQUIRED_WARN = "请填写声音描述。"
VD_BATCH_DESCRIPTION_REQUIRED = "请先填写声音描述"
VD_DESCRIPTION_HISTORY_LABEL = "声音描述"
VD_STYLE_HISTORY_LABEL = "风格指令"

# --- Voice Clone tab ---
TAB_VOICE_CLONE = "声音克隆"
VC_NOTICE_HTML = (
    "<div class='info-notice'>"
    "<strong>参考文本必须与音频中实际说出的内容完全一致。</strong>"
    "请使用清晰的 3–30 秒音频片段以获得更好效果。"
    "</div>"
)
VC_LIBRARY_VOICE = "从声音库加载"
VC_REF_AUDIO = "参考音频"
VC_TRANSCRIBE = "转写参考音频"
VC_TRANSCRIBE_HINT_HTML = "<div class='text-hint'>使用设备上的语音识别自动填写参考文本</div>"
VC_REF_TEXT = "参考文本（必填）"
VC_REF_TEXT_PLACEHOLDER = "参考音频中实际说出的原文"
VC_LIB_NAME_PLACEHOLDER = "我的克隆声音"
VC_NO_REF_AUDIO_WARN = "请先上传参考音频。"
VC_NO_REF_AUDIO = "没有参考音频"
VC_SAVE_NO_AUDIO_WARN = "没有可保存的参考音频。"
VC_SAVE_NO_AUDIO = "没有参考音频"
VC_SAVE_NO_NAME_WARN = "请输入声音名称。"
VC_SAVE_NO_NAME = "请输入声音名称"
VC_SAVE_NO_TEXT_WARN = "参考文本为必填项。"
VC_SAVE_NO_TEXT = "请输入参考文本"
VC_SAVED = "声音“{name}”已保存到声音库"

# --- Transcription tab ---
TAB_TRANSCRIPTION = "音频转写"
ASR_NOTICE_HTML = (
    "<div class='info-notice'>"
    "使用设备上的语音识别在本地转写音频文件，支持最长约 20 分钟的音频。"
    "</div>"
)
ASR_AUDIO = "要转写的音频"
ASR_TRANSCRIBE = "开始转写"
ASR_OUTPUT = "转写文本"
ASR_SAVE_TXT = "保存为 .txt"
ASR_SAVE_PLACEHOLDER = "保存路径将在此显示……"
ASR_INFO_MD = (
    "**模型：** Qwen3-ASR-1.7B-8bit\n\n"
    "**支持语言：** 自动检测、英语、中文、日语、韩语、德语、法语、俄语、"
    "葡萄牙语、西班牙语、意大利语\n\n"
    "**最长时长：** 每个文件约 20 分钟"
)
ASR_NO_AUDIO_WARN = "请先上传或录制音频。"
ASR_NO_AUDIO = "没有音频"
ASR_NOTHING_TO_SAVE_WARN = "没有可保存的转写文本。"
ASR_NOTHING_TO_SAVE = "没有可保存的内容"
ASR_SAVED = "已保存：{path}"

# --- YT Voice Clone tab ---
TAB_YT = "YouTube 声音克隆"
YT_NOTICE_HTML = (
    "<div class='info-notice'>"
    "从 YouTube 视频中提取声音：获取视频、选择时间范围、自动提取带对齐文本的片段，然后生成。"
    "</div>"
)
YT_STEP_1 = "**步骤 1——视频 URL**"
YT_URL_PLACEHOLDER = "https://www.youtube.com/watch?v=……"
YT_FETCH = "获取视频信息"
YT_STEP_2 = "**步骤 2——选择片段**"
YT_START = "开始时间（分:秒）"
YT_START_PLACEHOLDER = "留空 = 从开头开始"
YT_END = "结束时间（分:秒）"
YT_END_PLACEHOLDER = "留空 = 视频结尾"
YT_EXTRACT = "提取片段"
YT_STEP_3 = "**步骤 3——检查参考文本**"
YT_TRANSCRIPT = "参考文本"
YT_TRANSCRIPT_INFO = "已从字幕自动填写——如有需要可以编辑"
YT_TRANSCRIPT_PLACEHOLDER = "提取后文本将在此显示，也可以手动输入……"
YT_TRANSCRIBE = "转写片段"
YT_TRANSCRIBE_HINT_HTML = "<div class='text-hint'>字幕不可用或不准确时，使用语音识别</div>"
YT_STEP_4 = "**步骤 4——生成并保存**"
YT_VOICE_NAME = "声音名称（保存到声音库）"
YT_VOICE_NAME_PLACEHOLDER = "YouTube声音"
YT_CLONE = "克隆并保存到声音库"
YT_THUMBNAIL = "视频缩略图"
YT_INFO_EMPTY = "_获取视频信息后将在此显示。_"
YT_CLIP_PREVIEW = "参考片段预览"
YT_STATUS = "状态"
YT_ENTER_URL = "请输入 YouTube URL"
YT_ENTER_URL_MD = "_请先在上方输入 URL。_"
YT_SUBS_MANUAL = "手动字幕"
YT_SUBS_AUTO = "自动生成字幕（可能需要编辑）"
YT_SUBS_NONE = "没有字幕——请手动输入文本"
YT_FETCHED = "✓ 已获取：{title}"
YT_FETCH_FIRST = "请先获取视频信息（步骤 1）"
YT_BAD_START = "开始时间格式错误：{err}"
YT_ENTER_END_WARN = "请输入结束时间——视频时长未知。"
YT_ENTER_END = "请输入结束时间"
YT_BAD_END_WARN = "结束时间无效——请使用 分:秒 格式"
YT_BAD_END = "请输入有效的结束时间"
YT_END_BEFORE_START_WARN = "结束时间必须晚于开始时间。"
YT_END_BEFORE_START = "结束时间必须晚于开始时间"
YT_CLIP_TOO_SHORT_WARN = "片段至少需要 3 秒。"
YT_CLIP_TOO_SHORT = "片段过短（至少 3 秒）"
YT_CLIP_CAPPED_WARN = "片段已限制为最长 60 秒。"
YT_CLIP_LONG_WARN = "片段为 {secs:.0f} 秒——5–20 秒通常能获得最佳克隆质量。"
YT_END_BEYOND_WARN = "结束时间超过视频时长（{secs:.0f} 秒）。"
YT_END_BEYOND = "结束时间超过视频结尾"
YT_EXTRACT_PROGRESS_START = "正在开始……"
YT_EXTRACT_PROGRESS_TRANSCRIPT = "正在提取文本……"
YT_EXTRACT_PROGRESS_DONE = "完成"
YT_EXTRACTED = "✓ 已提取 {secs:.1f} 秒片段——{note}"
YT_TRANSCRIPT_FILLED = "已自动填写文本"
YT_EXTRACT_FAILED = "提取失败：{err}"
YT_NO_CLIP_WARN = "请先提取片段（步骤 2）。"
YT_NO_CLIP = "没有可转写的片段"
YT_REQUIRED = "必填：{items}"
YT_REQ_TEXT = "要朗读的文本"
YT_REQ_CLIP = "参考片段（请先提取）"
YT_REQ_TRANSCRIPT = "参考文本"
YT_REQ_NAME = "声音名称"
YT_STOPPED = "已停止"
YT_SAVED_OK = "✓ {msg}"
YT_FETCH_ERROR = "获取视频信息失败：{err}"
YT_ERROR = "错误：{err}"
YT_GENERATION_FAILED = "生成失败：{err}"
YT_DURATION = "时长：{duration}｜{subtitle}"
YT_CHANNEL = "频道：{channel}"
YT_PROGRESS_CACHED = "使用缓存片段"
YT_PROGRESS_AUDIO = "正在下载音频片段……"
YT_PROGRESS_WAV = "正在转换为 WAV……"
YT_PROGRESS_SUBTITLES = "正在下载字幕……"
YT_GENERATED_STATUS = "生成用时 {secs:.1f} 秒｜{library}{denoise}{save}"

# --- Script Mode tab ---
TAB_SCRIPT = "脚本模式"
SM_NOTICE_HTML = (
    "<div class='info-notice'>"
    "<strong>多说话人脚本</strong>　—　每行格式：<code>SPEAKER: 对话文本</code>；"
    "没有标签的行将作为旁白。"
    "</div>"
)
SM_SCRIPT = "脚本"
SM_SCRIPT_PLACEHOLDER = (
    "旁白：很久以前，住着一位好奇的发明家。\n"
    "艾玛：爸爸，快看我在阁楼里发现了什么！\n"
    "父亲：那是我很久以前做的东西。"
)
SM_PARSE = "解析脚本"
SM_SILENCE = "行与行之间的静音（毫秒）"
SM_PARSE_RESULT = "解析结果"
SM_ASSIGNMENTS_HEADER = "### 声音分配"
SM_SLOT_MODE = "说话人 {n} 的声音类型"
SM_SLOT_INSTRUCT = "风格指令 / 声音描述"
SM_SLOT_INSTRUCT_PLACEHOLDER = "输入风格指令或声音描述"
SM_SLOT_LIBRARY = "声音库中的声音"
SM_GENERATE = "生成脚本"
SM_BREAKDOWN = "逐行结果"
SM_TABLE_EMPTY = "*生成后将在此显示结果。*"
SM_STATUS = "状态"
SM_TABLE_HEADERS = ["行号", "说话人", "文本", "状态"]
SM_NO_RESULTS = "*没有结果。*"
SM_ENTER_SCRIPT_WARN = "请先输入脚本。"
SM_ENTER_SCRIPT_MD = "*请先输入脚本。*"
SM_ENTER_SCRIPT = "请先输入脚本"
SM_PARSE_FIRST_WARN = "请先解析脚本并分配声音。"
SM_PARSE_FIRST_MD = "*请先解析脚本并分配声音。*"
SM_PARSE_FIRST = "请先解析脚本"
SM_PARSE_ERROR = "脚本解析错误：{err}"
SM_TOO_MANY_SPEAKERS = "说话人过多（{count}），最多允许 {max_count} 位。"
SM_NO_VALID_LINES = "脚本中没有有效行。"
SM_SUMMARY_HEAD = "找到 {speakers} 位说话人、{lines} 行："
SM_SUMMARY_LINE = "　{speaker}：{count} 行"
SM_MODE_LABELS = {
    "custom_voice": "自定义音色",
    "voice_design": "声音设计",
    "base": "声音克隆",
    "voice_clone": "声音克隆",
}
SM_UNKNOWN_MODE = "未知声音模式：{mode}"
SM_PROGRESS = "正在生成{label}行……"
SM_FAILED = "失败"
SM_STOPPED = "已停止"
SM_ALL_FAILED = "所有行均生成失败"
SM_GENERATED = "已生成 {done}/{total} 行"
SM_FAILURE_SUFFIX = "（{count} 行失败）"
SM_NOISE_SUFFIX = "｜已应用降噪"
SM_CLONE_PROGRESS = "正在生成声音克隆行……"
SM_HISTORY_PARAMS = "脚本模式"

# --- Streaming / cancel ---
STOP = "停止"
STOPPING = "正在停止……"
GENERATING_STATUS = "正在生成…… {secs:.1f} 秒"
STOPPED_KEPT = "已停止——保留了 {secs:.1f} 秒的部分音频"
TIMED_OUT_KEPT = "已在 {timeout} 秒后超时——保留了 {secs:.1f} 秒的部分音频"
TIMEOUT_MSG = "生成超时——请降低最大长度或缩短文本"
BATCH_SEGMENT_PROGRESS = "片段 {done}/{total} · 音频 {secs:.1f} 秒"
BATCH_STOPPED = "已停止——完成 {done}/{total} 个片段"
SCRIPT_LINE_PROGRESS = "行 {done}/{total} · 音频 {secs:.1f} 秒"
SCRIPT_STOPPED = "已停止——完成 {done}/{total} 行"
ASR_LOADING = "正在加载语音识别模型……"
TRANSCRIBING = "正在转写……{words} 个词"
TRANSCRIBE_STOPPED = "已停止——保留部分转写文本"
TIMEOUT_SLIDER_INFO = "运行达到此秒数后自动停止，并保留部分音频"
GENERATION_FAILED = "生成失败：{err}"
ERROR = "错误：{err}"
NO_AUDIO_PRODUCED = "没有生成音频"
NO_TEXT_SEGMENTS_WARN = "没有找到文本片段。"
NO_SEGMENTS = "没有片段"
TOO_MANY_SEGMENTS_WARN = "片段过多（{count}）。最多允许 {max_count} 个。"
TOO_MANY_SEGMENTS = "片段过多（最多 {max_count} 个）"
SEGMENT_EMPTY = "（空）"
SEGMENT_ERROR = "（错误）"
SEGMENT_STOPPED = "已停止"
SEGMENT_FAILED = "失败：{err}"
ALL_SEGMENTS_FAILED = "所有片段均生成失败"
GENERATED_SEGMENTS = "已生成 {done}/{total} 个片段"
FAILED_SEGMENTS_SUFFIX = "（{count} 个失败）"
NOISE_REDUCTION_SUFFIX = "｜已应用降噪"
GENERATED_STATUS = "生成用时 {secs:.1f} 秒｜模型：{repo}{extras}{save}"
BATCH_GENERATED_STATUS = "已生成 {done}/{total} 个片段"
BATCH_FAILED_SUFFIX = "（{count} 个失败）"
LOADING_MODEL = "正在将模型加载到内存……（{repo}）"
DOWNLOADING_MODEL = "首次运行正在下载模型（约 6 GB），可能需要几分钟……（{repo}）"
TRANSCRIPTION_FAILED = "转写失败：{err}"
TRANSCRIPTION_EMPTY = "转写结果为空"
TRANSCRIBED = "已转写（{words} 个词）"
NO_AUDIO_TO_SAVE_WARN = "没有可保存的音频。"
NO_AUDIO_TO_SAVE = "没有可保存的音频"
FFMPEG_WARN = "未找到 ffmpeg——已改为保存 WAV，而不是 {format}。"
SAVED = "已保存：{path}"
TEXT_REQUIRED_WARN = "请输入要朗读的文本。"
TEXT_REQUIRED = "请先输入文本"
VOICE_NOT_FOUND_WARN = "声音“{name}”不在声音库中。"
VOICE_NOT_FOUND = "未找到声音"
REF_AUDIO_REQUIRED_WARN = "请上传参考音频，或从声音库中选择。"
NO_REF_AUDIO = "没有参考音频"
REF_TEXT_REQUIRED_WARN = "声音克隆需要参考文本。"
NO_REF_TEXT = "没有参考文本"
VOICE_DESCRIPTION_REQUIRED_WARN = "请描述想要的声音。"
VOICE_DESCRIPTION_REQUIRED = "请先描述声音"

# --- History tab ---
TAB_HISTORY = "历史记录"
HIST_TABLE_LABEL = "生成历史"
HIST_TABLE_HEADERS = ["编号", "时间", "模式", "文本", "时长"]
HIST_ENTRY_ID = "记录编号"
HIST_ENTRY_PLACEHOLDER = "粘贴记录编号以预览或管理"
HIST_PREVIEW = "预览"
HIST_DELETE = "删除记录"
HIST_CLEAR = "清空全部历史"
HIST_VIEW_SETTINGS = "查看生成设置"
HIST_SETTINGS_LABEL = "生成设置"
HIST_STATUS_PLACEHOLDER = "状态……"
HIST_SELECT_FIRST = "请先选择一条记录"
HIST_DELETED = "已删除记录 {entry_id}"
HIST_CLEARED = "历史记录已清空"
HIST_NOT_FOUND = "未找到记录"
HIST_AUDIO_NOT_FOUND = "未找到音频"
HIST_AUDIO_MISSING_WARN = "此记录对应的音频不存在。"
HIST_MODE = "模式：{mode}"
HIST_LANGUAGE = "语言：{language}"
HIST_VOICE = "声音：{speaker}"
HIST_PARAMS = "参数：{params}"
HIST_VOICE_DESCRIPTION = "声音描述：{description}"
HIST_STYLE_INSTRUCTION = "风格指令：{style}"
HIST_SEED = "随机种子：{seed}（{mode}）"
HIST_SAMPLING = (
    "采样设置：temperature={temperature}，top-k={top_k}，top-p={top_p}，"
    "重复惩罚={repetition_penalty}，最大 token={max_tokens}"
)
HIST_TEXT = "文本：{text}"

# --- Voice Library tab ---
TAB_LIBRARY = "声音库"
LIB_TABLE_LABEL = "已保存的声音"
LIB_TABLE_HEADERS = ["名称", "来源", "语言", "描述"]
LIB_SELECTED = "声音名称"
LIB_SELECTED_PLACEHOLDER = "输入或粘贴声音名称"
LIB_PREVIEW = "预览"
LIB_DELETE = "删除"
LIB_RENAME_TO = "重命名为"
LIB_RENAME_PLACEHOLDER = "新名称"
LIB_RENAME = "重命名"
LIB_PREVIEW_AUDIO = "参考音频预览"
LIB_STATUS_TEXT_PLACEHOLDER = "状态……"
LIB_IMPORT_HEADER = "### 导入声音"
LIB_IMPORT_TRANSCRIPT = "参考文本"
LIB_IMPORT_NAME = "声音名称"
LIB_IMPORT_NAME_PLACEHOLDER = "导入的声音"
LIB_IMPORT = "导入声音"
LIB_SELECT_FIRST = "请先选择一个声音"
LIB_DELETED = "已删除“{name}”"
LIB_VOICE_NOT_FOUND = "未找到声音“{name}”"
LIB_ENTER_NEW_NAME = "请输入新名称"
LIB_RENAMED = "已将“{old}”重命名为“{new}”"
LIB_RENAME_FAILED = "重命名失败（名称可能已存在）"
LIB_IMPORTED = "已导入“{name}”"
LIB_IMPORT_NO_AUDIO_WARN = "请上传音频后再导入。"
LIB_IMPORT_NO_AUDIO = "请先上传音频"
LIB_IMPORT_NO_NAME_WARN = "请输入导入声音的名称。"
LIB_IMPORT_NO_NAME = "请输入名称"
LIB_IMPORT_NO_TRANSCRIPT_WARN = "导入声音需要参考文本。"
LIB_IMPORT_NO_TRANSCRIPT = "请输入参考文本"
LIB_IMPORTED_DESCRIPTION = "导入的声音"

# --- Settings tab ---
TAB_SETTINGS = "设置"
SET_MODEL_HEADER = "### 模型"
SET_MODEL_SIZE = "模型大小"
SET_QUANT = "量化方式"
SET_QUANT_INFO = "更小的量化占用更少内存，但质量略低"
SET_LOADED_MODEL = "已加载模型"
SET_UNLOAD = "卸载模型 / 释放内存"
SET_REF_HEADER = "### 参考音频"
SET_DENOISE = "降低参考音频背景噪声"
SET_DENOISE_INFO = "声音克隆参考音频使用 DeepFilterNet 模型（8 MB，首次使用时下载）"
SET_LANGUAGE_HEADER = "### 语言"
SET_DEFAULT_LANGUAGE = "默认语言"
SET_JIT = "加速重复运行"
SET_JIT_INFO = "首次生成后编译模型；更改此项会重新加载模型"
SET_CACHE_ACCORDION = "模型缓存与语音识别"
SET_CACHE_HEADER = "### 模型缓存"
SET_CACHE_DIR = "模型下载目录"
SET_DELETE_MODELS = "删除已下载模型"
SET_DELETE_PLACEHOLDER = "下次使用时会重新下载模型。"
SET_ASR_HEADER = "### 语音识别"
SET_ASR_STATUS = "语音识别模型"
SET_ASR_NOT_LOADED = "未加载（按需加载）"
SET_ASR_UNLOAD = "卸载语音识别模型"
SET_GENERATION_HEADER = "### 生成"
SET_PRESET = "生成预设"
SET_PRESET_INFO = "预设会填入下方滑块，之后仍可自由调整。"
SET_TEMP = "温度"
SET_TEMP_INFO = "越高表达越多样，越低越稳定"
SET_TOP_K = "Top-K"
SET_TOP_K_INFO = "每一步考虑的候选音频数量——越低越稳妥"
SET_TOP_P = "Top-P"
SET_TOP_P_INFO = "只保留概率最高的候选音频——越低越可预测"
SET_REP_PENALTY = "重复惩罚"
SET_REP_PENALTY_INFO = "抑制重复声音；声音克隆始终不低于 1.5"
SET_MAX_TOKENS = "最大长度（tokens）"
SET_MAX_TOKENS_INFO = "单次生成的长度上限；运行过久时可以调低"
SET_TIMEOUT = "自动停止时间（秒）"
SET_BATCH_SIZE = "批量大小"
SET_BATCH_SIZE_INFO = "并行生成的片段数（批量模式和脚本模式）"
SET_RESET = "恢复默认值"
SET_OUTPUT_HEADER = "### 输出"
SET_OUTPUT_DIR = "输出目录"
SET_AUTOSAVE = "自动保存生成音频"
SET_EXPORT_HEADER = "### 导出格式"
SET_EXPORT_FORMAT = "音频格式"
SET_MP3_BITRATE = "MP3 比特率（kbps）"
SET_POST_HEADER = "### 后处理"
SET_LOUDNORM = "标准化响度"
SET_LOUDNORM_INFO = "EBU R128 广播标准"
SET_TRIM_SILENCE = "裁剪开头和结尾的静音"
SET_STORAGE_ACCORDION = "存储与缓存"
SET_YT_CACHE_HEADER = "### YouTube 缓存"
SET_YT_CACHE_CLEAR = "清除 YouTube 缓存"
SET_YT_CACHE_PLACEHOLDER = "缓存：{cache_dir}/"
SET_STORAGE_HEADER = "### 存储路径"
SET_STORAGE_LIBRARY = "声音库"
SET_STORAGE_HISTORY = "历史记录"
SET_APPLY = "应用设置"
SET_NO_MODEL = "没有加载模型"
SET_MODEL_LOADED = "已加载：{repo}"
SET_APPLIED = "设置已应用——{details}。"
SET_APPLIED_SIZE_QUANT = "大小：{size}，量化：{quant}"
SET_APPLIED_UNLOADED = "模型已卸载"
SET_UNLOADED_MSG = "模型已卸载，内存已释放。"
SET_ASR_UNLOADED_MSG = "语音识别模型已卸载，内存已释放。"
SET_YT_CACHE_CLEARED = "YouTube 缓存已清除——删除了 {n} 个条目"
SET_CACHE_DIR_MISSING = "未找到模型下载目录"
SET_DELETED_MODELS = "已删除 {n} 个模型：{names}"
SET_DELETE_FAILED = "删除 {n} 个模型失败：{details}"
SET_NO_MODELS_FOUND = "缓存中没有找到 Qwen3-TTS 模型"
SETTINGS_LOAD_FAILED = "本地设置文件无法读取，已使用安全默认值：{err}"
SETTINGS_INVALID_VALUE = "本地设置“{key}”无效，已回退为安全默认值。"
SETTINGS_UNSUPPORTED_VERSION = "本地设置版本不受支持，已读取可识别的设置并使用安全默认值补全。"
SETTINGS_SAVE_FAILED = "设置已应用，但无法保存到本地设置文件：{err}"
SETTINGS_SANITIZED = "部分无效设置已回退为安全默认值"

# --- Startup checks and table empty states ---
STARTUP_PYTHON_REQUIRED = "需要 Python 3.10 或更高版本"
STARTUP_MLX_AUDIO_MISSING = "未安装 mlx-audio——请运行：pip install mlx-audio"
STARTUP_FFMPEG_MISSING = "未找到 ffmpeg——请运行：brew install ffmpeg"
STARTUP_YT_DLP_MISSING = "未找到 yt-dlp——请运行：pip install yt-dlp"
STARTUP_PYSRT_MISSING = "未安装 pysrt（字幕功能不可用）——请运行：pip install pysrt"
STARTUP_YT_ERROR = "YouTube 声音克隆初始化错误：{err}"
NO_VOICES = "*尚未保存声音。*"
NO_HISTORY = "*尚无历史记录。*"
NO_ENTRIES = "*暂无条目。*"
