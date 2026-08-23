# Kurisu Master Timbre A

## Owner-confirmed local master

The owner has confirmed that this is the previously remembered preferred Kurisu
voice candidate, rated approximately 95/100. It is the authoritative current
local master timbre reference for future voice work.

The audio is intentionally kept outside the public repository:

- Local archive: `/Users/mizukinamachi/Qwen3-TTS/master_voice/Kurisu_Master_Timbre_A.wav`
- Original source: `/Users/mizukinamachi/Qwen3-TTS/output/design_20260823_173304-955911998-m.wav`
- SHA-256: `78a38fad92ec24babc43235f55f0ccffd70bcaa64216b787e4474ede901729eb`
- Matched History entry: `3701ed19fa69`
- Seed: `955911998`
- History timestamp: `2026-08-23T17:33:04`

The selection is supported by the owner-added seed and `m` filename markers,
the matching generation timestamp, and an exact byte-for-byte SHA-256 match to
the retained History WAV.

## Important limitations

The actual WAV and its local JSON provenance record are **not committed to this
public repository**. The local provenance record is:

`/Users/mizukinamachi/Qwen3-TTS/master_voice/Kurisu_Master_Timbre_A.json`

The recorded generation used the Qwen3-TTS VoiceDesign 1.7B BF16 model path,
with `seed_mode=random` and language `Auto-detect` in the retained metadata.
The seed is generation provenance, not a stable cross-text voice ID: the same
seed does not guarantee the same speaker identity when text, description,
style, language, or sampling conditions change.

Future voice-cloning or reference-audio work should use the archived WAV above
as the authoritative local reference. Do not rely on the original generated
filename remaining in place.

## Production Clone reference

The owner has separately confirmed the 7.45-second MEDIUM excerpt as the production
conditioning reference for Base Voice Clone:

- Local-only archive: `/Users/mizukinamachi/Qwen3-TTS/master_voice/Kurisu_Production_Clone_Reference_A.wav`
- SHA-256: `b3c4ea03803b3b7226d85c8ddc288e47caab4113b63513e5218abe722f5dbfbe`
- Exact reference text: `等一下，你这个结论是怎么得出来的？……不，我不是说一定有问题，只是这里少了一个必要条件。`
- Local Voice Library profile: `Kurisu_Production_Clone_A`

The MEDIUM reference was selected because it had the strongest same-woman and
long-term-listening result in the length comparison. Five owner-run non-streaming
checks scored 85/86/88/85/87 for Master similarity, with 0/5 severe startup
artifacts. Normal single Clone therefore uses the quality-first non-streaming
engine API. This improves startup reliability but removes responsive mid-generation
Stop/timeout for that single path. Streaming Clone remains available internally for
batch fallback and future low-latency work; no generic stream/non-stream toggle is
exposed.

This conditioning reference does **not** supersede `Kurisu_Master_Timbre_A.wav`
as the authoritative timbre Master. The audio and Voice Library profile remain
local-only and are not committed to the public repository.
