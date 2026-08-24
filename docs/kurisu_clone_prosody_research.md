# Kurisu Clone Prosody Research Checkpoint

Status: checkpointed; further subjective prosody work deferred.

This document records the current research boundary for the local Kurisu Base
Voice Clone. It does not change production inference behavior and does not
publish any local audio.

Related documentation:

- [Kurisu Master Voice](kurisu_master_voice.md)
- [Voice Clone generation and streaming stability](voice_clone_streaming.md)

## Locked production baseline

These decisions remain locked for the current Studio baseline:

- Authoritative timbre Master: local-only `Kurisu_Master_Timbre_A.wav`.
- Production Clone reference: local-only `Kurisu_Production_Clone_Reference_A.wav`.
- Production reference SHA-256: `b3c4ea03803b3b7226d85c8ddc288e47caab4113b63513e5218abe722f5dbfbe`.
- Production reference duration: 7.45-second MEDIUM excerpt.
- Normal single Voice Clone: quality-first non-streaming generation.
- Repeated owner non-stream similarity: `85 / 86 / 88 / 85 / 87`.
- Severe startup artifact in that validated non-stream check: `0/5`.

The Production Clone reference is a conditioning reference. It does not
supersede the authoritative timbre Master. The local audio and Voice Library
profile remain outside the public repository.

## Validated prosody findings

The broad punctuation experiment established that target-text punctuation and
structure can materially affect emotional interpretation:

- D2 produced more emotion than D0.
- B1 produced more emotion than B0.
- Stronger punctuation can also change pacing and perceived speaker quality.
- Real line breaks did not reliably improve pause naturalness.
- Maximum emotional intensity is not the product goal.

The desired production character remains restrained, intelligent, natural, and
emotionally alive: not theatrical, aggressive, overly cute, or overly fast.

The emotional Ground Truth excerpt remains a listening reference only. Its
pause naturalness was not considered an ideal target, so it is not used as a
production Clone reference.

## Minimal punctuation ablation

The later ablation isolated the three changes in the composite B1 text:

- P0: unchanged baseline.
- P1: add only the ellipsis after the first sentence.
- P2: change only `一件事，` to `一件事：`.
- P3: change only `算了，` to `算了。`.

Owner observations with high confidence at the qualitative level:

- P1 did not make the voice more aggressive. It may slightly soften or add
  emotional nuance, but can increase pacing somewhat.
- P2's most noticeable effect appeared to involve breath magnitude or duration.
  There is not sufficiently reliable evidence that it consistently adds an
  intellectual or logical tone.
- P3 can remove the connected breath between clauses. The transition was
  perceived as more emotionless and is not suitable for promotion for the
  current Kurisu target.
- Across P0/P1/P2/P3, speaker identity remained broadly stable, but no isolated
  rule produced a sufficiently large and reliable improvement to justify
  productionization.

Decision:

`MINIMAL_PUNCTUATION_PRODUCTION_DECISION = INCONCLUSIVE_NO_RULE_PROMOTED`

No punctuation formatter is implemented. In particular, P3 is not promoted;
P1 and P2 remain research observations rather than production rules.

## Perceptual fatigue stop condition

During repeated comparison of near-identical outputs, the owner reported:

- degraded attention;
- increasing difficulty distinguishing small differences;
- decreasing reliability of subjective numeric scores;
- only explicitly localized punctuation effects remaining clearly distinguishable.

Therefore, late-stage fine-grained scores are not authoritative evidence. Small
differences such as 0.2–0.5 subjective points must not be used to rank P0/P1/P2/P3.
This is an experiment stop condition, not a model or runtime failure.

`FURTHER_SUBJECTIVE_PROSODY_TESTING = DEFERRED_OWNER_PERCEPTUAL_FATIGUE`

## Current production policy

Production remains:

```text
LLM output text
  -> no automatic punctuation/prosody rewriting
  -> Kurisu_Production_Clone_Reference_A
  -> Qwen3-TTS Base BF16
  -> non-streaming single Voice Clone
```

No automatic punctuation formatter is active. User text is not rewritten.
Sampling, repetition penalty, reference audio, and the production Clone path
are unchanged.

## Deferred research backlog

The following are recorded for a later, explicitly authorized research cycle;
none were executed here:

1. Reference prosody transfer.
2. Effective MLX ICL repetition penalty behavior: requested `1.05`, local ICL
   effective clamp `>= 1.5`.
3. Sampling and prosody behavior.
4. MLX versus official Qwen Base Clone behavior.
5. Potential alternate emotional references.
6. A less fatiguing listening methodology: at most 2–3 clips per decision
   round, simple forced-choice A/B where possible, and no multidimensional
   fine-grained scoring unless needed for diagnosis.

Further subjective prosody testing should resume only after the owner is ready
and should begin with a small, low-fatigue comparison rather than another large
near-duplicate matrix.

## Scope and safety for this checkpoint

- No model was loaded or downloaded.
- No inference or audio generation was performed.
- No production source, settings, History, Voice Library, reference audio, or
  model cache was modified.
- Local audio and experiment data remain local-only and untracked.
