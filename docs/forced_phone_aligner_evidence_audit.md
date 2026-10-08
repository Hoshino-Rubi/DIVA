# Forced-Phone Aligner Evidence Audit

Audit date: 2026-08-16

This document records why currently known public candidates do not satisfy the
qualified-model route in `REDLINES.md`. It is a rejection log, not a model
manifest. A candidate remains unqualified unless primary sources establish all
of the following for the exact weights used by the product:

- an explicit license without a NonCommercial restriction;
- an immutable model revision and verified SHA-256 digests;
- a declared phone inventory and all required phone/dictionary resources; and
- training or fine-tuning on Cantonese singing, not merely Cantonese speech,
  Mandarin singing, multilingual support, or singing-capable inference.

## Candidate Decisions

| Candidate | Relevant primary-source evidence | Decision |
| --- | --- | --- |
| Qwen3-ForcedAligner-0.6B | The official model card lists Cantonese support, but classifies the forced aligner audio type as `Speech`. Singing benchmarks in the same card are for the ASR models, not evidence that the forced aligner was trained or fine-tuned on Cantonese singing. | Reject: no Cantonese-singing training evidence for the aligner weights. |
| HubertFA v0.0.7 | The signed release identifies the exact asset `1218_hfa_model_new_dict.zip` and its scope as Mandarin, Japanese, and English singing. The bundled `vocab.json` has no Yue/Cantonese vocabulary or dictionary, and the release does not ship a primary training-data/license record for the weights. | Reject: no Cantonese-singing checkpoint or weight-rights evidence. |
| SOFA-Cantonese-Model v0.1 | The official README says only that one hour of training data came from one provider. It does not identify that data as singing. The repository MIT license does not explicitly identify the separately published release ZIP weights or their training data. | Reject: training domain and weight-license coverage are not established. |
| STARS Chinese/bilingual | The official repository describes a singing alignment model, but its Chinese example and phone set are Mandarin. Its documented GTSinger training pipeline uses a dataset licensed CC BY-NC-SA 4.0. | Reject: not Cantonese training and the documented singing dataset is NC. |
| DeepSinger alignment model | The paper reports a mined singing corpus containing Cantonese and an internal phoneme-alignment model. | Reject: no public immutable aligner checkpoint, exact phone inventory, or explicit checkpoint license was found. |
| Akatsuki Cantonese Singing dataset | The dataset card identifies Cantonese singing, but access requires manual approval and commercial use requires separate written consent. It is a dataset, not a released forced-aligner checkpoint. | Reject: no usable aligner weights and no unconditional non-NC model evidence. |

The complete HubertFA v0.0.7 asset was fetched through the configured local
proxy for this audit: `1218_hfa_model_new_dict.zip` is 256,589,553 bytes with
SHA-256 `48bd6dbcc293e47cc6cbcc556baf575e3d91d61d0ac88ddeccd6098f8f346fa2`.
Its `model.onnx` is accompanied only by Mandarin, English, and Japanese
dictionaries/vocabulary; a valid archive and hash do not cure the missing
Cantonese-training and weights-license evidence.

## Primary Sources

- Qwen3-ForcedAligner-0.6B model card:
  <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
- HubertFA repository and v0.0.7 release:
  <https://github.com/wolfgitpr/HubertFA>
  <https://github.com/wolfgitpr/HubertFA/releases/tag/v0.0.7>
- SOFA-Cantonese-Model repository and v0.1 release:
  <https://github.com/RibosomeK/SOFA-Cantonese-Model>
  <https://github.com/RibosomeK/SOFA-Cantonese-Model/releases/tag/v0.1>
- STARS repository and paper:
  <https://github.com/gwx314/STARS>
  <https://arxiv.org/abs/2507.06670>
- GTSinger dataset license:
  <https://huggingface.co/datasets/AaronZ345/GTSinger/blob/main/dataset_license.md>
- DeepSinger paper:
  <https://arxiv.org/abs/2007.04590>
- Akatsuki Cantonese Singing dataset card:
  <https://huggingface.co/datasets/Akatsuki-Amemiya/Akatsuki_Cantonese_Singing>

## Current Product Consequence

None of these candidates may satisfy the qualified-model route. The current
product must therefore use the human-reviewed alignment route unless a future
candidate supplies stronger primary evidence for the exact weights.
