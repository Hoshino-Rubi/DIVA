# Third-party notices

## PhoneticXeus
Author/model: changelinglab/PhoneticXeus.
Source: https://huggingface.co/changelinglab/PhoneticXeus/tree/8d83dee94817a07dc150f87d08f7e0ee01bdb66d
Revision: 8d83dee94817a07dc150f87d08f7e0ee01bdb66d.
License metadata: Apache-2.0. See licenses/Apache-2.0.txt and the unchanged model README.
Upstream code/metadata are bundled; model.safetensors is downloaded unchanged, or reassembled byte-for-byte from offline parts. Changes: local packaging and path-only runtime integration; no new model training.

## GAME
Author/project: OpenVPI / GAME; pretrained GAME-1.0 medium model, ONNX v1.0.3.
Source release: https://github.com/openvpi/GAME/releases/tag/v1.0.3
Original model release and weight license: https://github.com/openvpi/GAME/releases/tag/v1.0.0
Code license: MIT (licenses/GAME-code-MIT.txt).
Pretrained model license: CC BY-NC-SA 4.0 (licenses/CC-BY-NC-SA-4.0.txt).
The original release acknowledges label contributors 挚彬Club, YQ之神, 小狼, 幽寂.
The ONNX archive is distributed unchanged. Model files are extracted verbatim and verified; local snapshot metadata and this application's inference wrapper are separate.
No endorsement by the model authors is implied. Preserve attribution, noncommercial and share-alike requirements for the covered material.

## Runtimes and dependencies
CPython 3.14.0 and 3.12.10 license texts are in runtime/core/LICENSE.txt and runtime/phonetic/LICENSE.txt.
Pinned package inventories are in runtime/*/DEPENDENCIES.json.
Original package license/notice files remain alongside dependencies under runtime/*/Lib/site-packages, including PyTorch, torchaudio, ONNX Runtime, NumPy, SciPy, librosa, CUDA and cuDNN components.
Only development static libraries, tests, caches and debug symbols were excluded from the runtime.
Microsoft .NET and Windows App SDK redistribution files are retained in this application directory.
This notice does not assign an overall open-source license to the application's own code; the repository owner has not selected one yet.
