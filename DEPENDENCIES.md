# Historical development dependencies

This source archive does not bundle .venv, CUDA, models, or NuGet packages. The versions below document the original development environment; install only the dependencies required for your chosen backend.


The bundled `.venv` uses CPython 3.14 and was validated with:

- Torch `2.13.0+cu126`, CUDA runtime `12.6`
- ONNX Runtime GPU `1.26.0` (CUDA 12.x build)
- Qwen ASR `0.0.6`
- Transformers `4.57.6`
- NumPy `2.4.6`
- SoundFile `0.14.0`
- gRPC / gRPC tools `1.83.0`
- Protobuf `7.35.1`
- pytest `9.1.1`
- PyCantonese `5.0.0` (MIT; wheel SHA-256
  `8f7bdc1e9a37b5e164cd57f22aecae8532d4d0b74ff5b6f5e28480a657fa53f9`)
- Rustling `0.9.0` (PyCantonese runtime; wheel SHA-256
  `e78ce1bc7b9167d251d53541ffe8c978836bff2c0a6af992974f039eb2ad5c30`)

The GPU smoke check used an NVIDIA GeForce RTX 4090 Laptop GPU. PyTorch completed
a real CUDA tensor operation, and the bundled ONNX/eSpeak model completed real
inference with `CUDAExecutionProvider` active. The ONNX adapter preloads the
CUDA 12/cuDNN 9 DLLs bundled by the locked Torch build; the separate NVIDIA
runtime extras are intentionally not installed. Use
`.venv\Scripts\python.exe -m pip check` to audit the environment after changes.

PyCantonese is used only behind the replaceable Cantonese G2P adapter. Its
package/data license was reviewed as non-NC, its exact version is locked, and
runtime code never downloads a G2P model. External G2P adapters must provide an
explicit license identifier, local model path, revision, and SHA-256.
Install the optional package boundary with `pip install -e .[cantonese-g2p]`
when timeline items do not provide explicit Jyutping; the fixture/reviewed
lexicon path remains available without this optional dependency.

NuGet packages are stored below `.nuget/packages`; `NuGet.Config` keeps the
upstream source explicit. Build and temporary output should be redirected to
`.state` so the source tree remains clean.
