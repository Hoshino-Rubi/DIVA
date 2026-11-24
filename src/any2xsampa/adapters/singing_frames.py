"""Frozen Cantonese Whisper and eSpeak-CTC features on a shared 20 ms clock."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def audio16(path: Path) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    g = gcd(rate, 16000)
    return resample_poly(mono, 16000 // g, rate // g) if rate != 16000 else mono


def phrase_windows(audio: np.ndarray):
    """Select a low-energy transition in the 15–17 second region, with context across joins."""
    hop = 320
    total = (len(audio) + hop - 1) // hop
    blocks = np.pad(audio, (0, total * hop - len(audio))).reshape(-1, hop)
    power = np.mean(blocks ** 2, axis=1)
    activity = np.convolve(power, np.ones(9) / 9, mode="same")[:total]
    left = 0
    while left < total:
        right = total
        if total - left > 875:
            interval = activity[left+750:left+851]
            # Prefer a pause nearest the end when silence spans many frames.
            quiet = np.flatnonzero(interval <= np.quantile(interval, .05))
            right = left + 750 + int(quiet[-1])
        yield left, right, max(0, left-12), min(total, right+12)
        left = right


class SingingEncoders:
    def __init__(self, bundle: Path, ctc: Path, device: str = "auto"):
        import torch
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
        from .onnx_espeak import _resolve_model_file

        root = str((bundle / "small").resolve(strict=True))
        self.extractor = WhisperFeatureExtractor.from_pretrained(root, local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        torch.set_num_threads(4)
        model = WhisperForConditionalGeneration.from_pretrained(root, local_files_only=True,
                                                               dtype=self.dtype, weights_only=True)
        self.whisper = model.model.encoder.eval().requires_grad_(False).to(self.device)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        from .ort_device import create_session
        self.ctc = create_session(_resolve_model_file(ctc), options, device)

    def encode(self, audio: np.ndarray, *, top_inputs: list | None = None) -> np.ndarray:
        import torch
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Expected finite nonempty mono audio")
        norm = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        chunks = []
        for left, right, start, end in phrase_windows(audio):
            n = end - start
            x = norm[start*320:end*320]
            mel = self.extractor(x, sampling_rate=16000, return_tensors="pt").input_features
            with torch.inference_mode():
                states = self.whisper(mel.to(self.device, self.dtype), output_hidden_states=True).hidden_states
                if top_inputs is not None:
                    # Input to the last two actual pretrained encoder blocks.
                    top_inputs.append(((left,right,start,end),states[-3][0,:n].cpu().half().numpy().copy()))
                # Intermediate blocks expose localized phone evidence before decoder conditioning.
                h = ((states[5][0, :n].float() + states[8][0, :n].float()) / 2).cpu().numpy()
            # Wav2vec receptive field 400 samples, stride 320. Align frame centres at 10 ms.
            wave = np.pad(x, (40, n*320+40-len(x)))[None, :].astype(np.float32)
            inputs = {self.ctc.get_inputs()[0].name: wave}
            if any(i.name == "attention_mask" for i in self.ctc.get_inputs()):
                inputs["attention_mask"] = np.ones(wave.shape, dtype=np.int64)
            logits = self.ctc.run(None, inputs)[0][0]
            if len(logits) != n:
                raise ValueError(f"Unexpected eSpeak sample clock: {len(logits)} vs {n}")
            features = np.concatenate([h, logits, mel[0, :, :2*n:2].numpy().T], axis=1)
            if not np.isfinite(features).all():
                raise ValueError("Nonfinite phrase features")
            chunks.append(features[left-start:right-start].astype(np.float16))
        return np.concatenate(chunks)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("ctc", type=Path)
    p.add_argument("stems", nargs="+", type=Path)
    args = p.parse_args()
    encoders = SingingEncoders(args.bundle, args.ctc)
    for stem in args.stems:
        data = encoders.encode(audio16(stem))
        np.save(stem.with_suffix(".npy"), data)
        print(stem.name, data.shape, flush=True)
