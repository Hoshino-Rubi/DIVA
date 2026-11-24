"""Offline stem/phone inference in a separately selected local Python runtime."""
from __future__ import annotations

import json
from math import gcd
from pathlib import Path
import subprocess
import sys


def invoke(python: Path, action: str, audio: Path, bundle: Path, output: Path,
           note_windows: Path | None = None) -> None:
    worker = Path(__file__).resolve()
    subprocess.run([str(python), "-X", "utf8", str(worker), action, str(audio.resolve()),
                    str(bundle.resolve()), str(output.resolve()),
                    *([str(note_windows.resolve())] if note_windows else [])], check=True)


def run(action: str, audio: Path, bundle: Path, output: Path,
           note_windows: Path | None = None) -> None:
    import numpy as np
    import soundfile as sf
    values, rate = sf.read(audio, dtype="float32", always_2d=True)
    if action == "separate":
        import torch
        from scipy.signal import resample_poly
        from any2xsampa.adapters.cantonese4_runtime import separate
        if rate != 44100:
            g = gcd(rate,44100)
            values = resample_poly(values,44100//g,rate//g,axis=0)
        stereo = np.repeat(values,2,axis=1) if values.shape[1]==1 else values
        stem = separate(torch,bundle,np.ascontiguousarray(stereo.T),lambda: None)
        sf.write(output,stem.mean(axis=0),44100,subtype="PCM_16")
    else:
        raise ValueError("Unknown Cantonese acoustic worker action")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    run(sys.argv[1], *(Path(arg) for arg in sys.argv[2:]))
