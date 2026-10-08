"""Capture existing ONNX adapter frame evidence, including CTC blank frames."""
from __future__ import annotations
from any2xsampa.adapters.onnx_espeak import OnnxEspeakCTCUniversalPhoneRecognizer

class RecordedOnnxRecognizer(OnnxEspeakCTCUniversalPhoneRecognizer):
    def _predict_frames(self, *args, **kwargs):
        predictions = super()._predict_frames(*args, **kwargs)
        self.last_frames = predictions
        return predictions
