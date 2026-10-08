"""Offline CPU GAME worker shared by desktop and CLI; audio-only model inputs."""
from __future__ import annotations

from pathlib import Path
import sys
import time

from .common import check_cancel, read, sha, write
from .server import Server, input_lines


class GameEngine:
    def __init__(self) -> None:
        self.runtime = None
        self.snapshot_key = None

    def close(self) -> None:
        self.runtime = None
        self.snapshot_key = None

    def execute(self, action: str, params: dict, cancel, progress) -> dict:
        allowed = {"audio_path", "audio_sha256", "start_sec", "end_sec", "snapshot", "output_dir"}
        if action != "transcribe_notes" or set(params) != allowed:
            raise ValueError("GAME worker accepts only source audio, interval, runtime and output")
        from any2xsampa.adapters.game_onnx import GameOnnxRuntime, checked_request, load_audio
        from any2xsampa.music.game_notes import POLICY_PATH, decode_regions
        import numpy as np

        request = checked_request({k: params[k] for k in ("audio_path", "audio_sha256", "start_sec", "end_sec")})
        output = Path(params["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        attempted = False
        initial_calls = 0
        write(output / "START.json", {"audio": request, "maximum_encoder_calls": 1,
                                     "maximum_graph_calls": 11, "device": "cpu", "retried": False})
        try:
            check_cancel(cancel)
            waveform, info = load_audio(request, output / "input-crop.wav")
            snapshot = Path(params["snapshot"]).resolve()
            key = (str(snapshot), sha(snapshot / "snapshot-lock.json"))
            if self.runtime is None or self.snapshot_key != key:
                self.close()
                progress({"stage": "loading_note_model", "device": "cpu"})
                self.runtime = GameOnnxRuntime(snapshot, device="cpu")
                self.snapshot_key = key
            initial_calls = len(self.runtime.calls)
            check_cancel(cancel)
            write(output / "FORWARD_STARTED.json", {"encoder_requests": 1, "maximum_graph_calls": 11})
            attempted = True
            raw = self.runtime.infer(waveform)
            calls = self.runtime.calls[initial_calls:]
            if len(calls) != 11:
                raise ValueError("Unexpected GAME graph call count")
            check_cancel(cancel)
            rawfile = output / "raw-notes.npz"
            with rawfile.open("xb") as handle:
                np.savez_compressed(handle, **raw)
            value = decode_regions(raw, start_sec=info["effective_start_sec"],
                                   duration_sec=info["model_duration_sec"])
            value.update(input=info, mode="real", model_id="openvpi/GAME-1.0.3-medium-onnx",
                         snapshot_lock_sha256=key[1], policy_sha256=sha(POLICY_PATH),
                         device="CPUExecutionProvider", language="yue", seed=17,
                         raw_evidence={"file": rawfile.name, "sha256": sha(rawfile)},
                         runtime_calls=calls)
            if sha(Path(request["audio_path"])) != request["audio_sha256"]:
                raise ValueError("Source audio changed during note inference")
            write(output / "prediction.json", value)
            response = {"prediction_path": str(output / "prediction.json"),
                        "prediction_sha256": sha(output / "prediction.json"),
                        "model_forwards": 1, "graph_calls": len(calls), "note_count": len(value["notes"])}
            write(output / "EXIT.json", {"status": "completed", **response, "retried": False,
                                        "elapsed_seconds": time.monotonic()-started})
            return response
        except BaseException as exc:
            write(output / "EXIT.json", {"status": "failed", "error": str(exc), "retried": False,
                    "forward_started": attempted,
                    "completed_graph_calls": len(self.runtime.calls)-initial_calls if self.runtime else 0,
                    "elapsed_seconds": time.monotonic()-started})
            self.close()
            raise


def main() -> None:
    import socket

    def offline(*args, **kwargs):
        raise RuntimeError("Network disabled in the note worker")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = offline
    wire, sys.stdout = sys.stdout, sys.stderr
    server = Server(GameEngine(), wire)
    server.emit("ready", data={"worker": "GAME", "device": "cpu"})
    try:
        import json
        for line in input_lines(sys.stdin):
            server.receive(json.loads(line))
    finally:
        server.close()


if __name__ == "__main__":
    main()
