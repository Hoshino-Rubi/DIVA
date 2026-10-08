from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "vocal_stem_preprocess_plan_v1"
DEFAULT_CHAIN = (
    "BS-RoFormer",
    "UVR-DeEcho-DeReverb",
    "UVR-DeNoise",
)


@dataclass(frozen=True, slots=True)
class VocalStemPreprocessPlanConfig:
    input_audio: Path
    output_root: Path
    stem_audio: Path | None = None
    chain: Sequence[str] = DEFAULT_CHAIN
    adapter_name: str = "replaceable_external_vocal_stem_adapter"
    dry_run: bool = True


def build_vocal_stem_preprocess_plan(config: VocalStemPreprocessPlanConfig) -> dict[str, Any]:
    """Build a machine-readable plan for the replaceable vocal-stem adapter.

    This deliberately does not hard-code a UVR GUI, model path, or shell command.
    The main product pipeline can consume ``selected_pitch_audio`` immediately
    when a stem already exists, while future work can replace this plan with a
    concrete BS-RoFormer/UVR command adapter behind the same contract.
    """

    input_audio = config.input_audio
    selected_pitch_audio = config.stem_audio or input_audio
    stem_available = config.stem_audio is not None and config.stem_audio.exists()
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter": config.adapter_name,
        "adapter_kind": "replaceable_external_preprocess",
        "dry_run": config.dry_run,
        "input_audio": str(input_audio),
        "output_root": str(config.output_root),
        "selected_pitch_audio": str(selected_pitch_audio),
        "selected_pitch_audio_role": "vocal_stem" if stem_available else "input_audio_passthrough",
        "stem_audio": str(config.stem_audio) if config.stem_audio is not None else None,
        "stem_available": stem_available,
        "chain": [
            {
                "order": index + 1,
                "name": str(name),
                "enabled": True,
                "model_output": False,
                "training_data": False,
            }
            for index, name in enumerate(config.chain)
        ],
        "integration_boundary": {
            "used_for_pitch_extraction": True,
            "does_not_infer_phoneme_truth": True,
            "does_not_train_model": True,
            "external_model_adapter_replaceable": True,
            "full_mix_pitch_is_fallback_only": True,
        },
        "model_output": False,
        "training_data": False,
    }
