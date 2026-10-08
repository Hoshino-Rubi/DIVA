"""Product adapter for a trained Miku V4C phone-candidate ranker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from any2xsampa.training.miku_v4c_ranker import (
    PAD_TOKEN,
    PhonemeRankerExample,
    UNK_TOKEN,
    load_ranker_checkpoint,
    phoneme_ranker_collate,
)


@dataclass(frozen=True, slots=True)
class RankedPhoneCandidate:
    candidate_id: str
    score: float
    rank: int
    checkpoint_sha256: str
    model_output: bool = True
    truth_layer: str = "phones_ipa_articulatory_features"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate(value: Mapping[str, Any], vocab: Any) -> PhonemeRankerExample:
    candidate_id = str(value.get("candidate_id") or "").strip()
    lyrics = str(value.get("lyrics") or "").strip()
    raw_tokens = value.get("phoneme_tokens")
    if not isinstance(raw_tokens, Sequence) or isinstance(raw_tokens, (str, bytes, bytearray)):
        raise ValueError(f"ranker candidate requires phoneme_tokens: {candidate_id or '<missing>'}")
    tokens = tuple(str(token).strip() for token in raw_tokens)
    if not candidate_id or not lyrics or not tokens or any(not token for token in tokens):
        raise ValueError("ranker candidate requires ID, lyrics, and non-empty phone tokens")
    if value.get("truth_layer") != "phones-ipa-articulatory":
        raise ValueError(f"ranker candidate has an unsupported truth layer: {candidate_id}")
    phones = value.get("phones")
    if not isinstance(phones, list) or len(phones) != len(tokens):
        raise ValueError(f"ranker candidate phone evidence length differs from tokens: {candidate_id}")
    for token, evidence in zip(tokens, phones):
        if (
            not isinstance(evidence, Mapping)
            or str(evidence.get("phone") or "").strip() != token
            or not str(evidence.get("ipa") or "").strip()
            or not isinstance(evidence.get("features"), Mapping)
            or not evidence["features"]
        ):
            raise ValueError(f"ranker candidate has incomplete phone evidence: {candidate_id}")
    unknown = sorted({token for token in tokens if token not in vocab.phone_to_id})
    if unknown:
        raise ValueError(f"ranker candidate contains out-of-vocabulary phones: {', '.join(unknown)}")
    special = sorted({token for token in tokens if token in {PAD_TOKEN, UNK_TOKEN}})
    if special:
        raise ValueError(f"ranker candidate contains reserved vocabulary tokens: {', '.join(special)}")
    return PhonemeRankerExample(
        lyrics=lyrics,
        phoneme_tokens=tokens,
        label=0.0,
        score=None,
        char_ids=vocab.encode_chars(lyrics),
        phoneme_ids=vocab.encode_phonemes(tokens),
        metadata=dict(value),
    )


class MikuV4CRankerAdapter:
    """Load once and score auditable phone candidates with stable ranking."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.payload = load_ranker_checkpoint(self.checkpoint_path, device=device)
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)

    def score(self, candidates: Sequence[Mapping[str, Any]]) -> list[RankedPhoneCandidate]:
        if not candidates:
            raise ValueError("ranker requires at least one candidate")
        identifiers = [str(value.get("candidate_id") or "").strip() for value in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ranker candidate IDs must be unique")
        examples = [_candidate(value, self.model.vocab) for value in candidates]
        batch = phoneme_ranker_collate(examples)
        device = next(self.model.parameters()).device
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        torch = __import__("torch")
        with torch.inference_mode():
            logits = self.model(
                batch["char_ids"],
                batch["phoneme_ids"],
                batch["char_mask"],
                batch["phoneme_mask"],
            )
            scores = torch.sigmoid(logits).detach().cpu().tolist()
        ordered = sorted(
            zip(identifiers, scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [
            RankedPhoneCandidate(
                candidate_id=candidate_id,
                score=float(score),
                rank=index,
                checkpoint_sha256=self.checkpoint_sha256,
            )
            for index, (candidate_id, score) in enumerate(ordered, start=1)
        ]


__all__ = ["MikuV4CRankerAdapter", "RankedPhoneCandidate"]
