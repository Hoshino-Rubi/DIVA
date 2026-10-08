from __future__ import annotations

"""Miku V4C note-phoneme safety policy.

The policy is intentionally small and empirical: it captures user/manual
VOCALOID4 FE playback observations about VSQX ``note/p`` strings.  PHG2 token
membership alone is not enough to prove that a sequence will sing in a concrete
VSQX note context.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


KNOWN_SILENT_OR_UNSAFE_TOKENS = frozenset({"i\\", "i`"})
KNOWN_SILENT_OR_UNSAFE_SEQUENCES = frozenset({
    "bare @N",
    "@N",
    "f UN",
    "h e i",
    "n e i",
})
KNOWN_UNSAFE_SUFFIXES = ("_ibs", "_ibt")


@dataclass(frozen=True, slots=True)
class PhonemePolicyIssue:
    severity: str
    code: str
    message: str
    sequence: str
    token: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "sequence": self.sequence,
            "token": self.token,
        }


def note_phoneme_sequence(phonemes: Sequence[str]) -> str:
    return " ".join(str(token).strip() for token in phonemes if str(token).strip())


def validate_miku_v4c_note_phonemes(phonemes: Sequence[str]) -> list[PhonemePolicyIssue]:
    """Return empirical safety issues for one VSQX note ``p`` sequence."""

    tokens = [str(token).strip() for token in phonemes if str(token).strip()]
    sequence = note_phoneme_sequence(tokens)
    issues: list[PhonemePolicyIssue] = []

    if sequence in KNOWN_SILENT_OR_UNSAFE_SEQUENCES:
        issues.append(
            PhonemePolicyIssue(
                severity="error",
                code="known_silent_sequence",
                message=(
                    f"Miku V4C manual playback audit marks {sequence!r} as silent/unsafe "
                    "in VSQX note p context"
                ),
                sequence=sequence,
            )
        )
    if tokens == ["@N"]:
        issues.append(
            PhonemePolicyIssue(
                severity="error",
                code="bare_atn_known_silent",
                message="bare @N was manually observed as silent; use only validated onset-attached contexts",
                sequence=sequence,
                token="@N",
            )
        )

    for token in tokens:
        if token in KNOWN_SILENT_OR_UNSAFE_TOKENS:
            issues.append(
                PhonemePolicyIssue(
                    severity="error",
                    code="known_silent_token",
                    message=f"Miku V4C manual playback audit marks token {token!r} as silent/unsafe",
                    sequence=sequence,
                    token=token,
                )
            )
        if token.endswith(KNOWN_UNSAFE_SUFFIXES):
            issues.append(
                PhonemePolicyIssue(
                    severity="error",
                    code="known_silent_suffix",
                    message="all *_ibs and *_ibt tokens were manually observed as silent in this VSQX context",
                    sequence=sequence,
                    token=token,
                )
            )
    return _dedupe_issues(issues)


def assert_miku_v4c_note_phonemes_playable(phonemes: Sequence[str]) -> None:
    issues = validate_miku_v4c_note_phonemes(phonemes)
    if issues:
        rendered = "; ".join(issue.message for issue in issues)
        raise ValueError(rendered)


def collect_miku_v4c_policy_issues(note_rows: Iterable[tuple[str, Sequence[str]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for note_id, phonemes in note_rows:
        issues = validate_miku_v4c_note_phonemes(phonemes)
        if issues:
            rows.append(
                {
                    "note_id": note_id,
                    "phonemes": list(phonemes),
                    "sequence": note_phoneme_sequence(phonemes),
                    "issues": [issue.to_dict() for issue in issues],
                }
            )
    return rows


def _dedupe_issues(issues: list[PhonemePolicyIssue]) -> list[PhonemePolicyIssue]:
    seen: set[tuple[str, str, str | None]] = set()
    result: list[PhonemePolicyIssue] = []
    for issue in issues:
        key = (issue.code, issue.sequence, issue.token)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


__all__ = [
    "KNOWN_SILENT_OR_UNSAFE_SEQUENCES",
    "KNOWN_SILENT_OR_UNSAFE_TOKENS",
    "KNOWN_UNSAFE_SUFFIXES",
    "PhonemePolicyIssue",
    "assert_miku_v4c_note_phonemes_playable",
    "collect_miku_v4c_policy_issues",
    "note_phoneme_sequence",
    "validate_miku_v4c_note_phonemes",
]
