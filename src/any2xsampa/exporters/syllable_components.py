from collections.abc import Sequence

# V4C strings come straight from <p>. Keep the rest intact as the whole final.
_ONSETS = frozenset({
    "p", "p_h", "t", "t_h", "k", "k_h", "ts", "ts_h", "ts`", "ts`_h", r"ts\_h",
    "m", "n", "l", "f", "x", "s",
})


def components(phones: Sequence[str]) -> tuple[str, tuple[str, ...], str]:
    if not phones:
        return "", (), ""
    onset = phones[0] if phones[0] in _ONSETS else ""
    final = tuple(phones[1:] if onset else phones)
    if not final:
        return onset, final, ""
    last = final[-1]
    if len(final) > 1 and last in {"p", "t", "k", "m", "n", "N", "i", "u"}:
        coda = last
    elif last.endswith(("N", "_n")):
        coda = "N" if last.endswith("N") else "n"
    elif last in {"aI", "uaI", "ei", "uei"}:
        coda = "i"
    elif last in {"AU", "iAU", "@U", "i@U"}:
        coda = "u"
    else:
        coda = ""
    return onset, final, coda
