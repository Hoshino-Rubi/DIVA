"""Independent phone edit metrics, without duration or inventory shortcuts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


GROUPS = {
    "aspirated": lambda phone: "\u02b0" in phone,
    "front_rounded": lambda phone: any(char in phone for char in ("y", "\u028f", "\u00f8", "\u0153")),
    "labiodental_f": lambda phone: phone == "f",
    "glottal_h": lambda phone: phone == "h",
}


def align_phone_edits(reference: Sequence[str], prediction: Sequence[str]) -> list[tuple[str, str | None, str | None]]:
    n, m = len(reference), len(prediction)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]),
                costs[i - 1][j] + 1, costs[i][j - 1] + 1,
            )
    result: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            result.append(("match" if reference[i - 1] == prediction[j - 1] else "substitution", reference[i - 1], prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            result.append(("deletion", reference[i - 1], None))
            i -= 1
        else:
            result.append(("insertion", None, prediction[j - 1]))
            j -= 1
    return list(reversed(result))


def phone_metrics(pairs: Sequence[tuple[Sequence[str], Sequence[str]]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    errors: Counter[tuple[str, str, str]] = Counter()
    groups = {name: {"reference_count": 0, "substitutions": 0, "deletions": 0, "insertions": 0} for name in GROUPS}
    exact = 0
    for reference, prediction in pairs:
        counts["reference_phones"] += len(reference)
        counts["predicted_phones"] += len(prediction)
        exact += list(reference) == list(prediction)
        for operation, expected, actual in align_phone_edits(reference, prediction):
            counts[operation] += 1
            if operation != "match":
                errors[(operation, expected or "", actual or "")] += 1
            for name, belongs in GROUPS.items():
                if expected is not None and belongs(expected):
                    groups[name]["reference_count"] += 1
                    if operation in {"substitution", "deletion"}:
                        groups[name][operation + "s"] += 1
                if operation == "insertion" and actual is not None and belongs(actual):
                    groups[name]["insertions"] += 1
    total_errors = sum(counts[key] for key in ("substitution", "deletion", "insertion"))
    for row in groups.values():
        row["per"] = (row["substitutions"] + row["deletions"] + row["insertions"]) / row["reference_count"] if row["reference_count"] else None
    return {
        "reference_phones": counts["reference_phones"], "predicted_phones": counts["predicted_phones"],
        "substitutions": counts["substitution"], "deletions": counts["deletion"], "insertions": counts["insertion"],
        "edit_distance": total_errors,
        "per": total_errors / counts["reference_phones"] if counts["reference_phones"] else None,
        "sequences": len(pairs), "exact_sequences": exact,
        "groups": groups,
        "confusions": [
            {"operation": operation, "reference": expected, "prediction": actual, "count": count}
            for (operation, expected, actual), count in sorted(errors.items(), key=lambda item: (-item[1], item[0]))
        ],
        "per_is_unclipped": True,
        "timing_used_as_phone_accuracy": False,
    }


def development_gate(baseline: dict, candidates: Sequence[dict]) -> dict[str, Any]:
    seeds = [row["seed"] for row in candidates]
    if len(seeds) != len(set(seeds)) or any(seed not in {17, 29, 43} for seed in seeds):
        raise ValueError("candidate seeds must be unique members of the predeclared set")
    base_per = baseline["per"]
    results = []
    for candidate in candidates:
        if any(candidate["metrics"][key] != baseline[key] for key in ("reference_phones", "sequences")):
            raise ValueError("candidate and baseline use different development references")
        new_groups = candidate["metrics"]["groups"]
        if set(new_groups) != set(baseline["groups"]) or any(new_groups[name]["reference_count"] != row["reference_count"] for name, row in baseline["groups"].items()):
            raise ValueError("candidate and baseline use different development phone groups")
        per = candidate["metrics"]["per"]
        improved = base_per is not None and base_per > 0 and per is not None and per <= base_per * 0.95
        regressions = []
        for name, base in baseline["groups"].items():
            new = candidate["metrics"]["groups"][name]
            if base["reference_count"] >= 20 and new["per"] is not None and new["per"] > base["per"] + 0.03:
                regressions.append(name)
        results.append({
            "seed": candidate["seed"], "relative_per_improvement": (base_per - per) / base_per if base_per and per is not None else None,
            "improvement_passed": improved, "group_regressions": regressions,
            "passed": improved and not regressions,
        })
    passing = [row for row in results if row["passed"]]
    eligible = [row for row in candidates if row["seed"] in {value["seed"] for value in passing}]
    selected = min(eligible, key=lambda row: (row["metrics"]["per"], row["seed"]))["seed"] if len(passing) >= 2 else None
    return {
        "passed": len(passing) >= 2, "selected_seed": selected, "seeds": results,
        "minimum_relative_improvement": 0.05, "minimum_passing_seeds": 2,
        "group_minimum_reference_count": 20, "maximum_group_per_regression": 0.03,
        "selection_source": "development_only", "automatic_model_promotion": False,
    }
