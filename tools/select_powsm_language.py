"""Select only between the two predeclared language conditions using training songs."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file
from any2xsampa.training.phone_error_metrics import phone_metrics


def select(run: Path) -> None:
    policy = read_json(run / "POLICY.json")
    frozen = read_json(run / "FREEZE.json")
    verify_file(run / "POWSM_SCORING_REFERENCES.json", frozen["original_references_sha256"])
    refs = {r["example_id"]: r["target_ipa"] for r in read_json(run / "POWSM_SCORING_REFERENCES.json")["rows"]}
    summaries = {}
    for name in ("unknown", "cantonese"):
        result = read_json(run / ("powsm-" + name) / "RESULT.json")
        if not result["complete"] or result["reference_received"] or result["development"]:
            raise ValueError("invalid training-language probe")
        pairs, by_song = [], {s: [] for s in policy["songs"]}
        details = []
        for row in result["entries"]:
            verify_file(Path(row["posterior_path"]), row["posterior_sha256"])
            key = row["request"]["example_id"]
            pair = refs[key], row["prediction"]
            pairs.append(pair)
            by_song[row["request"]["song_id"]].append(pair)
            details.append({"example_id": key, "reference": pair[0], "prediction": pair[1],
                "metrics": phone_metrics([pair]), "fixture_verified": row["fixture_verified"]})
        summaries[name] = {"metrics": phone_metrics(pairs),
            "songs": {s: phone_metrics(p) for s,p in by_song.items()}, "examples": details,
            "result_sha256": sha256_file(run / ("powsm-" + name) / "RESULT.json")}
    baseline, candidate = summaries["unknown"], summaries["cantonese"]
    gain = 1 - candidate["metrics"]["per"] / baseline["metrics"]["per"]
    nonworse = [s for s in policy["songs"] if candidate["songs"][s]["per"] <= baseline["songs"][s]["per"]]
    passed = gain >= policy["powsm"]["minimum_relative_gain"] and len(nonworse) >= policy["powsm"]["minimum_nonworse_songs"]
    write_json_new(run / "POWSM_LANGUAGE_SELECTION.json", {"passed": passed,
        "selected_condition": "cantonese" if passed else None, "training_relative_PER_gain": gain,
        "nonworse_songs": nonworse, "conditions": summaries, "development_used_for_selection": False,
        "passed_means": "language ablation gain only; not release or architectural superiority",
        "policy_sha256": sha256_file(run / "POLICY.json"), "default_replacement": False})
    print("Cantonese condition selected:", passed, "relative gain", gain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    select(parser.parse_args().run.resolve())
