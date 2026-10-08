"""Build a gate report after the V/MV screen, without opening development/test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from any2xsampa.training.input_supervision_evaluation import compare_screen, choose_pipeline
from any2xsampa.training.input_supervision import load_manifest
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new


SONGS = ("freedom-flower", "half-moon-night-song", "like-you", "promise", "queen-road-east")


def _screen_rows(run: Path, variant: str) -> list[dict]:
    rows = []
    for song in SONGS:
        result = run / "screen" / variant / song / "RESULT.json"
        if result.exists():
            rows.append(read_json(result))
    return rows


def _baseline(run: Path) -> tuple[dict, dict, dict]:
    probe_result = read_json(Path(read_json(run / "FREEZE.json")["probe_run"]) / "RESULT.json")
    metrics = probe_result["condition_metrics"]
    songs = probe_result["per_song"]
    return metrics, songs, probe_result['details']


def verify_completed_job(run: Path, row: dict, freeze: dict, schedule: dict) -> None:
    variant, song = row['variant'], row['held_song']
    if (row['freeze_sha256'] != sha256_file(run / 'FREEZE.json')
            or row['policy_sha256'] != sha256_file(run / 'POLICY.json')
            or row['code_bindings'] != freeze['code_bindings']
            or row['frozen_parameters_unchanged'] is not True
            or [e['epoch'] for e in row['history']] != list(range(1, 21))
            or set(row['initial_logit_errors']) != {'x', 'v', 'rendered'}
            or not all(0 <= error <= 2e-5 for error in row['initial_logit_errors'].values())):
        raise ValueError('completed job identity, initialization or freeze verification failed')
    verify_file(Path(row['checkpoint_path']), row['checkpoint_sha256'])
    for event in row['history']:
        expected = schedule[(variant, song, event['epoch'])]
        if any(event[k] != expected[k] for k in ('source_order_sha256', 'optimizer_steps')):
            raise ValueError('actual training schedule differs from preflight/control')
    monitor = read_json(run / 'screen' / variant / song / 'monitor/resource-monitor.json')
    if monitor['operation_succeeded'] is not True or monitor['completed_without_resource_abort'] is not True:
        raise ValueError('job did not exit successfully under resource protection')
    command_exit = read_json(run / 'commands' / f'{variant}-{song}.exit.json')
    if command_exit['exit_code'] != 0:
        raise ValueError('training worker exit status was not successful')


def report(run: Path) -> None:
    run = run.resolve()
    freeze = read_json(run / 'FREEZE.json')
    current = freeze.get('schema_version') == 'input_supervision_freeze_v2'
    if current:
        load_manifest(run)
        schedule = {(r['variant'], r['held_song'], r['epoch']): r
                    for r in read_json(run / 'PREFLIGHT.json')['schedule_checks']}
    baseline, per_song, details = _baseline(run)
    results = {}
    for variant in ("V", "MV"):
        rows = _screen_rows(run, variant)
        if any(row.get('schema_version') != 'input_supervision_job_v2' for row in rows):
            gate = {'variant': variant, 'complete': False, 'passed': False, 'failures': ['invalid_control_schedule_v1']}
        else:
            if current:
                for row in rows:
                    verify_completed_job(run, row, freeze, schedule)
            gate = compare_screen(variant, rows, baseline, per_song, details)
        results[variant] = {"folds_present": [row["held_song"] for row in rows], "gate": gate}
    payload = {
        "schema_version": "input_supervision_screen_report_v1",
        "run": str(run),
        "probe_result_sha256": sha256_file(Path(read_json(run / "FREEZE.json")["probe_run"]) / "RESULT.json"),
        "results": results,
        "screen_complete": all(value["gate"].get("complete") for value in results.values()),
        "any_variant_passed": any(value["gate"].get("passed") for value in results.values()),
        "development_opened": False,
        "final_test_opened": False,
        "automatic_promotion": False,
        "selected_pipeline": choose_pipeline([r['gate'] for r in results.values()]),
    }
    write_json_new(run / "SCREEN_REPORT.json", payload)
    lines = ["# V/MV 输入监督筛选", "", "本报告不打开沙龙开发集或最终测试。", ""]
    for variant, value in results.items():
        gate = value["gate"]
        status = '通过' if gate.get('passed') else '未通过' if gate.get('complete') else '未完成有效比较'
        lines.append(f"- **{variant}**：{status}；结果文件 {len(value['folds_present'])}/5；原因：{', '.join(gate.get('failures', [])) or '无'}")
    with (run / 'SCREEN_REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True); args = parser.parse_args(); report(args.run)
