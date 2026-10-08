"""One-shot completion/replay/report workflow. No training or automatic retries."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new
from event_head_completion_common import load_completion


def readiness(run: Path) -> dict:
    missing = []
    for song in ("promise", "queen-road-east"):
        exitfile = run / "commands" / f"E--{song}.exit.json"
        result = run / "screen/E" / song / "RESULT.json"
        if exitfile.exists() and read_json(exitfile)["exit_code"] != 0:
            return {"ready": False, "blocked": True, "reason": song + " failed; no automatic retry"}
        if not exitfile.exists() or not result.exists():missing.append(song)
        elif not read_json(result)["complete"]:
            return {"ready": False, "blocked": True, "reason": song + " incomplete"}
    return {"ready": not missing, "blocked": False, "missing": missing}


def execute(run: Path, name: str, command: list[str], timeout: int, *, cpu_only: bool) -> None:
    load_completion(run, "CODE_REVIEW_postprocess.json")
    review = read_json(run / "CODE_REVIEW_postprocess.json")
    for p, digest in review.get("data_bindings", {}).items():verify_file(Path(p), digest)
    output = run / "postprocess"
    write_json_new(output / (name + ".command.json"), {"command": command,
        "at_utc": datetime.now(timezone.utc).isoformat(), "cpu_only": cpu_only, "automatic_retry": False})
    before = time.monotonic();code = None;error = None
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if cpu_only:env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with (output / (name + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        code = result.returncode
    except Exception as exc:error = repr(exc)
    finally:write_json_new(output / (name + ".exit.json"), {"exit_code": code, "error": error, "elapsed_seconds": time.monotonic() - before})
    if error or code != 0:raise RuntimeError(name + " failed; no automatic retry")
    print("POSTPROCESS COMPLETE", name, flush=True)


def render(run: Path) -> None:
    report = read_json(run / "SCREEN_REPORT.json");verified = read_json(run / "FINAL_VERIFICATION.json")
    if not verified["passed"]:raise ValueError("delivery verification required")
    tests = verified["regression"];passed = tests["tests"] - tests["skipped"]
    status = {"status": "screen_completed_verified", "at_utc": datetime.now(timezone.utc).isoformat(),
        "screen_complete": True, "ledger": report["ledger"], "selected_arm": report["selected_arm"],
        "completed_models": 10, "effective_model_epochs": 200, "effective_model_updates": report["effective_optimizer_updates"],
        "epoch_snapshots_verified": 40, "head_output_replays": 298, "full_audio_replays": 10,
        "codec_coverage": {"previous_proofs_reverified": 229, "new_checks": 69},
        "regression_passed": passed, "regression_skipped": tests["skipped"],
        "conditional_final_status": "eligible_pending_separate_reviewed_worker" if report["selected_arm"] else "not_run_screen_failed",
        "default_replaced": False, "final_test_opened": False, "new_human_listening_required": False}
    write_json_new(run / "FINAL_STATUS.json", status)
    lines = ["# H／E完整五折：追加授权后的最终筛选报告", "", "完成时间（UTC）：" + status["at_utc"], "",
        "## 结论", "",
        ("存在满足冻结筛选条件的臂；最终三个种子尚未执行，需要继续原协议的独立阶段，不自动部署。" if report["selected_arm"] else "H和E均未通过完整冻结门槛。本轮不启动最终三个种子，不新增沙龙开发或最终测试评测，不替换默认模型。无需补听或补标。"), "",
        "H为独立线性CTC头；E为同一内容头加事件支路。两者使用相同原混音、参考、留歌划分、采样、20epoch和更新预算；E同时改变了序列目标及解码方式，不能把结果单独归因于边界头。", "",
        "## 原唱留歌结果（同1235个参考音素）", "", "| 系统 | 错误数 | PER | 替换 | 删除 | 插入 |", "|---|---:|---:|---:|---:|---:|"]
    metrics = [("旧endpoint", report["historical_endpoint"]["original"])] + [(a, report["gates"][a]["original"]) for a in ("H", "E")]
    for label, m in metrics:
        lines.append(f"| {label} | {m['edit_distance']} | {m['per']:.6f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} |")
    lines += ["", "固定必要条件包括：总错≤564、删除≤248、插入≤32、至少三首歌改善、歌曲宏平均不退化、关键音类及合成保护；E还须相对同轮H降低PER至少5%。", ""]
    for arm in ("H", "E"):
        gate = report["gates"][arm]
        lines.append(f"- {arm}：{'通过' if gate['passed'] else '未通过'}；失败项：`{'；'.join(gate['failures']) or '无'}`。")
    names = {"freedom-flower": "自由花", "half-moon-night-song": "月半小夜曲", "like-you": "喜欢你", "promise": "约定", "queen-road-east": "皇后大道東"}
    lines += ["", "| 留出歌曲 | 参考数 | 旧endpoint错误 | H错误 | E错误 |", "|---|---:|---:|---:|---:|"]
    for song, name in names.items():
        old = report["gates"]["H"]["per_song"]["baseline"][song]
        hh = report["gates"]["H"]["per_song"]["current"][song];ee = report["gates"]["E"]["per_song"]["current"][song]
        lines.append(f"| {name} | {old['reference_phones']} | {old['edit_distance']} | {hh['edit_distance']} | {ee['edit_distance']} |")
    lines += ["", "## 关键音类（错误/参考数）", "", "| 音类 | 旧endpoint | H | E | 证据范围 |", "|---|---:|---:|---:|---|"]
    for group, name in [("aspirated", "送气"), ("glottal_h", "h"), ("front_rounded", "前圆唇"), ("labiodental_f", "f")]:
        cells = []
        for _, m in metrics:
            g = m["groups"][group];errors = sum(g[k] for k in ("substitutions", "deletions", "insertions"))
            cells.append(f"{errors}/{g['reference_count']}")
        enough = metrics[0][1]["groups"][group]["reference_count"] >= 20
        lines.append("| " + name + " | " + " | ".join(cells) + " | " + ("纳入保护门槛" if enough else "样本不足，仅描述") + " |")
    lines += ["", "## 合成留歌保护（129个窗口，不是沙龙开发集）", "", "| 系统 | 参考数 | 总错误 | PER |", "|---|---:|---:|---:|"]
    synth = [("旧endpoint", report["historical_endpoint"]["rendered"])] + [(a, report["gates"][a]["rendered"]) for a in ("H", "E")]
    for label, m in synth:lines.append(f"| {label} | {m['reference_phones']} | {m['edit_distance']} | {m['per']:.6f} |")
    strata = defaultdict(Counter)
    for row in report["note_phone_strata"]:
        key = (row["arm"], row["position"], row["note_kind"])
        for field in ("reference", "match", "substitution", "deletion"):strata[key][field] += row.get(field, 0)
    lines += ["", "## 音符内部与多元音位置诊断", "", "下表按参考位置统计，不含无参考位置的插入；主PER和逐窗口指标仍完整包含插入。边缘指窗口首尾音符，不是人工音素持续时间。", "",
        "| 臂 | 位置 | 音符类型 | 参考数 | 替换 | 删除 |", "|---|---|---|---:|---:|---:|"]
    for (arm, position, kind), s in sorted(strata.items()):
        lines.append(f"| {arm} | {position} | {kind} | {s['reference']} | {s['substitution']} | {s['deletion']} |")
    lines += ["", "## 启动与失败账本", "",
        "- 累计11次质量启动：原9次加本次批准的2次；其中1次使用追加额度重做约定，1次是原未启动的皇后大道東。已有八个完整模型未重训。",
        "- 9次质量进程正常退出；自由花在20epoch完成后评测超时，保留退出73并只读恢复；旧约定在第14个epoch后整卡显存超限，退出73且无最终检查点，保留失败目录。",
        f"- 十个有效模型共200epoch、H/E各2140更新。包含失败约定的历史，已落盘{report['ledger']['logged_epochs_including_failed_attempt']}个epoch、{report['ledger']['logged_optimizer_steps_including_failed_attempt']}次更新；失败epoch内额外未落盘步数未知，不能把200epoch当作实际总消耗。",
        "- 新约定从相同预训练初始化重新开始，不是从丢失的第15epoch续跑。没有修改数据、损失、采样、epoch或晋级阈值。",
        f"- 新旧约定初始化哈希一致；前14个已记录epoch的平均序列损失最大差为{report['first14_logged_loss_maximum_difference']:.10g}（诊断，不用于选择）。",
        "- 新worker每epoch保存权重、AdamW、Python/NumPy/torch CPU/CUDA RNG及代码/数据绑定；共40份提交标记和状态哈希通过，最终状态与交付权重逐元素一致。中断写入没有提交标记则不接受。",
        "- 运行上限仍为1500秒、进程reserved 7 GiB、整卡14 GiB；启动前增加30秒内三次≤12 GiB余量检查。只停止自己的超限任务，不自动重试。",
        "- 第一折历史超时进程未落盘完整CUDA峰值，其reserved仍明确为epoch观测最大值，不冒称精确进程峰值。", "",
        "## 独立验证与可复现性", "",
        "- 新进程重放全部10个检查点、298份留出输出，并复算149份旧endpoint控制。",
        "- 10个预先固定的完整原唱音频前向与缓存特征、内容分数、事件输出一致；不是按分数挑片段。",
        "- 保存证据的codec/fixture覆盖298份：229份既有核验凭证重新核对绑定哈希，69份新增输出重新执行独立CPU解码及fixture核验；不重复昂贵旧核验冒充新工作。",
        f"- 完整回归：{passed} passed，{tests['skipped']} skipped，0 failed。51个受保护原WAV、人工VSQX、映射、默认配置等哈希通过。",
        "- 神经与音段打分FP32 CUDA，精确动态规划FP64 CPU；没有时长截断或标签剪枝。事件时间不当作人工音素时长准确率。",
        "- 同五首歌已用于多轮研究选择，结果不是新的独立泛化证据。没有打开最终测试预测，没有自动部署。", "",
        "## 文件与复现入口", "",
        "AUTHORIZATION.json：额外一次授权与跨目录预算；RESOLVED_RESULTS.json：8个旧结果和2个新结果的明确来源。",
        "SCREEN_REPORT.json：全部指标、音类、逐窗及位置分析；FINAL_VERIFICATION.json：重放、codec、状态和来源检查。",
        "各commands与postprocess/*.command.json记录实际命令；训练目录已经占用，不能照抄命令进行未授权重跑。", ""]
    text = "\n".join(lines)
    with (run / "RESULTS.md").open("x", encoding="utf-8") as stream:stream.write(text)
    if sum("\u4e00" <= ch <= "\u9fff" for ch in (run / "RESULTS.md").read_text(encoding="utf-8")) < 300:
        raise ValueError("Chinese report encoding damaged")
    files = [run / n for n in ("RESULTS.md", "FINAL_STATUS.json", "FINAL_VERIFICATION.json", "SCREEN_REPORT.json", "AUTHORIZATION.json", "RESOLVED_RESULTS.json", "CODE_REVIEW_training.json", "CODE_REVIEW_runtime_supplement.json", "CODE_REVIEW_postprocess.json", "CODE_REVIEW_replay.json")]
    files += list((run / "postprocess").glob("*.json")) + [run / "postprocess/regression.xml", run / "collection/independent-replay/RESULT.json"]
    write_json_new(run / "FINAL_INDEX.json", {"screen_complete": True, "artifacts": {str(p): sha256_file(p) for p in files}, "nested_model_and_raw_bindings": "FINAL_VERIFICATION.json"})
    print("FINAL DELIVERY WRITTEN", report["selected_arm"], flush=True)


def finish(run: Path) -> None:
    auth, source, freeze, spec, _ = load_completion(run, "CODE_REVIEW_postprocess.json")
    out = run / "postprocess"
    write_json_new(out / "START.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "starts_training": False, "automatic_retry": False})
    deadline = time.monotonic() + 2 * 3600
    while True:
        state = readiness(run)
        if state["blocked"]:write_json_new(out / "BLOCKED.json", state);return
        if state["ready"]:break
        if time.monotonic() > deadline:
            write_json_new(out / "BLOCKED.json", {"reason": "existing worker wait deadline; no new training"});return
        time.sleep(30)
    ordinary = str(ROOT / ".venv/Scripts/python.exe");isolated = freeze["python"]
    execute(run, "collect", [ordinary, "-X", "utf8", str(ROOT / "tools/event_head_completion_results.py"), "--run", str(run), "--stage", "collect"], 300, cpu_only=True)
    execute(run, "independent-replay", [isolated, "-X", "utf8", str(ROOT / "tools/replay_event_head_comparison.py"), "--run", str(run / "collection")], 1550, cpu_only=False)
    execute(run, "screen-report", [ordinary, "-X", "utf8", str(ROOT / "tools/event_head_completion_results.py"), "--run", str(run), "--stage", "report"], 300, cpu_only=True)
    execute(run, "regression", [ordinary, "-X", "utf8", "-m", "pytest", "-q", "--basetemp=" + str(out / "regression-temp"), "--junitxml=" + str(out / "regression.xml")], 900, cpu_only=True)
    execute(run, "final-verification", [isolated, "-X", "utf8", str(ROOT / "tools/verify_event_head_completion.py"), "--run", str(run)], 2400, cpu_only=True)
    render(run)
    write_json_new(out / "COMPLETE.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "selected_arm": read_json(run / "SCREEN_REPORT.json")["selected_arm"], "automatic_deployment": False})


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);run=p.parse_args().run.resolve()
    try:finish(run)
    except Exception as exc:
        path=run / "postprocess/FAILED.json"
        if not path.exists():write_json_new(path,{"error":repr(exc),"automatic_retry":False,"at_utc":datetime.now(timezone.utc).isoformat()})
        raise
