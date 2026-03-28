"""
论文重复实验脚本：仅训练 SR 3 次，汇总结果（均值±标准差）。
用法: python run_repeat_experiments.py

若数据不在当前项目下，请修改下面 SR_DATA_ROOT / PQP_ROOT_FOR_SR。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 减轻 CUDA OOM：减少显存碎片
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import numpy as np
from datetime import datetime

from entrain import Config, Enrun
from entrain_sr import SRConfig, Enrun_SR


# ---------- 数据路径（请按本机实际路径修改）--------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
# PQP 任务数据目录（run.py 的 data_root / split_root）
PQP_DATA_ROOT = os.path.join(_script_dir, "data", "PQP")
PQP_SPLIT_ROOT = os.path.join(_script_dir, "data", "PQP", "in-scope")
# SR 任务数据目录（run_sr.py 的 sr_root / pqp_root）
SR_DATA_ROOT = os.path.join(_script_dir, "data", "SR")
PQP_ROOT_FOR_SR = os.path.join(_script_dir, "data", "PQP")

# 3 次重复使用的随机种子（可改）
SEEDS = [42, 123, 2024]
NUM_RUNS = 3
OUTPUT_DIR = "output"
SUMMARY_FILE = "repeat_experiments_summary.json"


def run_pqp_n_times(n=3):
    """运行 PQP 训练 n 次，返回每次的 report 列表。"""
    reports = []
    for i in range(n):
        seed = SEEDS[i]
        cfg = Config()
        cfg.data_root = PQP_DATA_ROOT
        cfg.split_root = PQP_SPLIT_ROOT
        cfg.sc_labels_path = os.path.join(PQP_DATA_ROOT, "sc_labels.json")
        cfg.seed = seed
        cfg.exp_name = f"pqp_v5_run{i+1}_seed{seed}"
        cfg.output_dir = OUTPUT_DIR
        print("\n" + "=" * 60)
        print(f"  PQP 第 {i+1}/{n} 次 (seed={seed})")
        print("=" * 60)
        report = Enrun(cfg)
        reports.append((seed, report))
    return reports


def run_sr_n_times(n=3):
    """运行 SR 训练 n 次，返回每次的 report 列表。"""
    reports = []
    for i in range(n):
        seed = SEEDS[i]
        cfg = SRConfig()
        cfg.sr_root = SR_DATA_ROOT
        cfg.pqp_root = PQP_ROOT_FOR_SR
        cfg.seed = seed
        cfg.exp_name = f"sr_14class_run{i+1}_seed{seed}"
        cfg.output_dir = OUTPUT_DIR
        # 降低显存占用，避免 CUDA OOM（有效 batch 仍为 32）
        cfg.batch_size = 2
        cfg.gradient_accumulation_steps = 16
        print("\n" + "=" * 60)
        print(f"  SR 第 {i+1}/{n} 次 (seed={seed})")
        print("=" * 60)
        report = Enrun_SR(cfg)
        reports.append((seed, report))
    return reports


def summarize_pqp(reports):
    """从 PQP 的多次 report 汇总 acc / f1 的 mean±std。"""
    # 使用 test_with_best_f1 作为主结果（也可改为 test_with_best_acc）
    accs = []
    f1s = []
    for seed, report in reports:
        r = report.get("test_with_best_f1") or report.get("test_with_best_acc")
        if r:
            accs.append(r["acc"])
            f1s.append(r["f1"])
    if not accs:
        return None
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "runs": [
            {
                "seed": reports[i][0],
                "acc": accs[i],
                "f1": f1s[i],
            }
            for i in range(len(accs))
        ],
    }


def summarize_sr(reports):
    """从 SR 的多次 report 汇总 acc / macro_f1 / weighted_f1 的 mean±std。"""
    # 使用 test_best_macro_f1 作为主结果
    accs, macro_f1s, weighted_f1s = [], [], []
    for seed, report in reports:
        r = report.get("test_best_macro_f1") or report.get("test_best_acc")
        if r:
            accs.append(r["acc"])
            macro_f1s.append(r["macro_f1"])
            weighted_f1s.append(r["weighted_f1"])
    if not accs:
        return None
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
        "runs": [
            {
                "seed": reports[i][0],
                "acc": accs[i],
                "macro_f1": macro_f1s[i],
                "weighted_f1": weighted_f1s[i],
            }
            for i in range(len(accs))
        ],
    }


def main():
    print("Repeat experiments: SR only x%d (seeds: %s)" % (NUM_RUNS, SEEDS))
    start = datetime.now()

    sr_reports = run_sr_n_times(NUM_RUNS)
    sr_summary = summarize_sr(sr_reports)

    elapsed = (datetime.now() - start).total_seconds()
    summary = {
        "seeds": SEEDS,
        "sr": sr_summary,
        "elapsed_seconds": round(elapsed, 1),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, SUMMARY_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nSummary saved to: %s" % out_path)

    print("\n" + "=" * 60)
    print("  SR 结果汇总 (mean ± std)")
    print("=" * 60)
    if sr_summary:
        print("\n  SR (3 runs):")
        print("    Acc:         %.4f ± %.4f" % (sr_summary["acc_mean"], sr_summary["acc_std"]))
        print("    Macro F1:    %.4f ± %.4f" % (sr_summary["macro_f1_mean"], sr_summary["macro_f1_std"]))
        print("    Weighted F1: %.4f ± %.4f" % (sr_summary["weighted_f1_mean"], sr_summary["weighted_f1_std"]))
    print("\nTotal time: %.1f s" % elapsed)


if __name__ == "__main__":
    main()
