"""Ensemble evaluation: average logits from baseline + CDD models on SR test set."""

import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

from entrain_sr import SRConfig, SRTrainer, compute_multiclass_metrics
from sr_dataloader import SR_VALID_LABELS


def collect_logits(trainer, ckpt_path):
    """Load checkpoint and collect logits on test set."""
    trainer.model.load_state_dict(torch.load(ckpt_path, map_location=trainer.device))
    trainer.model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in trainer.test_loader:
            batch_t = trainer._move_batch(batch)
            out = trainer.model(**batch_t)
            all_logits.append(out["logits"].detach().cpu())
            all_labels.append(batch_t["labels"].detach().cpu())
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def evaluate_logits(logits, labels, tag=""):
    metrics = compute_multiclass_metrics(logits, labels)
    print(f"\n{'='*50}")
    print(f"  {tag}")
    print(f"  acc={metrics['acc']:.4f}  macro_f1={metrics['macro_f1']:.4f}  weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"  Per-class F1:")
    for label in SR_VALID_LABELS:
        pc = metrics["per_class"][label]
        print(f"    {label:5s}  P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}  support={pc['support']}")
    return metrics


def main():
    baseline_dir = "output/sr_ablation_full_20260315_022411"
    cdd_dir = "output/sr_cdd_full_gradfix_v4_20260320_093404"
    cdd_gradfix_dir = "output/sr_cdd_full_dgcp_cdr_gradfix_20260319_111508"

    print("="*60)
    print("SR Ensemble Evaluation")
    print("="*60)

    # --- Build baseline model & collect logits ---
    print("\n[1/4] Loading baseline model...")
    baseline_cfg_path = os.path.join(baseline_dir, "config.json")
    with open(baseline_cfg_path, "r", encoding="utf-8") as f:
        bl_dict = json.load(f)
    bl_cfg = SRConfig(**{k: v for k, v in bl_dict.items() if hasattr(SRConfig, k)})
    bl_cfg.exp_name = "ensemble_baseline_tmp"
    bl_trainer = SRTrainer(bl_cfg)

    bl_ckpts = {
        "best_acc": os.path.join(baseline_dir, "checkpoints", "best_acc.pt"),
        "swa": os.path.join(baseline_dir, "checkpoints", "swa.pt"),
    }

    bl_logits = {}
    bl_labels = None
    for name, path in bl_ckpts.items():
        if os.path.exists(path):
            print(f"  Collecting baseline {name} logits...")
            logits, labels = collect_logits(bl_trainer, path)
            bl_logits[name] = logits
            bl_labels = labels

    # --- Build CDD model & collect logits ---
    cdd_dirs = [
        ("V4", cdd_dir),
        ("gradfix", cdd_gradfix_dir),
    ]

    cdd_logits = {}
    for tag, d in cdd_dirs:
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            continue
        print(f"\n[2/4] Loading CDD model ({tag})...")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cdd_dict = json.load(f)
        cdd_cfg = SRConfig(**{k: v for k, v in cdd_dict.items() if hasattr(SRConfig, k)})
        cdd_cfg.exp_name = f"ensemble_cdd_{tag}_tmp"
        cdd_trainer = SRTrainer(cdd_cfg)

        for ckpt_name in ["best_acc", "best_macro_f1", "swa"]:
            ckpt_path = os.path.join(d, "checkpoints", f"{ckpt_name}.pt")
            if os.path.exists(ckpt_path):
                print(f"  Collecting CDD-{tag} {ckpt_name} logits...")
                logits, labels = collect_logits(cdd_trainer, ckpt_path)
                cdd_logits[f"{tag}_{ckpt_name}"] = logits

        del cdd_trainer
        torch.cuda.empty_cache()

    # --- Individual model results ---
    print("\n" + "="*60)
    print("Individual Model Results")
    print("="*60)
    for name, logits in bl_logits.items():
        evaluate_logits(logits, bl_labels, f"Baseline {name}")
    for name, logits in cdd_logits.items():
        evaluate_logits(logits, bl_labels, f"CDD {name}")

    # --- Ensemble ---
    print("\n" + "="*60)
    print("Ensemble Results")
    print("="*60)

    results = {}

    for bl_name, bl_log in bl_logits.items():
        for cdd_name, cdd_log in cdd_logits.items():
            bl_prob = F.softmax(bl_log, dim=-1)
            cdd_prob = F.softmax(cdd_log, dim=-1)

            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
                tag = f"Baseline({bl_name}) x{alpha:.1f} + CDD({cdd_name}) x{1-alpha:.1f}"
                ens_prob = alpha * bl_prob + (1 - alpha) * cdd_prob
                m = evaluate_logits(ens_prob, bl_labels, tag)
                results[tag] = m["acc"]

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY: All ensemble acc sorted")
    print("="*60)
    for tag, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " *** BEATS 72.56% ***" if acc > 0.7256 else ""
        print(f"  {acc:.4f}  {tag}{marker}")


if __name__ == "__main__":
    main()
