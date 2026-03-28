"""Quick evaluation script: load checkpoint from a crashed/interrupted run and evaluate on test set."""

import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from entrain_sr import SRConfig, SRTrainer, compute_multiclass_metrics
from sr_dataloader import SR_VALID_LABELS


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_checkpoint.py <run_dir> [checkpoint_name]")
        print("  e.g. python eval_checkpoint.py output/sr_cdd_full_dgcp_cdr_gradfix_20260319_111508")
        print("  checkpoint_name defaults to 'best_acc.pt'")
        sys.exit(1)

    run_dir = sys.argv[1]
    ckpt_name = sys.argv[2] if len(sys.argv) > 2 else "best_acc.pt"

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg = SRConfig(**{k: v for k, v in cfg_dict.items() if hasattr(SRConfig, k)})
    cfg.exp_name = "eval_" + os.path.basename(run_dir)

    trainer = SRTrainer(cfg)

    old_ckpt_dir = os.path.join(run_dir, "checkpoints")

    final_report = {}
    for ckpt in ["best_acc.pt", "best_macro_f1.pt"]:
        ckpt_path = os.path.join(old_ckpt_dir, ckpt)
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping")
            continue

        print(f"\n===== Loading {ckpt} =====")
        trainer.model.load_state_dict(torch.load(ckpt_path, map_location=trainer.device))
        test_m = trainer.do_test("test")
        key = ckpt.replace(".pt", "")
        final_report[f"test_{key}"] = test_m
        print(f"Test ({key}): acc={test_m['acc']:.4f}  macro_f1={test_m['macro_f1']:.4f}  weighted_f1={test_m['weighted_f1']:.4f}")
        print("Per-class F1:")
        for label in SR_VALID_LABELS:
            pc = test_m["per_class"][label]
            print(f"  {label:5s}  P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}  support={pc['support']}")

    trainer._log_json("final_test_report.json", final_report)
    print(f"\nReport saved to {trainer.run_dir}/final_test_report.json")


if __name__ == "__main__":
    main()
