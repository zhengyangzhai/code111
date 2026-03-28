"""run_sr_ccmt.py — Independent CLI for CCMT SR baseline

CCMT (Cascaded Cross-Modal Transformer) reproduced under our SR setting.
Usage:
    python run_sr_ccmt.py --exp_name sr_ccmt_seed42 --seed 42 --device cuda
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch

from entrain_sr_ccmt import CCMTConfig, Enrun_SR_CCMT


def parse_args():
    parser = argparse.ArgumentParser(description="CCMT SR 14-class baseline (independent entry)")

    parser.add_argument("--sr_root", type=str, default="data/SR")

    parser.add_argument("--text_model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--audio_model_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    parser.add_argument("--feature_extractor_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--freeze_text_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false")
    parser.add_argument("--freeze_audio_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_audio_encoder", dest="freeze_audio_encoder", action="store_false")
    parser.add_argument("--freeze_epochs", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use_cosine_schedule", action="store_true", default=True)
    parser.add_argument("--no_cosine_schedule", dest="use_cosine_schedule", action="store_false")

    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--rdrop_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.3)
    parser.add_argument("--swa_start_epoch", type=int, default=35)
    parser.add_argument("--use_weighted_sampler", action="store_true", default=False)
    parser.add_argument("--no_weighted_sampler", dest="use_weighted_sampler", action="store_false")

    parser.add_argument("--augment_train", action="store_true", default=True)
    parser.add_argument("--no_augment_train", dest="augment_train", action="store_false")
    parser.add_argument("--text_max_length", type=int, default=128)
    parser.add_argument("--audio_sampling_rate", type=int, default=16000)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--exp_name", type=str, default="sr_ccmt_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None, help="Speaker split seed; defaults to --seed when omitted")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None])

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = CCMTConfig(
        sr_root=args.sr_root,
        text_model_name=args.text_model_name,
        audio_model_name=args.audio_model_name,
        feature_extractor_name=args.feature_extractor_name,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_text_encoder=args.freeze_text_encoder,
        freeze_audio_encoder=args.freeze_audio_encoder,
        freeze_epochs=args.freeze_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        encoder_lr=args.encoder_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        use_cosine_schedule=args.use_cosine_schedule,
        focal_gamma=args.focal_gamma,
        rdrop_alpha=args.rdrop_alpha,
        mixup_alpha=args.mixup_alpha,
        swa_start_epoch=args.swa_start_epoch,
        use_weighted_sampler=args.use_weighted_sampler,
        augment_train=args.augment_train,
        text_max_length=args.text_max_length,
        audio_sampling_rate=args.audio_sampling_rate,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        seed=args.seed,
        split_seed=args.split_seed,
        device=device,
    )

    report = Enrun_SR_CCMT(cfg)

    print("\n===== Final CCMT SR Test Report =====")
    for key, metrics in report.items():
        if isinstance(metrics, dict) and "acc" in metrics:
            print("  %s: acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f" % (
                key, metrics["acc"], metrics["macro_f1"], metrics["weighted_f1"]))


if __name__ == "__main__":
    main()
