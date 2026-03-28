"""run.py — 启动入口"""

import os

# 修复 OpenMP 重复加载问题（必须在 import torch 之前设置）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch

from entrain import Config, Enrun


def parse_args():
    parser = argparse.ArgumentParser(description="PQP multimodal training (V5)")

    # ---- 数据 ----
    parser.add_argument("--data_root", type=str, default="data/PQP")
    parser.add_argument("--split_root", type=str, default="data/PQP/in-scope")
    parser.add_argument("--train_file", type=str, default="train.tsv")
    parser.add_argument("--dev_file", type=str, default="dev.tsv")
    parser.add_argument("--test_file", type=str, default="test.tsv")

    # ---- 模型 ----
    parser.add_argument("--text_model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--audio_model_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    parser.add_argument("--feature_extractor_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    # ---- SpeechCraft ----
    parser.add_argument("--sc_labels_path", type=str, default="data/PQP/sc_labels.json")

    # ---- 消融 / 单模态基线 ----
    parser.add_argument("--modality", type=str, default="multimodal", choices=["multimodal", "text_only", "audio_only", "pbcf", "pbcf_no_cross_attn", "pbcf_no_discrepancy", "drbf", "cdd"], help="multimodal | text_only | audio_only | pbcf | pbcf_no_cross_attn | pbcf_no_discrepancy | drbf | cdd")
    parser.add_argument("--no_prosody", dest="use_prosody", action="store_false", help="Ablation: disable prosody branch")
    parser.add_argument("--no_frame_acoustic", dest="use_frame_acoustic", action="store_false", help="Ablation: disable frame acoustic branch")
    parser.add_argument("--no_token_disc", action="store_true", help="CDD-Net ablation: use sentence-level discrepancy instead of token-level (TLDL)")
    parser.add_argument("--no_dual_contrastive", action="store_true", help="CDD-Net ablation: disable dual-space contrastive losses (align + sep)")
    parser.add_argument("--lambda_con", type=float, default=0.01, help="CDD contrastive loss weight")
    parser.add_argument("--lambda_orth", type=float, default=0.001, help="CDD orthogonality loss weight")
    parser.add_argument("--lambda_align", type=float, default=0.005, help="CDD alignment loss weight")
    parser.add_argument("--lambda_sep", type=float, default=0.001, help="CDD separation loss weight")
    parser.add_argument("--cdd_loss_warmup_epochs", type=int, default=5, help="Linearly warmup CDD auxiliary losses over this many epochs")

    # ---- 冻结策略 ----
    parser.add_argument("--freeze_text_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false")
    parser.add_argument("--freeze_audio_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_audio_encoder", dest="freeze_audio_encoder", action="store_false")
    parser.add_argument("--freeze_epochs", type=int, default=5)

    # ---- 优化 ----
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--exp_name", type=str, default="pqp_v5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.15)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use_cosine_schedule", action="store_true", default=True)
    parser.add_argument("--no_cosine_schedule", dest="use_cosine_schedule", action="store_false")

    # ---- R-Drop ----
    parser.add_argument("--rdrop_alpha", type=float, default=1.0)

    # ---- 数据增强 ----
    parser.add_argument("--augment_train", action="store_true", default=True)
    parser.add_argument("--no_augment_train", dest="augment_train", action="store_false")

    # ---- 数据处理 ----
    parser.add_argument("--text_max_length", type=int, default=64)
    parser.add_argument("--audio_sampling_rate", type=int, default=16000)
    parser.add_argument("--num_workers", type=int, default=0)

    # ---- 其他 ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None])

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_root=args.data_root,
        split_root=args.split_root,
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        text_model_name=args.text_model_name,
        audio_model_name=args.audio_model_name,
        feature_extractor_name=args.feature_extractor_name,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        sc_labels_path=args.sc_labels_path,
        modality=args.modality,
        use_prosody=args.use_prosody,
        use_frame_acoustic=args.use_frame_acoustic,
        no_token_disc=args.no_token_disc,
        no_dual_contrastive=args.no_dual_contrastive,
        lambda_con=args.lambda_con,
        lambda_orth=args.lambda_orth,
        lambda_align=args.lambda_align,
        lambda_sep=args.lambda_sep,
        cdd_loss_warmup_epochs=args.cdd_loss_warmup_epochs,
        freeze_text_encoder=args.freeze_text_encoder,
        freeze_audio_encoder=args.freeze_audio_encoder,
        freeze_epochs=args.freeze_epochs,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
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
        rdrop_alpha=args.rdrop_alpha,
        augment_train=args.augment_train,
        text_max_length=args.text_max_length,
        audio_sampling_rate=args.audio_sampling_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    report = Enrun(cfg)
    print("\n===== Final Test Report =====")
    for k, v in report.items():
        print(f"  {k}: acc={v['acc']:.4f}  f1={v['f1']:.4f}")


if __name__ == "__main__":
    main()
