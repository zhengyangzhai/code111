"""run_sr.py — SR 14 类意图分类启动入口"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch

from entrain_sr import Enrun_SR, SRConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SR 14-class intent classification")

    parser.add_argument("--sr_root", type=str, default="data/SR")
    parser.add_argument("--pqp_root", type=str, default="data/PQP", help="PQP data root for context modeling")

    parser.add_argument("--text_model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--audio_model_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    parser.add_argument("--feature_extractor_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Model type: "baseline" or "moe"
    parser.add_argument("--model_type", type=str, default="moe", choices=["baseline", "moe"])
    parser.add_argument("--use_pqp", action="store_true", default=True, help="Enable PQP context modeling")
    parser.add_argument("--no_use_pqp", dest="use_pqp", action="store_false", help="Disable PQP context modeling")
    parser.add_argument("--use_moe", action="store_true", default=True, help="Enable MOE classifier")
    parser.add_argument("--no_use_moe", dest="use_moe", action="store_false", help="Disable MOE classifier")

    parser.add_argument("--modality", type=str, default="multimodal", choices=["multimodal", "text_only", "audio_only", "pbcf", "pbcf_no_cross_attn", "pbcf_no_discrepancy", "drbf", "cdd", "mult", "misa"], help="Ablation: multimodal | text_only | audio_only | pbcf | ... | cdd | mult | misa")
    parser.add_argument("--no_prosody", dest="use_prosody", action="store_false", help="Ablation: disable prosody branch")
    parser.add_argument("--no_frame_acoustic", dest="use_frame_acoustic", action="store_false", help="Ablation: disable frame acoustic branch")
    parser.add_argument("--use_hierarchical", action="store_true", default=False, help="Hierarchical SR: add layer prediction head and auxiliary loss")
    parser.add_argument("--layer_loss_weight", type=float, default=0.3, help="Weight for layer auxiliary loss when use_hierarchical")
    parser.add_argument("--no_token_disc", action="store_true", help="CDD-Net ablation: use sentence-level discrepancy instead of token-level (TLDL)")
    parser.add_argument("--no_dual_contrastive", action="store_true", help="CDD-Net ablation: disable dual-space contrastive losses (align + sep)")
    parser.add_argument("--lambda_con", type=float, default=0.01, help="CDD contrastive loss weight")
    parser.add_argument("--lambda_orth", type=float, default=0.001, help="CDD orthogonality loss weight")
    parser.add_argument("--lambda_align", type=float, default=0.005, help="CDD alignment loss weight")
    parser.add_argument("--lambda_sep", type=float, default=0.001, help="CDD separation loss weight")
    parser.add_argument("--lambda_recon", type=float, default=0.01, help="CDR reconstruction loss weight; set 0 to disable CDR")
    parser.add_argument("--no_dgcp", action="store_true", help="Disable DGCP: use standard supcon instead of discrepancy-guided contrastive pairs")
    parser.add_argument("--cdd_loss_warmup_epochs", type=int, default=5, help="Linearly warmup CDD auxiliary losses over this many epochs")
    parser.add_argument("--no_rfr", action="store_true", help="CDD-Net ablation: disable raw fusion residual and use CDD fused branch only")
    parser.add_argument("--rfr_gate_tau", type=float, default=1.0, help="RFR gate temperature; >1 softens gate, prevents early polarization")
    parser.add_argument("--rfr_beta_init", type=float, default=1.0, help="RFR branch initial scaling factor (learnable); <1 suppresses raw path early on")

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
    parser.add_argument("--exp_name", type=str, default="sr_14class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None, help="Speaker split seed; defaults to --seed when omitted")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None])

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SRConfig(
        sr_root=args.sr_root,
        pqp_root=args.pqp_root,
        text_model_name=args.text_model_name,
        audio_model_name=args.audio_model_name,
        feature_extractor_name=args.feature_extractor_name,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        model_type=args.model_type,
        use_pqp=args.use_pqp,
        use_moe=args.use_moe,
        modality=args.modality,
        use_prosody=args.use_prosody,
        use_frame_acoustic=args.use_frame_acoustic,
        use_hierarchical=args.use_hierarchical,
        layer_loss_weight=args.layer_loss_weight,
        no_token_disc=args.no_token_disc,
        no_dual_contrastive=args.no_dual_contrastive,
        lambda_con=args.lambda_con,
        lambda_orth=args.lambda_orth,
        lambda_align=args.lambda_align,
        lambda_sep=args.lambda_sep,
        lambda_recon=args.lambda_recon,
        use_dgcp=not args.no_dgcp,
        cdd_loss_warmup_epochs=args.cdd_loss_warmup_epochs,
        use_rfr=not args.no_rfr,
        rfr_gate_tau=args.rfr_gate_tau,
        rfr_beta_init=args.rfr_beta_init,
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
        device=args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    report = Enrun_SR(cfg)

    print("\n===== Final SR Test Report =====")
    for key, metrics in report.items():
        if isinstance(metrics, dict) and "acc" in metrics:
            line = "  %s: acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f" % (key, metrics["acc"], metrics["macro_f1"], metrics["weighted_f1"])
            if "layer_prediction_acc" in metrics:
                line += "  layer_acc=%.4f" % metrics["layer_prediction_acc"]
            print(line)


if __name__ == "__main__":
    main()
