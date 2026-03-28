"""run_contradiction.py — IntentContradictionNet 命令行入口"""

import os

# 修复 OpenMP 重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch

from entrain_contradiction import ContradictionConfig, Enrun


def parse_args():
    p = argparse.ArgumentParser(
        description="IntentContradictionNet — 多模态矛盾网络训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 数据 ----
    g = p.add_argument_group("Data")
    g.add_argument("--data_root", type=str, default="data/PQP")
    g.add_argument("--split_root", type=str, default="data/PQP/in-scope")
    g.add_argument("--train_file", type=str, default="train.tsv")
    g.add_argument("--dev_file", type=str, default="dev.tsv")
    g.add_argument("--test_file", type=str, default="test.tsv")

    # ---- 模型 ----
    g = p.add_argument_group("Model")
    g.add_argument("--text_model_name", type=str, default="bert-base-chinese")
    g.add_argument("--audio_model_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    g.add_argument("--feature_extractor_name", type=str, default="TencentGameMate/chinese-wav2vec2-base")
    g.add_argument("--proj_dim", type=int, default=128,
                   help="共享投影维度")
    g.add_argument("--rank", type=int, default=32,
                   help="低秩瓶颈维度 (建议 proj_dim // 4)")
    g.add_argument("--num_experts", type=int, default=3)
    g.add_argument("--kernel_sizes", type=str, default="1,3,5",
                   help="多尺度卷积核 (逗号分隔)")
    g.add_argument("--sinkhorn_iters", type=int, default=8)
    g.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
    g.add_argument("--supcon_temperature", type=float, default=0.07)
    g.add_argument("--dropout", type=float, default=0.2)

    # ---- 损失权重 ----
    g = p.add_argument_group("Loss Weights")
    g.add_argument("--lambda_cls", type=float, default=1.0)
    g.add_argument("--lambda_aux", type=float, default=0.05)
    g.add_argument("--lambda_ortho", type=float, default=0.01)
    g.add_argument("--lambda_supcon", type=float, default=0.1)
    g.add_argument("--label_smoothing", type=float, default=0.1,
                   help="标签平滑系数")

    # ---- SpeechCraft ----
    g = p.add_argument_group("SpeechCraft")
    g.add_argument("--sc_labels_path", type=str, default="data/PQP/sc_labels.json",
                   help="SpeechCraft 类别标签 JSON 路径")

    # ---- 冻结策略 ----
    g = p.add_argument_group("Freeze Strategy")
    g.add_argument("--freeze_text_encoder", action="store_true", default=True)
    g.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false")
    g.add_argument("--freeze_audio_encoder", action="store_true", default=True)
    g.add_argument("--no_freeze_audio_encoder", dest="freeze_audio_encoder", action="store_false")
    g.add_argument("--freeze_epochs", type=int, default=5,
                   help="前 N 个 epoch 冻结编码器 (头部预热)")

    # ---- 优化 ----
    g = p.add_argument_group("Optimization")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--batch_size", type=int, default=4)
    g.add_argument("--gradient_accumulation_steps", type=int, default=4,
                   help="梯度累积步数 (有效 batch = batch_size * 此值)")
    g.add_argument("--encoder_lr", type=float, default=1e-5,
                   help="编码器学习率 (解冻后)")
    g.add_argument("--classifier_lr", type=float, default=1e-4,
                   help="分类头 + OT/MoE 等模块学习率")
    g.add_argument("--weight_decay", type=float, default=1e-2)
    g.add_argument("--warmup_ratio", type=float, default=0.1)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--patience", type=int, default=12,
                   help="Early stopping 容忍轮数")
    g.add_argument("--use_cosine_schedule", action="store_true", default=True)
    g.add_argument("--no_cosine_schedule", dest="use_cosine_schedule", action="store_false")

    # ---- 数据增强 ----
    g = p.add_argument_group("Augmentation")
    g.add_argument("--augment_train", action="store_true", default=True)
    g.add_argument("--no_augment_train", dest="augment_train", action="store_false")

    # ---- 数据处理 ----
    g = p.add_argument_group("Data Processing")
    g.add_argument("--text_max_length", type=int, default=64)
    g.add_argument("--audio_sampling_rate", type=int, default=16000)
    g.add_argument("--num_workers", type=int, default=0)

    # ---- 输出 ----
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", type=str, default="output")
    g.add_argument("--exp_name", type=str, default="contradiction_v1")

    # ---- 其他 ----
    g = p.add_argument_group("Other")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None])

    return p.parse_args()


def main():
    args = parse_args()
    cfg = ContradictionConfig(
        data_root=args.data_root,
        split_root=args.split_root,
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        text_model_name=args.text_model_name,
        audio_model_name=args.audio_model_name,
        feature_extractor_name=args.feature_extractor_name,
        proj_dim=args.proj_dim,
        rank=args.rank,
        num_experts=args.num_experts,
        kernel_sizes=args.kernel_sizes,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        supcon_temperature=args.supcon_temperature,
        dropout=args.dropout,
        lambda_cls=args.lambda_cls,
        lambda_aux=args.lambda_aux,
        lambda_ortho=args.lambda_ortho,
        lambda_supcon=args.lambda_supcon,
        label_smoothing=args.label_smoothing,
        sc_labels_path=args.sc_labels_path,
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
        patience=args.patience,
        use_cosine_schedule=args.use_cosine_schedule,
        augment_train=args.augment_train,
        text_max_length=args.text_max_length,
        audio_sampling_rate=args.audio_sampling_rate,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    report = Enrun(cfg)
    print("\n===== Final Test Report =====")
    for k, v in report.items():
        print(f"  {k}: acc={v['acc']:.4f}  f1={v['f1']:.4f}")


if __name__ == "__main__":
    main()
