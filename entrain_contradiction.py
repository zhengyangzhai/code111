"""entrain_contradiction.py — IntentContradictionNet 训练入口

适配 model_contradiction.py 中的 IntentContradictionNet 模型。
核心特性:
1. OT 软对齐 → 矛盾路由 MoE → 多任务 + 对比学习
2. 四项联合损失: L_cls + L_aux + L_ortho + L_supcon
3. 完善的冻结 → 部分解冻策略
4. Cosine Annealing with Warmup
"""

import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import (
    build_audio_index,
    build_pitchtier_index,
    create_dataloader,
)
from MetricsTop import MetricsTop
from model_contradiction import IntentContradictionNet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ContradictionConfig:
    # ---- 数据 ----
    data_root: str = "data/PQP"
    split_root: str = "data/PQP/in-scope"
    train_file: str = "train.tsv"
    dev_file: str = "dev.tsv"
    test_file: str = "test.tsv"

    # ---- 模型 ----
    text_model_name: str = "bert-base-chinese"
    audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base"
    feature_extractor_name: str = "TencentGameMate/chinese-wav2vec2-base"
    proj_dim: int = 128                 # 共享投影维度
    rank: int = 32                      # 低秩瓶颈维度 (proj_dim // 4)
    num_experts: int = 3                # MoE 专家数
    kernel_sizes: str = "1,3,5"         # 多尺度卷积核 (逗号分隔)
    sinkhorn_iters: int = 8             # Sinkhorn 迭代次数
    sinkhorn_epsilon: float = 0.1       # 熵正则化系数
    supcon_temperature: float = 0.07    # SupCon 温度
    dropout: float = 0.2               # 全局 Dropout

    # ---- 损失权重 ----
    lambda_cls: float = 1.0
    lambda_aux: float = 0.05            # 未提供 aux_labels 时此项自动为 0
    lambda_ortho: float = 0.01
    lambda_supcon: float = 0.1
    label_smoothing: float = 0.1

    # ---- SpeechCraft ----
    sc_labels_path: str = "data/PQP/sc_labels.json"

    # ---- 冻结策略 ----
    freeze_text_encoder: bool = True
    freeze_audio_encoder: bool = True
    freeze_epochs: int = 5              # 前 N 个 epoch 冻结编码器做头部预热

    # ---- 优化 ----
    epochs: int = 50
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 有效 batch = 16
    encoder_lr: float = 1e-5             # 编码器 (解冻后)
    classifier_lr: float = 1e-4          # 分类头 + 其他模块
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 12
    use_cosine_schedule: bool = True

    # ---- 数据增强 ----
    augment_train: bool = True

    # ---- 数据处理 ----
    text_max_length: int = 64
    audio_sampling_rate: int = 16000
    num_workers: int = 0

    # ---- 输出 ----
    output_dir: str = "output"
    exp_name: str = "contradiction_v1"

    # ---- 其他 ----
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def _get_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("Contradiction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Cosine Annealing with Warmup
# ---------------------------------------------------------------------------
class CosineAnnealingWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ContradictionTrainer:
    def __init__(self, cfg: ContradictionConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._set_seed(cfg.seed)

        # ---- 目录 ----
        os.makedirs(cfg.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.output_dir, f"{cfg.exp_name}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.logger = _get_logger(self.run_dir)
        self.logger.info(f"Run directory: {self.run_dir}")

        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        # ---- 索引 ----
        self.audio_index = build_audio_index(cfg.data_root)
        self.pitchtier_index = build_pitchtier_index(cfg.data_root)
        self.logger.info(
            f"Audio index: {len(self.audio_index)} files | "
            f"PitchTier index: {len(self.pitchtier_index)} files"
        )

        # ---- SpeechCraft 标签 ----
        self.sc_labels = None
        if cfg.sc_labels_path and os.path.exists(cfg.sc_labels_path):
            with open(cfg.sc_labels_path, "r", encoding="utf-8") as f:
                self.sc_labels = json.load(f)
            self.logger.info(f"SpeechCraft labels loaded: {len(self.sc_labels)} entries")
        else:
            self.logger.info("SpeechCraft labels not found, proceeding without them")

        # ---- 数据 ----
        self.train_loader = self._build_loader(cfg.train_file, shuffle=True, augment=cfg.augment_train)
        self.dev_loader = self._build_loader(cfg.dev_file, shuffle=False, augment=False)
        self.test_loader = self._build_loader(cfg.test_file, shuffle=False, augment=False)
        self.logger.info(
            f"Data loaded — train: {len(self.train_loader.dataset)}, "
            f"dev: {len(self.dev_loader.dataset)}, "
            f"test: {len(self.test_loader.dataset)}"
        )

        # ---- 解析 kernel_sizes ----
        kernel_sizes = [int(k) for k in cfg.kernel_sizes.split(",")]

        # ---- 模型 ----
        self.model = IntentContradictionNet(
            text_model_name=cfg.text_model_name,
            audio_model_name=cfg.audio_model_name,
            num_labels=2,
            proj_dim=cfg.proj_dim,
            rank=cfg.rank,
            num_experts=cfg.num_experts,
            kernel_sizes=kernel_sizes,
            sinkhorn_iters=cfg.sinkhorn_iters,
            sinkhorn_epsilon=cfg.sinkhorn_epsilon,
            supcon_temperature=cfg.supcon_temperature,
            dropout=cfg.dropout,
            label_smoothing=cfg.label_smoothing,
            lambda_cls=cfg.lambda_cls,
            lambda_aux=cfg.lambda_aux,
            lambda_ortho=cfg.lambda_ortho,
            lambda_supcon=cfg.lambda_supcon,
            freeze_text_encoder=cfg.freeze_text_encoder,
            freeze_audio_encoder=cfg.freeze_audio_encoder,
        ).to(self.device)

        # ---- 参数统计 ----
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(
            f"Model created — total_params={total_params:,}, trainable_params={trainable_params:,}, "
            f"proj_dim={cfg.proj_dim}, rank={cfg.rank}, kernels={kernel_sizes}"
        )

        # ---- 优化器 ----
        self.optimizer = self._build_optimizer()
        total_steps = (len(self.train_loader) // cfg.gradient_accumulation_steps) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        if cfg.use_cosine_schedule:
            self.scheduler = CosineAnnealingWarmup(
                self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
            )
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
            )

        self.logger.info(
            f"Optimizer — total_steps={total_steps}, warmup={warmup_steps}, "
            f"schedule={'cosine' if cfg.use_cosine_schedule else 'linear'}, "
            f"freeze_epochs={cfg.freeze_epochs}"
        )

        self.best_scores: Dict[str, float] = {"acc": -1.0, "f1": -1.0}
        self.history: List[Dict] = []

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self) -> AdamW:
        cfg = self.cfg
        encoder_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "audio_encoder" in name or "text_encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": cfg.encoder_lr, "weight_decay": cfg.weight_decay})
        groups.append({"params": head_params, "lr": cfg.classifier_lr, "weight_decay": cfg.weight_decay})
        return AdamW(groups)

    def _rebuild_optimizer_after_unfreeze(self):
        self.optimizer = self._build_optimizer()
        remaining_epochs = self.cfg.epochs - self.cfg.freeze_epochs
        total_steps = (len(self.train_loader) // self.cfg.gradient_accumulation_steps) * remaining_epochs
        warmup_steps = int(total_steps * 0.05)

        if self.cfg.use_cosine_schedule:
            self.scheduler = CosineAnnealingWarmup(
                self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
            )
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
            )
        self.logger.info(f"Optimizer rebuilt after unfreeze — remaining_steps={total_steps}, warmup={warmup_steps}")

    def _build_loader(self, split_file: str, shuffle: bool, augment: bool = False) -> DataLoader:
        tsv_path = os.path.join(self.cfg.split_root, split_file)
        return create_dataloader(
            tsv_path=tsv_path,
            audio_root=self.cfg.data_root,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            tokenizer_name=self.cfg.text_model_name,
            feature_extractor_name=self.cfg.feature_extractor_name,
            text_max_length=self.cfg.text_max_length,
            audio_sampling_rate=self.cfg.audio_sampling_rate,
            audio_index=self.audio_index,
            pitchtier_index=self.pitchtier_index,
            sc_labels=self.sc_labels,
            augment=augment,
        )

    def _move_batch(self, batch: Dict) -> Dict:
        return {
            "text_input_ids": batch["text_input_ids"].to(self.device),
            "text_attention_mask": batch["text_attention_mask"].to(self.device),
            "audio_input_values": batch["audio_input_values"].to(self.device),
            "audio_attention_mask": batch["audio_attention_mask"].to(self.device),
            "prosody_features": batch["prosody_features"].to(self.device),
            "speechcraft_features": batch["speechcraft_features"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }

    # ------------------------------------------------------------------ #
    #  训练
    # ------------------------------------------------------------------ #
    def do_train(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        losses: List[float] = []
        losses_cls: List[float] = []
        losses_ortho: List[float] = []
        losses_supcon: List[float] = []
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        accum = self.cfg.gradient_accumulation_steps
        running_correct = 0
        running_total = 0

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}/{self.cfg.epochs}",
            ncols=130,
        )
        for step, batch in enumerate(pbar, start=1):
            batch_t = self._move_batch(batch)
            out = self.model(**batch_t)
            loss = out["loss"] / accum
            loss.backward()

            losses.append(out["loss"].item())
            losses_cls.append(out["loss_cls"].item())
            losses_ortho.append(out["loss_ortho"].item())
            losses_supcon.append(out["loss_supcon"].item())
            all_logits.append(out["logits"].detach().cpu())
            all_labels.append(batch_t["labels"].detach().cpu())

            preds = out["logits"].argmax(dim=-1)
            running_correct += (preds == batch_t["labels"]).sum().item()
            running_total += batch_t["labels"].numel()
            running_acc = running_correct / max(running_total, 1)

            if step % accum == 0 or step == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(
                loss=f"{losses[-1]:.4f}",
                cls=f"{losses_cls[-1]:.4f}",
                acc=f"{running_acc:.4f}",
            )

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = MetricsTop.binary_acc_f1(logits, labels)
        metrics["loss"] = float(np.mean(losses))
        metrics["loss_cls"] = float(np.mean(losses_cls))
        metrics["loss_ortho"] = float(np.mean(losses_ortho))
        metrics["loss_supcon"] = float(np.mean(losses_supcon))

        self.logger.info(
            f"[Train] Epoch {epoch}/{self.cfg.epochs}  "
            f"loss={metrics['loss']:.4f}  cls={metrics['loss_cls']:.4f}  "
            f"ortho={metrics['loss_ortho']:.4f}  supcon={metrics['loss_supcon']:.4f}  "
            f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
        )
        print(
            f"  >> Train Epoch {epoch}: loss={metrics['loss']:.4f}  "
            f"ACC={metrics['acc']:.4f}  F1={metrics['f1']:.4f}"
        )
        self._log_json(f"train_epoch_{epoch}.json", metrics)
        return metrics

    # ------------------------------------------------------------------ #
    #  验证 / 测试
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def do_test(self, split: str = "dev") -> Dict[str, float]:
        assert split in {"dev", "test"}
        loader = self.dev_loader if split == "dev" else self.test_loader
        label_name = "Eval(dev)" if split == "dev" else "Test"

        self.model.eval()
        losses: List[float] = []
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        running_correct = 0
        running_total = 0

        pbar = tqdm(loader, desc=f"{label_name}", ncols=120)
        for batch in pbar:
            batch_t = self._move_batch(batch)
            out = self.model(**batch_t)
            losses.append(out["loss"].item())
            all_logits.append(out["logits"].detach().cpu())
            all_labels.append(batch_t["labels"].detach().cpu())

            preds = out["logits"].argmax(dim=-1)
            running_correct += (preds == batch_t["labels"]).sum().item()
            running_total += batch_t["labels"].numel()
            pbar.set_postfix(
                loss=f"{losses[-1]:.4f}",
                acc=f"{running_correct / max(running_total, 1):.4f}",
            )

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = MetricsTop.binary_acc_f1(logits, labels)
        metrics["loss"] = float(np.mean(losses))

        self.logger.info(
            f"[{split.upper()}]  loss={metrics['loss']:.4f}  "
            f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  "
            f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}"
        )
        print(
            f"  >> {label_name}: loss={metrics['loss']:.4f}  "
            f"ACC={metrics['acc']:.4f}  F1={metrics['f1']:.4f}  "
            f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}"
        )
        self._log_json(f"{split}_metrics.json", metrics)
        return metrics

    # ------------------------------------------------------------------ #
    #  检查点
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, name)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Checkpoint saved: {name}")

    def load_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, name)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.logger.info(f"Checkpoint loaded: {name}")

    def _log_json(self, filename: str, payload: Dict):
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 启动函数
# ---------------------------------------------------------------------------
def Enrun(cfg: ContradictionConfig):
    trainer = ContradictionTrainer(cfg)
    no_improve_count = 0

    for epoch in range(1, cfg.epochs + 1):
        # ---- 冻结 → 解冻 ----
        if epoch == cfg.freeze_epochs + 1 and cfg.freeze_epochs > 0:
            trainer.logger.info(
                f"=== Epoch {epoch}: Partially unfreezing audio encoder (top 4 layers) ==="
            )
            trainer.model.set_freeze(freeze_text=True, freeze_audio=False)
            trainer._rebuild_optimizer_after_unfreeze()
            no_improve_count = 0

        # ---- 清理 GPU 缓存 ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- 训练 & 验证 ----
        train_metrics = trainer.do_train(epoch)
        dev_metrics = trainer.do_test("dev")

        # ---- 保存最优 ----
        improved = []
        if dev_metrics["acc"] > trainer.best_scores["acc"]:
            trainer.best_scores["acc"] = dev_metrics["acc"]
            trainer.save_checkpoint("best_acc.pt")
            improved.append("acc")
        if dev_metrics["f1"] > trainer.best_scores["f1"]:
            trainer.best_scores["f1"] = dev_metrics["f1"]
            trainer.save_checkpoint("best_f1.pt")
            improved.append("f1")

        # ---- early stopping ----
        if "acc" in improved or "f1" in improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= cfg.patience and epoch > cfg.freeze_epochs:
            trainer.logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {cfg.patience} epochs)"
            )
            print(f"  >> Early stopping at epoch {epoch}")
            break

        epoch_log = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
            "best_scores": trainer.best_scores,
            "improved": improved,
        }
        trainer._log_json(f"epoch_{epoch}_summary.json", epoch_log)
        trainer.history.append(epoch_log)

        trainer.logger.info(
            f"--- Epoch {epoch} done | best_acc={trainer.best_scores['acc']:.4f} "
            f"best_f1={trainer.best_scores['f1']:.4f} "
            f"{'★ improved: ' + ','.join(improved) if improved else ''}"
            f" | patience={no_improve_count}/{cfg.patience}"
        )

    # ---- 最终测试 ----
    trainer.logger.info("===== Final Test Evaluation =====")
    final_report: Dict = {}
    for key in ["acc", "f1"]:
        ckpt = f"best_{key}.pt"
        trainer.load_checkpoint(ckpt)
        test_metrics = trainer.do_test("test")
        final_report[f"test_with_best_{key}"] = test_metrics
        trainer.logger.info(
            f"Test (best_{key}): acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}"
        )

    trainer._log_json("final_test_report.json", final_report)
    trainer._log_json("full_history.json", trainer.history)
    return final_report
