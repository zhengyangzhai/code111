"""entrain_sr_ccmt.py — Standalone trainer for CCMT SR baseline

Reuses SR training logic (Focal, R-Drop, Mixup, SWA) without touching CDD/main SR code.
Binds only to CCMTSRModel; model returns logits + loss only (no CDD-specific losses).
"""

import gc
import json
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from model_sr_ccmt import CCMTSRModel
from sr_dataloader import (
    SR_ID2LABEL,
    SR_NUM_LABELS,
    SR_VALID_LABELS,
    build_sr_samples,
    create_sr_dataloader,
    split_sr_samples,
)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha[targets]
        return (focal_weight * ce).mean()


def soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    return -(soft_labels * log_p).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# R-Drop
# ---------------------------------------------------------------------------
def compute_rdrop_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    p_soft = F.softmax(logits1, dim=-1)
    q_soft = F.softmax(logits2, dim=-1)
    p_log = F.log_softmax(logits1, dim=-1)
    q_log = F.log_softmax(logits2, dim=-1)
    return (F.kl_div(q_log, p_soft, reduction="batchmean") + F.kl_div(p_log, q_soft, reduction="batchmean")) / 2


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict:
    preds = logits.argmax(dim=-1)
    total = labels.numel()
    correct = (preds == labels).sum().item()
    acc = correct / max(total, 1)

    per_class = {}
    f1_list = []
    for c in range(SR_NUM_LABELS):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        support = (labels == c).sum().item()
        per_class[SR_ID2LABEL[c]] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    supports = np.array([per_class[SR_ID2LABEL[c]]["support"] for c in range(SR_NUM_LABELS)])
    f1_arr = np.array(f1_list)
    weighted_f1 = float((f1_arr * supports).sum() / max(supports.sum(), 1))

    from sr_dataloader import SR_LAYER_MAP
    layer_acc = {}
    for layer_name in ["factual", "attitude", "emotion", "commitment", "continuation"]:
        class_ids = [c for c in range(SR_NUM_LABELS) if SR_LAYER_MAP.get(SR_ID2LABEL[c]) == layer_name]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for cid in class_ids:
            mask |= (labels == cid)
        if mask.any():
            layer_correct = (preds[mask] == labels[mask]).sum().item()
            layer_total = mask.sum().item()
            layer_acc[layer_name] = layer_correct / max(layer_total, 1)

    return {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_class": per_class, "layer_acc": layer_acc}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CCMTConfig:
    sr_root: str = "data/SR"

    text_model_name: str = "bert-base-chinese"
    audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base"
    feature_extractor_name: str = "TencentGameMate/chinese-wav2vec2-base"
    proj_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.3

    freeze_text_encoder: bool = True
    freeze_audio_encoder: bool = True
    freeze_epochs: int = 5

    epochs: int = 50
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    encoder_lr: float = 1e-5
    classifier_lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    patience: int = 15
    use_cosine_schedule: bool = True

    focal_gamma: float = 2.0
    rdrop_alpha: float = 1.0
    mixup_alpha: float = 0.3
    swa_start_epoch: int = 35
    use_weighted_sampler: bool = False

    augment_train: bool = True
    text_max_length: int = 128
    audio_sampling_rate: int = 16000
    num_workers: int = 0

    output_dir: str = "output"
    exp_name: str = "sr_ccmt_seed42"
    seed: int = 42
    split_seed: Optional[int] = None
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def _get_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("SR_CCMT")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Cosine Annealing Warmup
# ---------------------------------------------------------------------------
import math


class CosineAnnealingWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SRTrainerCCMT:
    def __init__(self, cfg: CCMTConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._set_seed(cfg.seed)

        os.makedirs(cfg.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.output_dir, "%s_%s" % (cfg.exp_name, ts))
        os.makedirs(self.run_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.logger = _get_logger(self.run_dir)
        self.logger.info("Run directory: %s" % self.run_dir)

        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        # ---- Data ----
        self.logger.info("Building SR samples from %s ..." % cfg.sr_root)
        all_samples = build_sr_samples(cfg.sr_root)
        self.logger.info("Total SR utterances: %d" % len(all_samples))

        split_seed = cfg.seed if cfg.split_seed is None else cfg.split_seed
        self.logger.info("Using split_seed=%d and train_seed=%d" % (split_seed, cfg.seed))
        train_samples, dev_samples, test_samples = split_sr_samples(all_samples, seed=split_seed)
        label_counts = Counter(s["label_id"] for s in train_samples)
        self.logger.info("Train label distribution:")
        for lid in range(SR_NUM_LABELS):
            self.logger.info("  %s: %d" % (SR_ID2LABEL[lid], label_counts.get(lid, 0)))

        self.train_loader = create_sr_dataloader(
            train_samples, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            tokenizer_name=cfg.text_model_name, feature_extractor_name=cfg.feature_extractor_name,
            text_max_length=cfg.text_max_length, audio_sampling_rate=cfg.audio_sampling_rate,
            augment=cfg.augment_train, use_weighted_sampler=cfg.use_weighted_sampler,
            pqp_root=None,
        )
        self.dev_loader = create_sr_dataloader(
            dev_samples, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
            tokenizer_name=cfg.text_model_name, feature_extractor_name=cfg.feature_extractor_name,
            text_max_length=cfg.text_max_length, audio_sampling_rate=cfg.audio_sampling_rate,
            pqp_root=None,
        )
        self.test_loader = create_sr_dataloader(
            test_samples, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
            tokenizer_name=cfg.text_model_name, feature_extractor_name=cfg.feature_extractor_name,
            text_max_length=cfg.text_max_length, audio_sampling_rate=cfg.audio_sampling_rate,
            pqp_root=None,
        )

        # ---- Model ----
        self.logger.info("Using CCMTSRModel (cascaded cross-modal transformer)")
        self.model = CCMTSRModel(
            text_model_name=cfg.text_model_name,
            audio_model_name=cfg.audio_model_name,
            num_labels=SR_NUM_LABELS,
            proj_dim=cfg.proj_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            label_smoothing=0.0,
            freeze_text_encoder=cfg.freeze_text_encoder,
            freeze_audio_encoder=cfg.freeze_audio_encoder,
        ).to(self.device)

        n_total = len(train_samples)
        alpha = torch.tensor(
            [n_total / (SR_NUM_LABELS * max(label_counts.get(i, 1), 1)) for i in range(SR_NUM_LABELS)],
            dtype=torch.float32,
        )
        alpha = alpha / alpha.mean()
        alpha = torch.sqrt(alpha).clamp(max=3.0)
        alpha = alpha / alpha.mean()
        alpha = alpha.to(self.device)
        self.model.loss_fn = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Model: total=%d, trainable=%d" % (total_params, trainable_params))

        # ---- Optimizer ----
        self.optimizer = self._build_optimizer()
        total_steps = (len(self.train_loader) // cfg.gradient_accumulation_steps) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        if cfg.use_cosine_schedule:
            self.scheduler = CosineAnnealingWarmup(self.optimizer, warmup_steps, total_steps)
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        self.best_scores: Dict[str, float] = {"acc": -1.0, "macro_f1": -1.0}
        self.history: List[Dict] = []
        self.swa_state: Optional[Dict] = None
        self.swa_count: int = 0

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self) -> AdamW:
        cfg = self.cfg
        encoder_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "audio_encoder" in name:
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
        remaining = self.cfg.epochs - self.cfg.freeze_epochs
        total_steps = (len(self.train_loader) // self.cfg.gradient_accumulation_steps) * remaining
        warmup_steps = int(total_steps * 0.05)
        if self.cfg.use_cosine_schedule:
            self.scheduler = CosineAnnealingWarmup(self.optimizer, warmup_steps, total_steps)
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

    def _move_batch(self, batch: Dict) -> Dict:
        return {
            "text_input_ids": batch["text_input_ids"].to(self.device),
            "text_attention_mask": batch["text_attention_mask"].to(self.device),
            "audio_input_values": batch["audio_input_values"].to(self.device),
            "audio_attention_mask": batch["audio_attention_mask"].to(self.device),
            "prosody_features": batch["prosody_features"].to(self.device),
            "textgrid_features": batch["textgrid_features"].to(self.device),
            "speechcraft_features": batch["speechcraft_features"].to(self.device),
            "frame_acoustic_features": batch["frame_acoustic_features"].to(self.device),
            "frame_acoustic_mask": batch["frame_acoustic_mask"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }

    def _apply_mixup(self, batch_t: Dict, lam: float, idx: torch.Tensor, num_classes: int) -> tuple:
        mixed = dict(batch_t)
        mixed["prosody_features"] = lam * batch_t["prosody_features"] + (1 - lam) * batch_t["prosody_features"][idx]
        mixed["frame_acoustic_features"] = lam * batch_t["frame_acoustic_features"] + (1 - lam) * batch_t["frame_acoustic_features"][idx]
        one_hot = F.one_hot(batch_t["labels"], num_classes=num_classes).float()
        one_hot_mix = F.one_hot(batch_t["labels"][idx], num_classes=num_classes).float()
        soft_labels = lam * one_hot + (1 - lam) * one_hot_mix
        mixed["labels"] = None
        return mixed, soft_labels

    def _update_swa(self):
        state = self.model.state_dict()
        if self.swa_state is None:
            self.swa_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k, v in state.items():
                self.swa_state[k] += v.float()
        self.swa_count += 1

    def _apply_swa(self):
        if self.swa_state is None or self.swa_count == 0:
            return
        avg = {k: (v / self.swa_count) for k, v in self.swa_state.items()}
        self.model.load_state_dict(avg)
        self.logger.info("SWA applied: averaged %d checkpoints" % self.swa_count)

    def do_train(self, epoch: int) -> Dict:
        self.model.train()
        losses, all_logits, all_labels = [], [], []
        accum = self.cfg.gradient_accumulation_steps
        rdrop_warmup_epoch = self.cfg.freeze_epochs + 5
        rdrop_alpha = self.cfg.rdrop_alpha if epoch >= rdrop_warmup_epoch else 0.0
        mixup_alpha = self.cfg.mixup_alpha
        use_mixup = mixup_alpha > 0 and epoch > self.cfg.freeze_epochs
        running_correct, running_total = 0, 0

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc="Train %d/%d" % (epoch, self.cfg.epochs), ncols=120)
        for step, batch in enumerate(pbar, start=1):
            batch_t = self._move_batch(batch)
            use_mixup_step = use_mixup and random.random() < 0.5

            if use_mixup_step:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)
                idx = torch.randperm(batch_t["labels"].size(0), device=self.device)
                mixed_batch, soft_labels = self._apply_mixup(batch_t, lam, idx, SR_NUM_LABELS)
                out1 = self.model(**mixed_batch)
                if rdrop_alpha > 0:
                    out2 = self.model(**mixed_batch)
                    ce_loss = (soft_cross_entropy(out1["logits"], soft_labels) + soft_cross_entropy(out2["logits"], soft_labels)) / 2
                else:
                    ce_loss = soft_cross_entropy(out1["logits"], soft_labels)
                acc_labels = batch_t["labels"]
            else:
                out1 = self.model(**batch_t)
                if rdrop_alpha > 0:
                    out2 = self.model(**batch_t)
                    ce_loss = (out1["loss"] + out2["loss"]) / 2
                else:
                    ce_loss = out1["loss"]
                acc_labels = batch_t["labels"]

            total_loss = ce_loss
            if rdrop_alpha > 0:
                total_loss = total_loss + rdrop_alpha * compute_rdrop_kl_loss(out1["logits"], out2["logits"])

            (total_loss / accum).backward()
            losses.append(total_loss.item())
            all_logits.append(out1["logits"].detach().cpu())
            all_labels.append(acc_labels.detach().cpu())
            preds = out1["logits"].argmax(dim=-1)
            running_correct += (preds == acc_labels).sum().item()
            running_total += batch_t["labels"].numel()

            if step % accum == 0 or step == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(loss="%.4f" % losses[-1], acc="%.4f" % (running_correct / max(running_total, 1)))

        logits = torch.cat(all_logits, 0)
        labels = torch.cat(all_labels, 0)
        metrics = compute_multiclass_metrics(logits, labels)
        metrics["loss"] = float(np.mean(losses))
        self.logger.info("[Train] Epoch %d  loss=%.4f  acc=%.4f  macro_f1=%.4f" % (epoch, metrics["loss"], metrics["acc"], metrics["macro_f1"]))
        del all_logits, all_labels, logits, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    @torch.no_grad()
    def do_test(self, split: str = "dev") -> Dict:
        loader = self.dev_loader if split == "dev" else self.test_loader
        self.model.eval()
        losses, all_logits, all_labels = [], [], []
        for batch in tqdm(loader, desc=split.upper(), ncols=120):
            batch_t = self._move_batch(batch)
            out = self.model(**batch_t)
            losses.append(out["loss"].item())
            all_logits.append(out["logits"].detach().cpu())
            all_labels.append(batch_t["labels"].detach().cpu())
        logits = torch.cat(all_logits, 0)
        labels = torch.cat(all_labels, 0)
        metrics = compute_multiclass_metrics(logits, labels)
        metrics["loss"] = float(np.mean(losses))
        self.logger.info("[%s]  loss=%.4f  acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f" % (split.upper(), metrics["loss"], metrics["acc"], metrics["macro_f1"], metrics["weighted_f1"]))
        for lname, lacc in metrics["layer_acc"].items():
            self.logger.info("  Layer %-14s acc=%.4f" % (lname, lacc))
        del all_logits, all_labels, logits, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def save_checkpoint(self, name: str):
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, name))

    def load_checkpoint(self, name: str):
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, name), map_location=self.device))

    def _log_json(self, filename: str, payload):
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def Enrun_SR_CCMT(cfg: CCMTConfig):
    trainer = SRTrainerCCMT(cfg)
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        if epoch == cfg.freeze_epochs + 1 and cfg.freeze_epochs > 0:
            trainer.logger.info("=== Epoch %d: Unfreezing audio encoder top 4 layers ===" % epoch)
            trainer.model.set_freeze(freeze_text=True, freeze_audio=False)
            trainer._rebuild_optimizer_after_unfreeze()
            no_improve = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_m = trainer.do_train(epoch)
        dev_m = trainer.do_test("dev")

        if epoch >= cfg.swa_start_epoch:
            trainer._update_swa()

        improved = []
        if dev_m["acc"] > trainer.best_scores["acc"]:
            trainer.best_scores["acc"] = dev_m["acc"]
            trainer.save_checkpoint("best_acc.pt")
            improved.append("acc")
        if dev_m["macro_f1"] > trainer.best_scores["macro_f1"]:
            trainer.best_scores["macro_f1"] = dev_m["macro_f1"]
            trainer.save_checkpoint("best_macro_f1.pt")
            improved.append("macro_f1")

        if improved:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.patience and epoch > cfg.freeze_epochs:
            trainer.logger.info("Early stopping at epoch %d" % epoch)
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_log = {
            "epoch": epoch,
            "train": {k: v for k, v in train_m.items() if k != "per_class"},
            "dev": {k: v for k, v in dev_m.items() if k != "per_class"},
            "best_scores": dict(trainer.best_scores),
            "improved": improved,
        }
        trainer._log_json("epoch_%d_summary.json" % epoch, epoch_log)
        trainer.history.append(epoch_log)

        trainer.logger.info("--- Epoch %d | best_acc=%.4f  best_macro_f1=%.4f  %s | patience=%d/%d" % (
            epoch, trainer.best_scores["acc"], trainer.best_scores["macro_f1"],
            ("improved: " + ",".join(improved)) if improved else "",
            no_improve, cfg.patience))

    # ---- Final test ----
    trainer.logger.info("===== Final Test =====")
    final_report = {}
    for key in ["acc", "macro_f1"]:
        ckpt = "best_%s.pt" % key
        trainer.load_checkpoint(ckpt)
        test_m = trainer.do_test("test")
        final_report["test_best_%s" % key] = test_m
        trainer.logger.info("Test (best_%s): acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f" % (key, test_m["acc"], test_m["macro_f1"], test_m["weighted_f1"]))
        for label in SR_VALID_LABELS:
            pc = test_m["per_class"][label]
            trainer.logger.info("  %-5s  P=%.3f  R=%.3f  F1=%.3f  support=%d" % (label, pc["precision"], pc["recall"], pc["f1"], pc["support"]))

    if trainer.swa_count > 0:
        trainer._apply_swa()
        trainer.save_checkpoint("swa.pt")
        swa_m = trainer.do_test("test")
        final_report["test_swa"] = swa_m
        trainer.logger.info("Test (SWA, %d epochs): acc=%.4f  macro_f1=%.4f" % (trainer.swa_count, swa_m["acc"], swa_m["macro_f1"]))

    trainer._log_json("final_test_report.json", final_report)
    trainer._log_json("full_history.json", trainer.history)
    return final_report
