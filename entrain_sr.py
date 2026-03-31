"""entrain_sr.py — SR 14 类意图分类训练入口

核心特性:
1. Focal Loss + 逆频率类别加权
2. R-Drop 正则化 + Feature-level Mixup
3. SWA (随机权重平均)
4. 加权过采样平衡稀有类
5. Per-class / Per-layer 分层评估
"""

import gc
import json
import logging
import math
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CDDSRMoEModel, DRBFSRMoEModel, MISASRModel, MulTSRModel, MultiModalPQPModel, PBCFSRMoEModel, SRMoEModel, SR_LABEL_ID_TO_LAYER_ID
from sr_dataloader import (
    SR_ID2LABEL,
    SR_LABEL2ID,
    SR_LAYER_MAP,
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
    """Focal Loss with per-class weights for extreme class imbalance."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = focal_weight * alpha_t

        return (focal_weight * ce).mean()


def soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for soft targets: -sum(soft_labels * log_softmax(logits))."""
    log_p = F.log_softmax(logits, dim=-1)
    return -(soft_labels * log_p).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SRConfig:
    sr_root: str = "data/SR"
    pqp_root: str = "data/PQP"

    text_model_name: str = "bert-base-chinese"
    audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base"
    feature_extractor_name: str = "TencentGameMate/chinese-wav2vec2-base"
    proj_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.3

    # Model type: "baseline" or "moe"
    model_type: str = "moe"
    use_pqp: bool = True
    use_moe: bool = True

    # 消融 / 单模态基线
    modality: str = "multimodal"  # "multimodal" | "text_only" | "audio_only" | "pbcf" | "pbcf_no_cross_attn" | "pbcf_no_discrepancy" | "drbf" | "cdd"
    use_prosody: bool = True
    use_frame_acoustic: bool = True

    # 层级 SR：意图层辅助损失
    use_hierarchical: bool = False
    layer_loss_weight: float = 0.3

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
    lambda_con: float = 0.01
    lambda_orth: float = 0.001
    lambda_align: float = 0.005
    lambda_sep: float = 0.001
    lambda_recon: float = 0.01
    use_dgcp: bool = True
    no_token_disc: bool = False
    no_dual_contrastive: bool = False
    cdd_loss_warmup_epochs: int = 5
    use_rfr: bool = True
    rfr_gate_tau: float = 1.0
    rfr_beta_init: float = 1.0
    use_tone_aware_tldl: bool = False
    tone_mask_gamma: float = 0.5
    tone_mask_temp: float = 1.0
    tone_var_dim: int = 1
    mixup_alpha: float = 0.3
    swa_start_epoch: int = 35
    use_weighted_sampler: bool = False  # FocalLoss已处理类不平衡,不再需要采样器

    augment_train: bool = True
    text_max_length: int = 128
    audio_sampling_rate: int = 16000
    num_workers: int = 0

    output_dir: str = "output"
    exp_name: str = "sr_14class"
    seed: int = 42
    split_seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def _get_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("SR14")
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
# Cosine Annealing with Warmup
# ---------------------------------------------------------------------------
class CosineAnnealingWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# R-Drop KL
# ---------------------------------------------------------------------------
def compute_rdrop_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    p_soft = F.softmax(logits1, dim=-1)
    q_soft = F.softmax(logits2, dim=-1)
    p_log = F.log_softmax(logits1, dim=-1)
    q_log = F.log_softmax(logits2, dim=-1)
    return (F.kl_div(q_log, p_soft, reduction="batchmean")
            + F.kl_div(p_log, q_soft, reduction="batchmean")) / 2


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_multiclass_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict:
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
        per_class[SR_ID2LABEL[c]] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))

    supports = np.array([per_class[SR_ID2LABEL[c]]["support"] for c in range(SR_NUM_LABELS)])
    f1_arr = np.array(f1_list)
    weighted_f1 = float((f1_arr * supports).sum() / max(supports.sum(), 1))

    layer_acc = {}
    for layer_name in ["factual", "attitude", "emotion", "commitment", "continuation"]:
        class_ids = [SR_LABEL2ID[l] for l in SR_VALID_LABELS if SR_LAYER_MAP[l] == layer_name]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for cid in class_ids:
            mask |= (labels == cid)
        if mask.any():
            layer_correct = (preds[mask] == labels[mask]).sum().item()
            layer_total = mask.sum().item()
            layer_acc[layer_name] = layer_correct / max(layer_total, 1)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "layer_acc": layer_acc,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SRTrainer:
    def __init__(self, cfg: SRConfig):
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
        if cfg.use_pqp:
            self.logger.info("PQP context modeling enabled, root: %s" % cfg.pqp_root)
        all_samples = build_sr_samples(cfg.sr_root)
        self.logger.info("Total SR utterances: %d" % len(all_samples))

        split_seed = cfg.seed if cfg.split_seed is None else cfg.split_seed
        self.logger.info("Using split_seed=%d and train_seed=%d" % (split_seed, cfg.seed))
        train_samples, dev_samples, test_samples = split_sr_samples(all_samples, seed=split_seed)
        train_spk = set(s["base_name"].split("_")[1] for s in train_samples)
        dev_spk = set(s["base_name"].split("_")[1] for s in dev_samples)
        test_spk = set(s["base_name"].split("_")[1] for s in test_samples)
        self.logger.info(
            "Split (speaker-level): train=%d (%d spk), dev=%d (%d spk), test=%d (%d spk)"
            % (len(train_samples), len(train_spk), len(dev_samples), len(dev_spk),
               len(test_samples), len(test_spk))
        )
        self.logger.info("  train speakers: %s" % sorted(train_spk))
        self.logger.info("  dev speakers:   %s" % sorted(dev_spk))
        self.logger.info("  test speakers:  %s" % sorted(test_spk))

        label_counts = Counter(s["label_id"] for s in train_samples)
        self.logger.info("Train label distribution:")
        for lid in range(SR_NUM_LABELS):
            self.logger.info("  %s: %d" % (SR_ID2LABEL[lid], label_counts.get(lid, 0)))

        self.train_loader = create_sr_dataloader(
            train_samples, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            tokenizer_name=cfg.text_model_name, feature_extractor_name=cfg.feature_extractor_name,
            text_max_length=cfg.text_max_length, audio_sampling_rate=cfg.audio_sampling_rate,
            augment=cfg.augment_train, use_weighted_sampler=cfg.use_weighted_sampler,
            pqp_root=cfg.pqp_root if cfg.use_pqp else None,
        )
        self.dev_loader = create_sr_dataloader(
            dev_samples, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
            tokenizer_name=cfg.text_model_name, feature_extractor_name=cfg.feature_extractor_name,
            text_max_length=cfg.text_max_length, audio_sampling_rate=cfg.audio_sampling_rate,
            pqp_root=cfg.pqp_root if cfg.use_pqp else None,
        )
        self.test_loader = create_sr_dataloader(
            test_samples, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
            tokenizer_name=cfg.text_model_name, feature_extractor_name=cfg.feature_extractor_name,
            text_max_length=cfg.text_max_length, audio_sampling_rate=cfg.audio_sampling_rate,
            pqp_root=cfg.pqp_root if cfg.use_pqp else None,
        )

        # ---- Model ----
        # NOTE: SpeechCraft branch receives all-zero input for SR (no SC labels available).
        # The gate mechanism should learn to suppress it; architecture kept for PQP compat.
        if cfg.modality == "cdd":
            self.logger.info("Using CDDSRMoEModel (modality=cdd, hierarchical=%s)" % cfg.use_hierarchical)
            self.model = CDDSRMoEModel(
                text_model_name=cfg.text_model_name,
                audio_model_name=cfg.audio_model_name,
                num_labels=SR_NUM_LABELS,
                proj_dim=cfg.proj_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                label_smoothing=0.0,
                freeze_text_encoder=cfg.freeze_text_encoder,
                freeze_audio_encoder=cfg.freeze_audio_encoder,
                use_pqp=cfg.use_pqp,
                use_moe=cfg.use_moe,
                use_prosody=cfg.use_prosody,
                use_frame_acoustic=cfg.use_frame_acoustic,
                use_hierarchical=cfg.use_hierarchical,
                layer_loss_weight=cfg.layer_loss_weight,
                use_token_disc=not cfg.no_token_disc,
                use_dual_contrastive=not cfg.no_dual_contrastive,
                use_dgcp=cfg.use_dgcp,
                use_rfr=cfg.use_rfr,
                rfr_gate_tau=cfg.rfr_gate_tau,
                rfr_beta_init=cfg.rfr_beta_init,
                use_tone_aware_tldl=cfg.use_tone_aware_tldl,
                tone_mask_gamma=cfg.tone_mask_gamma,
                tone_mask_temp=cfg.tone_mask_temp,
                tone_var_dim=cfg.tone_var_dim,
            ).to(self.device)
        elif cfg.modality == "drbf":
            self.logger.info("Using DRBFSRMoEModel (modality=drbf, hierarchical=%s)" % cfg.use_hierarchical)
            self.model = DRBFSRMoEModel(
                text_model_name=cfg.text_model_name,
                audio_model_name=cfg.audio_model_name,
                num_labels=SR_NUM_LABELS,
                proj_dim=cfg.proj_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                label_smoothing=0.0,
                freeze_text_encoder=cfg.freeze_text_encoder,
                freeze_audio_encoder=cfg.freeze_audio_encoder,
                use_pqp=cfg.use_pqp,
                use_moe=cfg.use_moe,
                use_prosody=cfg.use_prosody,
                use_frame_acoustic=cfg.use_frame_acoustic,
                use_hierarchical=cfg.use_hierarchical,
                layer_loss_weight=cfg.layer_loss_weight,
            ).to(self.device)
        elif cfg.modality == "mult":
            self.logger.info("Using MulTSRModel (crossmodal transformer baseline)")
            self.model = MulTSRModel(
                text_model_name=cfg.text_model_name,
                audio_model_name=cfg.audio_model_name,
                num_labels=SR_NUM_LABELS,
                proj_dim=cfg.proj_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                label_smoothing=0.0,
                freeze_text_encoder=cfg.freeze_text_encoder,
                freeze_audio_encoder=cfg.freeze_audio_encoder,
                use_prosody=cfg.use_prosody,
                use_frame_acoustic=cfg.use_frame_acoustic,
            ).to(self.device)
        elif cfg.modality == "misa":
            self.logger.info("Using MISASRModel (modality-invariant/specific baseline)")
            self.model = MISASRModel(
                text_model_name=cfg.text_model_name,
                audio_model_name=cfg.audio_model_name,
                num_labels=SR_NUM_LABELS,
                proj_dim=cfg.proj_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                label_smoothing=0.0,
                freeze_text_encoder=cfg.freeze_text_encoder,
                freeze_audio_encoder=cfg.freeze_audio_encoder,
                use_prosody=cfg.use_prosody,
                use_frame_acoustic=cfg.use_frame_acoustic,
            ).to(self.device)
        elif cfg.modality.startswith("pbcf"):
            self.logger.info("Using PBCFSRMoEModel (modality=%s, hierarchical=%s)" % (cfg.modality, cfg.use_hierarchical))
            self.model = PBCFSRMoEModel(
                text_model_name=cfg.text_model_name,
                audio_model_name=cfg.audio_model_name,
                num_labels=SR_NUM_LABELS,
                proj_dim=cfg.proj_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                label_smoothing=0.0,
                freeze_text_encoder=cfg.freeze_text_encoder,
                freeze_audio_encoder=cfg.freeze_audio_encoder,
                use_pqp=cfg.use_pqp,
                use_moe=cfg.use_moe,
                use_prosody=cfg.use_prosody,
                use_frame_acoustic=cfg.use_frame_acoustic,
                use_hierarchical=cfg.use_hierarchical,
                layer_loss_weight=cfg.layer_loss_weight,
                use_cross_attn=(cfg.modality != "pbcf_no_cross_attn"),
                use_discrepancy=(cfg.modality != "pbcf_no_discrepancy"),
            ).to(self.device)
        elif cfg.model_type == "moe" and cfg.use_moe:
            self.logger.info("Using SRMoEModel with PQP context and MOE classifier (modality=%s, hierarchical=%s)" % (cfg.modality, cfg.use_hierarchical))
            self.model = SRMoEModel(
                text_model_name=cfg.text_model_name,
                audio_model_name=cfg.audio_model_name,
                num_labels=SR_NUM_LABELS,
                proj_dim=cfg.proj_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                label_smoothing=0.0,
                freeze_text_encoder=cfg.freeze_text_encoder,
                freeze_audio_encoder=cfg.freeze_audio_encoder,
                use_pqp=cfg.use_pqp,
                use_moe=cfg.use_moe,
                use_text_only=(cfg.modality == "text_only"),
                use_audio_only=(cfg.modality == "audio_only"),
                use_prosody=cfg.use_prosody,
                use_frame_acoustic=cfg.use_frame_acoustic,
                use_hierarchical=cfg.use_hierarchical,
                layer_loss_weight=cfg.layer_loss_weight,
            ).to(self.device)
        else:
            self.logger.info("Using baseline MultiModalPQPModel (modality=%s)" % cfg.modality)
            self.model = MultiModalPQPModel(
                text_model_name=cfg.text_model_name,
                audio_model_name=cfg.audio_model_name,
                num_labels=SR_NUM_LABELS,
                proj_dim=cfg.proj_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                label_smoothing=0.0,
                freeze_text_encoder=cfg.freeze_text_encoder,
                freeze_audio_encoder=cfg.freeze_audio_encoder,
                use_text_only=(cfg.modality == "text_only"),
                use_audio_only=(cfg.modality == "audio_only"),
                use_prosody=cfg.use_prosody,
                use_frame_acoustic=cfg.use_frame_acoustic,
            ).to(self.device)

        # Replace loss_fn with FocalLoss
        n_total = len(train_samples)
        alpha = torch.tensor(
            [n_total / (SR_NUM_LABELS * max(label_counts.get(i, 1), 1)) for i in range(SR_NUM_LABELS)],
            dtype=torch.float32,
        )
        alpha = alpha / alpha.mean()  # 归一化使均值=1
        # 压缩极端值: 使用sqrt压缩 + 上限裁剪, 避免与weighted sampler双重补偿
        alpha = torch.sqrt(alpha)
        alpha = alpha.clamp(max=3.0)
        alpha = alpha / alpha.mean()  # 再次归一化
        alpha = alpha.to(self.device)
        self.logger.info("FocalLoss alpha (normalized): %s" % alpha.tolist())
        self.model.loss_fn = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma,
                                       label_smoothing=cfg.label_smoothing)

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
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps)

        self.logger.info("Optimizer: total_steps=%d, warmup=%d" % (total_steps, warmup_steps))

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
        self.logger.info("Optimizer rebuilt: remaining_steps=%d" % total_steps)

    def _move_batch(self, batch: Dict) -> Dict:
        result = {
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

        # PQP features (if available)
        if "pqp_text_input_ids" in batch:
            result.update({
                "pqp_text_input_ids": batch["pqp_text_input_ids"].to(self.device),
                "pqp_text_attention_mask": batch["pqp_text_attention_mask"].to(self.device),
                "pqp_audio_input_values": batch["pqp_audio_input_values"].to(self.device),
                "pqp_audio_attention_mask": batch["pqp_audio_attention_mask"].to(self.device),
                "pqp_prosody_features": batch["pqp_prosody_features"].to(self.device),
                "pqp_frame_acoustic_features": batch["pqp_frame_acoustic_features"].to(self.device),
                "pqp_frame_acoustic_mask": batch["pqp_frame_acoustic_mask"].to(self.device),
            })

        return result

    def _apply_mixup(
        self, batch_t: Dict, lam: float, idx: torch.Tensor, num_classes: int
    ) -> tuple:
        """Mix continuous features and build soft labels. Returns (mixed_batch, soft_labels)."""
        mixed = dict(batch_t)
        # Mix prosody and frame_acoustic (continuous); leave text/audio as-is for simplicity.
        mixed["prosody_features"] = lam * batch_t["prosody_features"] + (1 - lam) * batch_t["prosody_features"][idx]
        mixed["frame_acoustic_features"] = lam * batch_t["frame_acoustic_features"] + (1 - lam) * batch_t["frame_acoustic_features"][idx]
        # Soft labels: lam * one_hot(y) + (1-lam) * one_hot(y[idx])
        one_hot = F.one_hot(batch_t["labels"], num_classes=num_classes).float()
        one_hot_mix = F.one_hot(batch_t["labels"][idx], num_classes=num_classes).float()
        soft_labels = lam * one_hot + (1 - lam) * one_hot_mix
        mixed["labels"] = None  # forward will not compute loss; we use soft_ce in trainer
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

    # ------------------------------------------------------------------ #
    def do_train(self, epoch: int) -> Dict:
        self.model.train()
        losses, all_logits, all_labels = [], [], []
        accum = self.cfg.gradient_accumulation_steps
        # 早期 epoch 禁用 R-Drop，待模型收敛后再启用
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

            if rdrop_alpha > 0:
                kl_loss = compute_rdrop_kl_loss(out1["logits"], out2["logits"])
                total_loss = ce_loss + rdrop_alpha * kl_loss
            else:
                total_loss = ce_loss
            if "contrastive_loss" in out1:
                warmup_ep = self.cfg.cdd_loss_warmup_epochs
                cdd_scale = min(1.0, epoch / max(warmup_ep, 1))
                n_out = 2 if rdrop_alpha > 0 else 1
                avg_con = (out1["contrastive_loss"] + (out2["contrastive_loss"] if rdrop_alpha > 0 else out1["contrastive_loss"])) / n_out
                avg_orth = (out1["ortho_loss"] + (out2["ortho_loss"] if rdrop_alpha > 0 else out1["ortho_loss"])) / n_out
                total_loss = total_loss + cdd_scale * self.cfg.lambda_con * avg_con
                total_loss = total_loss + cdd_scale * self.cfg.lambda_orth * avg_orth
                if "align_loss" in out1:
                    avg_align = (out1["align_loss"] + (out2["align_loss"] if rdrop_alpha > 0 else out1["align_loss"])) / n_out
                    total_loss = total_loss + cdd_scale * self.cfg.lambda_align * avg_align
                if "sep_loss" in out1:
                    avg_sep = (out1["sep_loss"] + (out2["sep_loss"] if rdrop_alpha > 0 else out1["sep_loss"])) / n_out
                    total_loss = total_loss + cdd_scale * self.cfg.lambda_sep * avg_sep
                if "recon_loss" in out1:
                    avg_recon = (out1["recon_loss"] + (out2["recon_loss"] if rdrop_alpha > 0 else out1["recon_loss"])) / n_out
                    total_loss = total_loss + cdd_scale * self.cfg.lambda_recon * avg_recon

            if "misa_cmd_loss" in out1:
                n_out = 2 if rdrop_alpha > 0 else 1
                avg_cmd = (out1["misa_cmd_loss"] + (out2["misa_cmd_loss"] if rdrop_alpha > 0 else out1["misa_cmd_loss"])) / n_out
                avg_diff = (out1["misa_diff_loss"] + (out2["misa_diff_loss"] if rdrop_alpha > 0 else out1["misa_diff_loss"])) / n_out
                avg_recon_m = (out1["misa_recon_loss"] + (out2["misa_recon_loss"] if rdrop_alpha > 0 else out1["misa_recon_loss"])) / n_out
                warmup_ep = self.cfg.cdd_loss_warmup_epochs
                misa_scale = min(1.0, epoch / max(warmup_ep, 1))
                total_loss = total_loss + misa_scale * 0.01 * avg_cmd
                total_loss = total_loss + misa_scale * 0.01 * avg_diff
                total_loss = total_loss + misa_scale * 0.01 * avg_recon_m

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

            pbar.set_postfix(
                loss="%.4f" % losses[-1],
                acc="%.4f" % (running_correct / max(running_total, 1)),
            )

        logits = torch.cat(all_logits, 0)
        labels = torch.cat(all_labels, 0)
        metrics = compute_multiclass_metrics(logits, labels)
        metrics["loss"] = float(np.mean(losses))

        self.logger.info(
            "[Train] Epoch %d  loss=%.4f  acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f"
            % (epoch, metrics["loss"], metrics["acc"], metrics["macro_f1"], metrics["weighted_f1"])
        )
        del all_logits, all_labels, logits, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def do_test(self, split: str = "dev") -> Dict:
        loader = self.dev_loader if split == "dev" else self.test_loader
        self.model.eval()
        losses, all_logits, all_labels = [], [], []
        all_layer_logits = [] if self.cfg.use_hierarchical else None

        for batch in tqdm(loader, desc=split.upper(), ncols=120):
            batch_t = self._move_batch(batch)
            out = self.model(**batch_t)
            losses.append(out["loss"].item())
            all_logits.append(out["logits"].detach().cpu())
            all_labels.append(batch_t["labels"].detach().cpu())
            if self.cfg.use_hierarchical and "layer_logits" in out:
                all_layer_logits.append(out["layer_logits"].detach().cpu())

        logits = torch.cat(all_logits, 0)
        labels = torch.cat(all_labels, 0)
        metrics = compute_multiclass_metrics(logits, labels)
        metrics["loss"] = float(np.mean(losses))

        if self.cfg.use_hierarchical and all_layer_logits:
            layer_logits = torch.cat(all_layer_logits, 0)
            layer_pred = layer_logits.argmax(dim=-1)
            layer_ids = torch.tensor([SR_LABEL_ID_TO_LAYER_ID[int(l)] for l in labels.tolist()], dtype=torch.long)
            metrics["layer_prediction_acc"] = (layer_pred == layer_ids).float().mean().item()
            self.logger.info("  [Hierarchical] layer_prediction_acc=%.4f" % metrics["layer_prediction_acc"])

        self.logger.info(
            "[%s]  loss=%.4f  acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f"
            % (split.upper(), metrics["loss"], metrics["acc"], metrics["macro_f1"], metrics["weighted_f1"])
        )
        for lname, lacc in metrics["layer_acc"].items():
            self.logger.info("  Layer %-14s acc=%.4f" % (lname, lacc))

        del all_logits, all_labels, logits, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, name)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, name)
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def _log_json(self, filename: str, payload):
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def Enrun_SR(cfg: SRConfig):
    trainer = SRTrainer(cfg)
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

        trainer.logger.info(
            "--- Epoch %d | best_acc=%.4f  best_macro_f1=%.4f  %s | patience=%d/%d"
            % (epoch, trainer.best_scores["acc"], trainer.best_scores["macro_f1"],
               ("improved: " + ",".join(improved)) if improved else "",
               no_improve, cfg.patience)
        )

    # ---- Final test ----
    trainer.logger.info("===== Final Test =====")
    final_report = {}

    for key in ["acc", "macro_f1"]:
        ckpt = "best_%s.pt" % key
        trainer.load_checkpoint(ckpt)
        test_m = trainer.do_test("test")
        final_report["test_best_%s" % key] = test_m
        trainer.logger.info(
            "Test (best_%s): acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f"
            % (key, test_m["acc"], test_m["macro_f1"], test_m["weighted_f1"])
        )
        trainer.logger.info("Per-class F1:")
        for label in SR_VALID_LABELS:
            pc = test_m["per_class"][label]
            trainer.logger.info(
                "  %-5s  P=%.3f  R=%.3f  F1=%.3f  support=%d"
                % (label, pc["precision"], pc["recall"], pc["f1"], pc["support"])
            )

    if trainer.swa_count > 0:
        trainer._apply_swa()
        trainer.save_checkpoint("swa.pt")
        swa_m = trainer.do_test("test")
        final_report["test_swa"] = swa_m
        trainer.logger.info(
            "Test (SWA, %d epochs): acc=%.4f  macro_f1=%.4f"
            % (trainer.swa_count, swa_m["acc"], swa_m["macro_f1"])
        )

    trainer._log_json("final_test_report.json", final_report)
    trainer._log_json("full_history.json", trainer.history)
    return final_report
