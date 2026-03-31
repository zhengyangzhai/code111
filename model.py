"""model.py — V5 多模态 PQP 模型

改进 (相对 V4):
1. Wav2Vec2 加权层求和 — 可学习地融合所有 Transformer 层输出
2. SpeechCraft 类别嵌入 — 利用 pitch/energy/speed 分类特征
3. 帧级声学特征分支 — 1D-CNN 处理 eGeMAPS-inspired 帧级特征
4. 注意力池化 — 学习音频中哪些时间步最重要
5. 增强韵律分支 — 更深的 MLP，18 维 F0 特征
6. 部分解冻 — 仅解冻音频编码器顶部 4 层
7. 文本永久冻结 — BERT 仅提供轻量上下文嵌入
8. SRMoEModel: PQP 上下文建模 + MOE 分类器
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model

from dataloader import FRAME_ACOUSTIC_DIM, PROSODY_FEAT_DIM
from sr_dataloader import SR_LAYER_MAP, SR_VALID_LABELS


class AttentionPooling(nn.Module):
    """Learned attention-weighted pooling over a sequence."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scores = self.query(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, T)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)


class MultiModalPQPModel(nn.Module):
    """PQP 多模态模型；支持消融：text_only / audio_only / 去掉韵律 / 去掉帧级声学。"""

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 2,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_text_only: bool = False,
        use_audio_only: bool = False,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_text_only = use_text_only
        self.use_audio_only = use_audio_only
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic

        # ---------- 编码器 ----------
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)

        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size   # 768
        audio_hidden = self.audio_encoder.config.hidden_size  # 768

        # ---------- 加权层求和 (embedding + N transformer layers) ----------
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1  # +1 for CNN embedding
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        # ---------- 投影层 ----------
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ---------- 韵律特征分支 (18 维 → MLP → proj_dim) ----------
        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ---------- SpeechCraft 类别嵌入分支 ----------
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(
            nn.Linear(sc_embed_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---------- 帧级声学特征分支 (10 维 × T → 1D-CNN → proj_dim) ----------
        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)

        # ---------- 注意力池化 ----------
        self.audio_attn_pool = AttentionPooling(proj_dim)

        # ---------- 门控融合 ----------
        # audio + prosody + text + speechcraft + frame_acoustic
        fusion_dim = proj_dim * 5
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # ---------- 分类器 ----------
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # -------- 工具方法 --------
    @staticmethod
    def freeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        """仅解冻音频编码器顶部 num_layers 个 Transformer 层。"""
        self.freeze_module(self.audio_encoder)
        total_layers = len(self.audio_encoder.encoder.layers)
        for i in range(total_layers - num_layers, total_layers):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        # 文本编码器始终冻结
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(
        hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(
        self, attention_mask: torch.Tensor, hidden_seq_len: int
    ) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        batch_size = attention_mask.shape[0]
        reduced = torch.zeros(
            batch_size, hidden_seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        for i, length in enumerate(output_lengths):
            reduced[i, : length.item()] = 1
        return reduced

    # -------- 前向 --------
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B_size = text_input_ids.shape[0]
        dev = text_input_ids.device

        # ===== 1. 文本编码（audio_only 时跳过） =====
        if self.use_audio_only:
            text_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            with torch.no_grad():
                text_out = self.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                    return_dict=True,
                )
            text_seq = self.text_proj(text_out.last_hidden_state)
            text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        # ===== 2. 音频编码（text_only 时跳过） =====
        if self.use_text_only:
            audio_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            audio_encoder_frozen = not any(
                p.requires_grad for p in self.audio_encoder.parameters()
            )
            encode_kwargs = dict(
                input_values=audio_input_values,
                attention_mask=audio_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if audio_encoder_frozen:
                with torch.no_grad():
                    audio_out = self.audio_encoder(**encode_kwargs)
            else:
                audio_out = self.audio_encoder(**encode_kwargs)

            hidden_states = torch.stack(audio_out.hidden_states, dim=0)
            w = F.softmax(self.layer_weights, dim=0)
            audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
            audio_seq = self.audio_proj(audio_hidden)
            if audio_attention_mask is not None:
                reduced_mask = self._compute_reduced_audio_mask(
                    audio_attention_mask, audio_hidden.shape[1]
                )
            else:
                reduced_mask = None
            audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        # ===== 3. 韵律特征（消融可关） =====
        if not self.use_prosody or self.use_text_only or prosody_features is None:
            prosody_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)  # (B, proj_dim)

        # ===== 4. SpeechCraft =====
        if self.use_text_only or self.use_audio_only or speechcraft_features is None:
            sc_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            sc_pitch = self.pitch_embed(speechcraft_features[:, 0])
            sc_energy = self.energy_embed(speechcraft_features[:, 1])
            sc_speed = self.speed_embed(speechcraft_features[:, 2])
            sc_cat = torch.cat([sc_pitch, sc_energy, sc_speed], dim=-1)
            sc_emb = self.sc_mlp(sc_cat)

        # ===== 5. 帧级声学（消融可关） =====
        if not self.use_frame_acoustic or self.use_text_only or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            fa = frame_acoustic_features.transpose(1, 2)
            fa = self.frame_acoustic_conv(fa)
            fa = fa.transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        # ===== 6. 五路门控融合 =====
        fused = torch.cat([audio_emb, prosody_emb, text_emb, sc_emb, fa_emb], dim=-1)
        gate_weights = self.gate(fused)
        fused = fused * gate_weights

        # ===== 7. 分类 =====
        logits = self.classifier(fused)

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


# ---------------------------------------------------------------------------
# MulT-style baseline: Crossmodal Transformer (Tsai et al., ACL 2019)
# Adapted for bimodal text-audio with shared encoder backbone.
# ---------------------------------------------------------------------------
class CrossModalTransformerLayer(nn.Module):
    """One layer of crossmodal attention: query from modality A, key/value from modality B."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv_key_padding = ~kv_mask.bool() if kv_mask is not None else None
        attn_out, _ = self.cross_attn(query, kv, kv, key_padding_mask=kv_key_padding)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MulTSRModel(nn.Module):
    """MulT-style bimodal crossmodal transformer baseline for SR 14-class.

    Uses the same BERT + Wav2Vec2 + prosody/SC/FA encoders as MultiModalPQPModel,
    but replaces gated concatenation with bidirectional crossmodal transformer layers.
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 14,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
        num_cross_layers: int = 2,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic

        # ---- Encoders (same as MultiModalPQPModel) ----
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        MultiModalPQPModel.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            MultiModalPQPModel.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        # ---- Auxiliary branches ----
        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM), nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(64, 128), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)
        self.audio_attn_pool = AttentionPooling(proj_dim)

        # ---- Crossmodal Transformer ----
        self.text_with_audio = nn.ModuleList([
            CrossModalTransformerLayer(proj_dim, num_heads, dropout) for _ in range(num_cross_layers)
        ])
        self.audio_with_text = nn.ModuleList([
            CrossModalTransformerLayer(proj_dim, num_heads, dropout) for _ in range(num_cross_layers)
        ])

        # ---- Classifier ----
        # text_cross + audio_cross + prosody + SC + FA = 5 * proj_dim
        fusion_dim = proj_dim * 5
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(proj_dim, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ---- Utility (delegated) ----
    freeze_module = staticmethod(MultiModalPQPModel.freeze_module)
    unfreeze_module = staticmethod(MultiModalPQPModel.unfreeze_module)

    def partial_unfreeze_audio(self, num_layers: int = 4):
        MultiModalPQPModel.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            MultiModalPQPModel.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        MultiModalPQPModel.freeze_module(self.text_encoder)
        if freeze_audio:
            MultiModalPQPModel.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(4)

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B = text_input_ids.shape[0]
        dev = text_input_ids.device

        # 1. Text sequence
        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)  # (B, L_text, D)

        # 2. Audio sequence
        frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kwargs = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                          output_hidden_states=True, return_dict=True)
        if frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kwargs)
        else:
            audio_out = self.audio_encoder(**enc_kwargs)
        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)  # (B, L_audio, D)
        if audio_attention_mask is not None:
            reduced_mask = MultiModalPQPModel._compute_reduced_audio_mask(self, audio_attention_mask, audio_hidden.shape[1])
        else:
            reduced_mask = None

        # 3. Crossmodal transformer: text attends to audio, audio attends to text
        t_seq, a_seq = text_seq, audio_seq
        for layer in self.text_with_audio:
            t_seq = layer(t_seq, a_seq, text_attention_mask, reduced_mask)
        for layer in self.audio_with_text:
            a_seq = layer(a_seq, t_seq, reduced_mask, text_attention_mask)

        # Pool crossmodal outputs
        text_emb = MultiModalPQPModel._masked_mean_pool(t_seq, text_attention_mask)
        audio_emb = self.audio_attn_pool(a_seq, reduced_mask)

        # 4. Auxiliary branches
        prosody_emb = self.prosody_mlp(prosody_features) if (self.use_prosody and prosody_features is not None) else torch.zeros(B, self.proj_dim, device=dev)
        if speechcraft_features is not None:
            sc_cat = torch.cat([self.pitch_embed(speechcraft_features[:, 0]),
                                self.energy_embed(speechcraft_features[:, 1]),
                                self.speed_embed(speechcraft_features[:, 2])], dim=-1)
            sc_emb = self.sc_mlp(sc_cat)
        else:
            sc_emb = torch.zeros(B, self.proj_dim, device=dev)
        if self.use_frame_acoustic and frame_acoustic_features is not None and frame_acoustic_features.shape[1] > 0:
            fa = self.frame_acoustic_norm(self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2))
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)
        else:
            fa_emb = torch.zeros(B, self.proj_dim, device=dev)

        # 5. Classify
        fused = torch.cat([text_emb, audio_emb, prosody_emb, sc_emb, fa_emb], dim=-1)
        logits = self.classifier(fused)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


# ---------------------------------------------------------------------------
# MISA-style baseline: Modality-Invariant and -Specific Representations
# (Hazarika et al., EMNLP 2020), adapted for bimodal text-audio.
# ---------------------------------------------------------------------------
class MISASRModel(nn.Module):
    """MISA-style baseline for SR 14-class.

    Decomposes text and audio representations into modality-invariant (shared)
    and modality-specific subspaces, with similarity loss (CMD) between
    invariant parts and orthogonality loss between shared/specific parts.
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 14,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic

        # ---- Encoders (same backbone) ----
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        MultiModalPQPModel.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            MultiModalPQPModel.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_attn_pool = AttentionPooling(proj_dim)

        # ---- Auxiliary branches ----
        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM), nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(64, 128), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)

        # ---- MISA: shared/private projectors ----
        self.text_shared_proj = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.text_private_proj = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.audio_shared_proj = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.audio_private_proj = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

        # Reconstruction decoder
        self.text_recon = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.audio_recon = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

        # ---- Classifier ----
        # text_shared + text_private + audio_shared + audio_private + prosody + SC + FA = 7 * proj_dim
        fusion_dim = proj_dim * 7
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(proj_dim, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ---- Utility ----
    freeze_module = staticmethod(MultiModalPQPModel.freeze_module)
    unfreeze_module = staticmethod(MultiModalPQPModel.unfreeze_module)

    def partial_unfreeze_audio(self, num_layers: int = 4):
        MultiModalPQPModel.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            MultiModalPQPModel.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        MultiModalPQPModel.freeze_module(self.text_encoder)
        if freeze_audio:
            MultiModalPQPModel.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(4)

    @staticmethod
    def _cmd_loss(x: torch.Tensor, y: torch.Tensor, n_moments: int = 3) -> torch.Tensor:
        """Central Moment Discrepancy between two sets of samples."""
        mx, my = x.mean(dim=0), y.mean(dim=0)
        loss = (mx - my).pow(2).mean()
        cx, cy = x - mx.unsqueeze(0), y - my.unsqueeze(0)
        for k in range(2, n_moments + 1):
            loss = loss + (cx.pow(k).mean(dim=0) - cy.pow(k).mean(dim=0)).pow(2).mean()
        return loss

    @staticmethod
    def _diff_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
        """Orthogonality loss between shared and private representations."""
        s_norm = F.normalize(shared, dim=-1)
        p_norm = F.normalize(private, dim=-1)
        return (s_norm * p_norm).sum(dim=-1).pow(2).mean()

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B = text_input_ids.shape[0]
        dev = text_input_ids.device

        # 1. Text encoding
        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_emb = MultiModalPQPModel._masked_mean_pool(text_seq, text_attention_mask)

        # 2. Audio encoding
        frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kwargs = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                          output_hidden_states=True, return_dict=True)
        if frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kwargs)
        else:
            audio_out = self.audio_encoder(**enc_kwargs)
        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)
        if audio_attention_mask is not None:
            reduced_mask = MultiModalPQPModel._compute_reduced_audio_mask(self, audio_attention_mask, audio_hidden.shape[1])
        else:
            reduced_mask = None
        audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        # 3. MISA decomposition
        text_shared = self.text_shared_proj(text_emb)
        text_private = self.text_private_proj(text_emb)
        audio_shared = self.audio_shared_proj(audio_emb)
        audio_private = self.audio_private_proj(audio_emb)

        # Auxiliary MISA losses
        cmd = self._cmd_loss(text_shared, audio_shared)
        diff = self._diff_loss(text_shared, text_private) + self._diff_loss(audio_shared, audio_private)
        text_recon_out = self.text_recon(torch.cat([text_shared, text_private], dim=-1))
        audio_recon_out = self.audio_recon(torch.cat([audio_shared, audio_private], dim=-1))
        recon = F.mse_loss(text_recon_out, text_emb.detach()) + F.mse_loss(audio_recon_out, audio_emb.detach())

        # 4. Auxiliary branches
        prosody_emb = self.prosody_mlp(prosody_features) if (self.use_prosody and prosody_features is not None) else torch.zeros(B, self.proj_dim, device=dev)
        if speechcraft_features is not None:
            sc_cat = torch.cat([self.pitch_embed(speechcraft_features[:, 0]),
                                self.energy_embed(speechcraft_features[:, 1]),
                                self.speed_embed(speechcraft_features[:, 2])], dim=-1)
            sc_emb = self.sc_mlp(sc_cat)
        else:
            sc_emb = torch.zeros(B, self.proj_dim, device=dev)
        if self.use_frame_acoustic and frame_acoustic_features is not None and frame_acoustic_features.shape[1] > 0:
            fa = self.frame_acoustic_norm(self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2))
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)
        else:
            fa_emb = torch.zeros(B, self.proj_dim, device=dev)

        # 5. Classify (all 4 decomposed + 3 auxiliary)
        fused = torch.cat([text_shared, text_private, audio_shared, audio_private,
                           prosody_emb, sc_emb, fa_emb], dim=-1)
        logits = self.classifier(fused)

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        output["misa_cmd_loss"] = cmd
        output["misa_diff_loss"] = diff
        output["misa_recon_loss"] = recon
        return output


# ---------------------------------------------------------------------------
# SRMoEModel: PQP 上下文建模 + MOE 分类器
# ---------------------------------------------------------------------------
class PQPContextAttention(nn.Module):
    """Cross-Attention: SR features query PQP features for context."""

    def __init__(self, proj_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        sr_emb: torch.Tensor,
        pqp_emb: torch.Tensor,
        pqp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sr_emb: (B, 1, D) - pooled SR feature as query
        pqp_emb: (B, T, D) - PQP sequence as key/value
        pqp_mask: (B, T) - padding mask for PQP
        """
        if pqp_emb.shape[1] == 0:
            return sr_emb.squeeze(1)  # Return (B, D)

        # Cross-attention
        attn_out, _ = self.cross_attn(sr_emb, pqp_emb, pqp_emb, key_padding_mask=pqp_mask)
        attn_out = self.dropout(attn_out)
        # Residual connection
        out = self.norm(sr_emb + attn_out)
        return out.squeeze(1)  # (B, D)


class MOEClassifier(nn.Module):
    """5大类MOE - 每类4个expert，top-2路由

    Layers: factual (4 classes), attitude (3), emotion (2), commitment (2), continuation (3)
    Total: 14 classes
    """

    LAYER_EXPERTS = {
        "factual": ["F", "A", "T", "W"],      # 4
        "attitude": ["S", "D", "R"],           # 3
        "emotion": ["E", "V"],                  # 2
        "commitment": ["I", "H"],               # 2
        "continuation": ["C", "K", "M"],       # 3
    }

    # Map each label to its layer
    LABEL_TO_LAYER = {}
    for layer_name, labels in LAYER_EXPERTS.items():
        for label in labels:
            LABEL_TO_LAYER[label] = layer_name

    def __init__(self, input_dim: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_layers = len(self.LAYER_EXPERTS)

        # 5个layer的专家
        self.experts = nn.ModuleDict()
        for layer_name, labels in self.LAYER_EXPERTS.items():
            num_classes = len(labels)
            self.experts[layer_name] = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim // 2, num_classes)
                ) for _ in range(num_experts)
            ])

        # 门控路由网络 (每个layer独立门控)
        self.gate = nn.Linear(input_dim, num_experts * self.num_layers)

        # 可学习的layer权重 (用于融合)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

        # 存储每个layer的class数量用于输出组装
        self.layer_num_classes = {layer: len(labels) for layer, labels in self.LAYER_EXPERTS.items()}
        self.layer_start_idx = {}
        idx = 0
        for layer_name, labels in self.LAYER_EXPERTS.items():
            self.layer_start_idx[layer_name] = idx
            idx += len(labels)

    def forward(self, x: torch.Tensor):
        """
        x: (B, D) - 融合后的特征
        returns: (logits: (B, 14), balance_loss: scalar)
        """
        batch_size = x.shape[0]

        # 1. 计算路由logits
        gate_logits = self.gate(x)  # (B, 20) = 5*4

        # 2. 每个layer独立top-k路由
        layer_outputs = []
        layer_names = list(self.LAYER_EXPERTS.keys())
        total_balance_loss = 0.0

        for i, layer_name in enumerate(layer_names):
            layer_logits = gate_logits[:, i * self.num_experts:(i + 1) * self.num_experts]  # (B, 4)
            topk_logits, topk_idx = torch.topk(layer_logits, self.top_k, dim=-1)

            # Softmax weights
            weights = F.softmax(topk_logits, dim=-1)  # (B, top_k)

            # --- Load Balancing Loss (Switch Transformer style) ---
            router_probs = F.softmax(layer_logits, dim=-1)  # (B, num_experts)
            # f_i: 分配到每个专家的样本比例
            expert_mask = F.one_hot(topk_idx[:, 0], num_classes=self.num_experts).float()  # (B, num_experts)
            f_i = expert_mask.mean(dim=0)  # (num_experts,)
            # P_i: 路由到每个专家的平均概率
            p_i = router_probs.mean(dim=0)  # (num_experts,)
            total_balance_loss = total_balance_loss + (f_i * p_i).sum() * self.num_experts

            # Expert outputs
            layer_experts = self.experts[layer_name]
            num_classes = self.layer_num_classes[layer_name]
            expert_outputs = torch.stack([exp(x) for exp in layer_experts], dim=1)  # (B, num_experts, num_classes)

            # Gather top-k expert outputs
            topk_expert_outputs = torch.gather(
                expert_outputs, 1,
                topk_idx.unsqueeze(-1).expand(-1, -1, num_classes)
            )  # (B, top_k, num_classes)

            # Weighted sum of top-k experts
            final_out = (topk_expert_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (B, num_classes)
            layer_outputs.append(final_out)

        # 3. 组装14类输出
        full_logits = torch.zeros(batch_size, 14, device=x.device)
        for i, layer_name in enumerate(layer_names):
            start_idx = self.layer_start_idx[layer_name]
            num_classes = self.layer_num_classes[layer_name]
            full_logits[:, start_idx:start_idx + num_classes] = layer_outputs[i]

        # 平均各层的 balance loss
        balance_loss = total_balance_loss / self.num_layers
        return full_logits, balance_loss


# 14 类意图 → 5 个意图层 (factual, attitude, emotion, commitment, continuation)
SR_LABEL_ID_TO_LAYER_ID = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4]  # F,A,T,W | S,D,R | E,V | I,H | C,K,M
SR_NUM_LAYERS = 5


class SRMoEModel(nn.Module):
    """SR 14类分类模型 with PQP 上下文建模 + MOE 分类器；支持消融与层级辅助损失。

    架构:
    SR特征 → 投影 → [可选] Cross-Attention(PQP) → 门控融合 → MOE/分类
    可选：层级头 fusion_dim→5，联合损失 L_class + λ*L_layer
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 14,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_pqp: bool = True,
        use_moe: bool = True,
        use_text_only: bool = False,
        use_audio_only: bool = False,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
        use_hierarchical: bool = False,
        layer_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_pqp = use_pqp
        self.use_moe = use_moe
        self.use_text_only = use_text_only
        self.use_audio_only = use_audio_only
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic
        self.use_hierarchical = use_hierarchical
        self.layer_loss_weight = layer_loss_weight

        # ---------- 编码器 (SR & PQP 共享) ----------
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)

        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size   # 768
        audio_hidden = self.audio_encoder.config.hidden_size  # 768

        # ---------- 加权层求和 ----------
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        # ---------- SR 投影层 ----------
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ---------- PQP 投影层 (独立) ----------
        self.pqp_text_proj = nn.Sequential(
            nn.Linear(text_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.pqp_audio_proj = nn.Sequential(
            nn.Linear(audio_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ---------- 韵律特征分支 ----------
        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # PQP prosody
        self.pqp_prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ---------- SpeechCraft 类别嵌入 ----------
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(
            nn.Linear(sc_embed_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---------- 帧级声学特征分支 ----------
        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)

        # PQP frame acoustic
        self.pqp_frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pqp_frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.pqp_frame_attn_pool = AttentionPooling(proj_dim)

        # ---------- 注意力池化 ----------
        self.audio_attn_pool = AttentionPooling(proj_dim)
        self.pqp_audio_attn_pool = AttentionPooling(proj_dim)

        # ---------- Cross-Attention 上下文建模 ----------
        self.cross_attn = PQPContextAttention(proj_dim, num_heads)
        # 投影层: 将 cross-attn 输出 (proj_dim) 映射到融合维度 (proj_dim*5)
        fusion_dim = proj_dim * 5  # audio + prosody + text + speechcraft + frame_acoustic
        self.context_proj = nn.Sequential(
            nn.Linear(proj_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # ---------- 门控融合 (SR-only for baseline) ----------
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # ---------- MOE 分类器 ----------
        if use_moe:
            self.moe_classifier = MOEClassifier(fusion_dim, num_experts=4, top_k=2, dropout=dropout)
            # 备用的简单分类器 (用于非MOE模式)
            self.simple_classifier = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_labels),
            )

        # ---------- 层级头（意图层 5 类，用于联合损失 / 层级评估） ----------
        self.register_buffer("label_to_layer", torch.tensor(SR_LABEL_ID_TO_LAYER_ID, dtype=torch.long))
        if use_hierarchical:
            self.layer_head = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, SR_NUM_LAYERS),
            )
        else:
            self.layer_head = None

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # -------- 工具方法 --------
    @staticmethod
    def freeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        self.freeze_module(self.audio_encoder)
        total_layers = len(self.audio_encoder.encoder.layers)
        for i in range(total_layers - num_layers, total_layers):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(
        hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(
        self, attention_mask: torch.Tensor, hidden_seq_len: int
    ) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        batch_size = attention_mask.shape[0]
        reduced = torch.zeros(
            batch_size, hidden_seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        for i, length in enumerate(output_lengths):
            reduced[i, : length.item()] = 1
        return reduced

    # -------- 前向 --------
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        # PQP features
        pqp_text_input_ids: Optional[torch.Tensor] = None,
        pqp_text_attention_mask: Optional[torch.Tensor] = None,
        pqp_audio_input_values: Optional[torch.Tensor] = None,
        pqp_audio_attention_mask: Optional[torch.Tensor] = None,
        pqp_prosody_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_mask: Optional[torch.Tensor] = None,
        # Other
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B_size = text_input_ids.shape[0]
        dev = text_input_ids.device

        # ===== 1. SR 文本编码（audio_only 时跳过） =====
        if self.use_audio_only:
            text_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            with torch.no_grad():
                text_out = self.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                    return_dict=True,
                )
            text_seq = self.text_proj(text_out.last_hidden_state)
            text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        # ===== 2. SR 音频编码（text_only 时跳过） =====
        if self.use_text_only:
            audio_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            audio_encoder_frozen = not any(
                p.requires_grad for p in self.audio_encoder.parameters()
            )
            encode_kwargs = dict(
                input_values=audio_input_values,
                attention_mask=audio_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if audio_encoder_frozen:
                with torch.no_grad():
                    audio_out = self.audio_encoder(**encode_kwargs)
            else:
                audio_out = self.audio_encoder(**encode_kwargs)

            hidden_states = torch.stack(audio_out.hidden_states, dim=0)
            w = F.softmax(self.layer_weights, dim=0)
            audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
            audio_seq = self.audio_proj(audio_hidden)
            if audio_attention_mask is not None:
                reduced_mask = self._compute_reduced_audio_mask(
                    audio_attention_mask, audio_hidden.shape[1]
                )
            else:
                reduced_mask = None
            audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        # ===== 3. SR 韵律（消融可关） =====
        if not self.use_prosody or self.use_text_only or prosody_features is None:
            prosody_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)

        # ===== 4. SR SpeechCraft =====
        if speechcraft_features is not None and speechcraft_features.abs().sum() > 0:
            sc_pitch = self.pitch_embed(speechcraft_features[:, 0])
            sc_energy = self.energy_embed(speechcraft_features[:, 1])
            sc_speed = self.speed_embed(speechcraft_features[:, 2])
            sc_cat = torch.cat([sc_pitch, sc_energy, sc_speed], dim=-1)
            sc_emb = self.sc_mlp(sc_cat)
        else:
            sc_emb = torch.zeros(B_size, self.proj_dim, device=dev)

        # ===== 5. SR 帧级声学（消融可关） =====
        if not self.use_frame_acoustic or self.use_text_only or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            fa = frame_acoustic_features.transpose(1, 2)
            fa = self.frame_acoustic_conv(fa)
            fa = fa.transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        # ===== 6. 门控融合 (SR-only baseline) =====
        fused = torch.cat([audio_emb, prosody_emb, text_emb, sc_emb, fa_emb], dim=-1)
        gate_weights = self.gate(fused)
        fused = fused * gate_weights  # (B, proj_dim * 5)

        # ===== 7. PQP 上下文建模 (如果启用) =====
        if self.use_pqp and pqp_audio_input_values is not None:
            # 检测PQP数据是否有效 (attention_mask全为0表示该batch无PQP数据)
            pqp_has_data = True
            if pqp_text_attention_mask is not None:
                pqp_has_data = pqp_text_attention_mask.sum() > 0

            if pqp_has_data:
                # PQP 文本编码
                if pqp_text_input_ids is not None:
                    with torch.no_grad():
                        pqp_text_out = self.text_encoder(
                            input_ids=pqp_text_input_ids,
                            attention_mask=pqp_text_attention_mask,
                            return_dict=True,
                        )
                    pqp_text_seq = self.pqp_text_proj(pqp_text_out.last_hidden_state)
                else:
                    pqp_text_seq = torch.zeros(B_size, 1, self.proj_dim, device=audio_emb.device)

                # PQP 音频编码
                pqp_encode_kwargs = dict(
                    input_values=pqp_audio_input_values,
                    attention_mask=pqp_audio_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                with torch.no_grad():
                    pqp_audio_out = self.audio_encoder(**pqp_encode_kwargs)

                pqp_hidden_states = torch.stack(pqp_audio_out.hidden_states, dim=0)
                pqp_w = F.softmax(self.layer_weights, dim=0)
                pqp_audio_hidden = (pqp_hidden_states * pqp_w.view(-1, 1, 1, 1)).sum(dim=0)
                pqp_audio_seq = self.pqp_audio_proj(pqp_audio_hidden)

                if pqp_audio_attention_mask is not None:
                    pqp_reduced_mask = self._compute_reduced_audio_mask(
                        pqp_audio_attention_mask, pqp_audio_hidden.shape[1]
                    )
                else:
                    pqp_reduced_mask = None

                pqp_audio_emb = self.pqp_audio_attn_pool(pqp_audio_seq, pqp_reduced_mask)  # (B, D)

                # PQP 韵律
                if pqp_prosody_features is not None and pqp_prosody_features.abs().sum() > 0:
                    pqp_prosody_emb = self.pqp_prosody_mlp(pqp_prosody_features)
                else:
                    pqp_prosody_emb = torch.zeros_like(pqp_audio_emb)

                # PQP 帧级声学
                if pqp_frame_acoustic_features is not None and pqp_frame_acoustic_features.shape[1] > 0:
                    pqp_fa = pqp_frame_acoustic_features.transpose(1, 2)
                    pqp_fa = self.pqp_frame_acoustic_conv(pqp_fa)
                    pqp_fa = pqp_fa.transpose(1, 2)
                    pqp_fa = self.pqp_frame_acoustic_norm(pqp_fa)
                    pqp_fa_emb = self.pqp_frame_attn_pool(pqp_fa, pqp_frame_acoustic_mask)
                else:
                    pqp_fa_emb = torch.zeros(B_size, self.proj_dim, device=audio_emb.device)

                # PQP 融合
                pqp_fused = torch.cat([pqp_audio_emb, pqp_prosody_emb, pqp_text_seq.mean(1), sc_emb, pqp_fa_emb], dim=-1)

                # Cross-Attention: SR fused query PQP fused
                sr_query = fused[:, :self.proj_dim].unsqueeze(1)  # (B, 1, proj_dim)
                pqp_key_val = pqp_fused[:, :self.proj_dim].unsqueeze(1)  # (B, 1, proj_dim)

                context_emb = self.cross_attn(sr_query, pqp_key_val, None)  # (B, proj_dim)

                # 投影到融合维度后做残差
                context_proj = self.context_proj(context_emb)  # (B, fusion_dim)
                fused = fused + context_proj  # Residual

        # ===== 8. 分类 =====
        balance_loss = 0.0
        if self.use_moe:
            logits, balance_loss = self.moe_classifier(fused)
        else:
            logits = self.classifier(fused)

        output = {"logits": logits}

        # ===== 9. 层级头（联合损失与评估） =====
        if self.layer_head is not None:
            layer_logits = self.layer_head(fused)  # (B, 5)
            output["layer_logits"] = layer_logits
            if labels is not None:
                layer_ids = self.label_to_layer[labels]  # (B,)
                output["layer_loss"] = self.loss_fn(layer_logits, layer_ids)

        if labels is not None:
            loss = self.loss_fn(logits, labels) + 0.01 * balance_loss
            if self.layer_head is not None and "layer_loss" in output:
                loss = loss + self.layer_loss_weight * output["layer_loss"]
            output["loss"] = loss
        return output


# ===========================================================================
# PBCF: Prosody-Bridged Cross-Modal Fusion
# ===========================================================================

class CrossModalBridge(nn.Module):
    """Token-frame cross-attention + semantic-acoustic discrepancy detection.

    Sits between per-modality encoding and gated fusion.  Produces a
    discrepancy-aware augmentation vector that modulates the fusion gate.
    """

    def __init__(self, proj_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # --- bidirectional cross-attention ---
        self.text2audio = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio2text = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t2a = nn.LayerNorm(proj_dim)
        self.norm_a2t = nn.LayerNorm(proj_dim)

        # --- discrepancy detection MLP ---
        self.disc_mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(
        self,
        text_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            text_seq:  (B, T_text, D) — projected text token sequence
            audio_seq: (B, T_audio, D) — projected audio frame sequence
            text_emb:  (B, D) — pooled text embedding
            audio_emb: (B, D) — pooled audio embedding
            text_mask:  (B, T_text) — 1=valid, 0=pad
            audio_mask: (B, T_audio) — 1=valid, 0=pad
        Returns:
            text_cross_emb:  (B, D) — cross-attended text
            audio_cross_emb: (B, D) — cross-attended audio
            disc_vector:     (B, D) — discrepancy signal
        """
        text_key_pad = (~text_mask.bool()) if text_mask is not None else None
        audio_key_pad = (~audio_mask.bool()) if audio_mask is not None else None

        # text tokens attend to audio frames
        t2a_out, _ = self.text2audio(text_seq, audio_seq, audio_seq, key_padding_mask=audio_key_pad)
        t2a_out = self.norm_t2a(text_seq + t2a_out)

        # audio frames attend to text tokens
        a2t_out, _ = self.audio2text(audio_seq, text_seq, text_seq, key_padding_mask=text_key_pad)
        a2t_out = self.norm_a2t(audio_seq + a2t_out)

        # pool cross-attended sequences
        if text_mask is not None:
            m = text_mask.unsqueeze(-1).to(t2a_out.dtype)
            text_cross_emb = (t2a_out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            text_cross_emb = t2a_out.mean(dim=1)

        if audio_mask is not None:
            m = audio_mask.unsqueeze(-1).to(a2t_out.dtype)
            audio_cross_emb = (a2t_out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            audio_cross_emb = a2t_out.mean(dim=1)

        # discrepancy = diff between original pooled emb and cross-attended emb
        text_disc = text_emb - text_cross_emb
        audio_disc = audio_emb - audio_cross_emb
        disc_input = torch.cat([text_disc, audio_disc, text_disc * audio_disc], dim=-1)
        disc_vector = self.disc_mlp(disc_input)  # (B, D)

        return text_cross_emb, audio_cross_emb, disc_vector


class CrossModalBridgeV2(nn.Module):
    """Token-level discrepancy localization (TLDL): per-token discrepancy scoring + discrepancy-weighted pooling.

    Replaces CrossModalBridge + GuidedDisentanglement: produces text_cons, text_disc, audio_cons, audio_disc
    directly from weighted pooling over token-level discrepancy scores.
    """

    def __init__(
        self,
        proj_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_tone_aware_mask: bool = False,
        tone_mask_gamma: float = 0.5,
        tone_mask_temp: float = 1.0,
        tone_var_dim: int = 1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.text2audio = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio2text = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t2a = nn.LayerNorm(proj_dim)
        self.norm_a2t = nn.LayerNorm(proj_dim)
        self.use_tone_aware_mask = use_tone_aware_mask
        self.tone_mask_gamma = tone_mask_gamma
        self.tone_mask_temp = tone_mask_temp
        self.tone_var_dim = tone_var_dim

        self.text_disc_scorer = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
        )
        self.audio_disc_scorer = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
        )
        self.disc_temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.cons_proj_text = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.disc_proj_text = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.cons_proj_audio = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.disc_proj_audio = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim))

        self.disc_mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    @staticmethod
    def _weighted_masked_pool(
        seq: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """seq (B, T, D), weights (B, T, 1), mask (B, T) 1=valid. Return (B, D)."""
        if mask is not None:
            w = weights * mask.unsqueeze(-1).to(weights.dtype)
        else:
            w = weights
        denom = w.sum(dim=1).clamp(min=1e-6)
        return (seq * w).sum(dim=1) / denom

    def forward(
        self,
        text_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        tone_var: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        text_key_pad = (~text_mask.bool()) if text_mask is not None else None
        audio_key_pad = (~audio_mask.bool()) if audio_mask is not None else None

        t2a_raw, t2a_attn = self.text2audio(
            text_seq, audio_seq, audio_seq, key_padding_mask=audio_key_pad, need_weights=True
        )
        t2a_out = self.norm_t2a(text_seq + t2a_raw)

        a2t_raw, a2t_attn = self.audio2text(
            audio_seq, text_seq, text_seq, key_padding_mask=text_key_pad, need_weights=True
        )
        a2t_out = self.norm_a2t(audio_seq + a2t_raw)

        text_diff = t2a_raw
        audio_diff = a2t_raw
        tau = self.disc_temperature.clamp(min=0.01)
        text_disc_scores = torch.sigmoid(self.text_disc_scorer(text_diff) / tau)
        audio_disc_scores = torch.sigmoid(self.audio_disc_scorer(audio_diff) / tau)
        if self.use_tone_aware_mask and tone_var is not None:
            tone_var = tone_var.to(text_disc_scores.dtype)
            tone_mask = torch.sigmoid(
                tone_var.view(-1, 1, 1) / max(float(self.tone_mask_temp), 1e-6)
            )
            scale = 1.0 + float(self.tone_mask_gamma) * tone_mask
            text_disc_scores = torch.clamp(text_disc_scores * scale, 0.0, 1.0)
            audio_disc_scores = torch.clamp(audio_disc_scores * scale, 0.0, 1.0)

        text_cons_emb = self._weighted_masked_pool(text_seq, 1.0 - text_disc_scores, text_mask)
        text_disc_emb = self._weighted_masked_pool(text_seq, text_disc_scores, text_mask)
        audio_cons_emb = self._weighted_masked_pool(audio_seq, 1.0 - audio_disc_scores, audio_mask)
        audio_disc_emb = self._weighted_masked_pool(audio_seq, audio_disc_scores, audio_mask)

        text_cons = self.cons_proj_text(text_cons_emb)
        text_disc = self.disc_proj_text(text_disc_emb)
        audio_cons = self.cons_proj_audio(audio_cons_emb)
        audio_disc = self.disc_proj_audio(audio_disc_emb)

        if text_mask is not None:
            m = text_mask.unsqueeze(-1).to(t2a_out.dtype)
            text_cross_emb = (t2a_out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            text_cross_emb = t2a_out.mean(dim=1)
        if audio_mask is not None:
            m = audio_mask.unsqueeze(-1).to(a2t_out.dtype)
            audio_cross_emb = (a2t_out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            audio_cross_emb = a2t_out.mean(dim=1)
        text_disc_vec = text_emb - text_cross_emb
        audio_disc_vec = audio_emb - audio_cross_emb
        disc_input = torch.cat([text_disc_vec, audio_disc_vec, text_disc_vec * audio_disc_vec], dim=-1)
        disc_vector = self.disc_mlp(disc_input)

        return {
            "text_cons": text_cons,
            "text_disc": text_disc,
            "audio_cons": audio_cons,
            "audio_disc": audio_disc,
            "disc_vector": disc_vector,
            "text_disc_scores": text_disc_scores.squeeze(-1),
            "audio_disc_scores": audio_disc_scores.squeeze(-1),
            "t2a_attn": t2a_attn,
        }


class PBCFMultiModalPQPModel(nn.Module):
    """PQP 增强模型: 在 MultiModalPQPModel 架构上加入 PBCF 跨模态韵律桥接。

    与原模型相同的 5 路特征编码，在门控融合前插入 CrossModalBridge，
    用不一致信号扩展门控输入维度，增强对字面-深层含义冲突的判别。
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 2,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
        use_cross_attn: bool = True,
        use_discrepancy: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic
        self.use_cross_attn = use_cross_attn
        self.use_discrepancy = use_discrepancy

        # ---------- encoders ----------
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size

        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        # ---------- projections ----------
        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        # ---------- prosody ----------
        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )

        # ---------- SpeechCraft ----------
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))

        # ---------- frame acoustic ----------
        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)

        # ---------- attention pooling ----------
        self.audio_attn_pool = AttentionPooling(proj_dim)

        # ---------- PBCF: Cross-Modal Bridge ----------
        self.bridge = CrossModalBridge(proj_dim, num_heads, dropout)

        # ---------- discrepancy-modulated gating ----------
        fusion_dim = proj_dim * 5  # audio + prosody + text + sc + fa
        gate_input_dim = fusion_dim + proj_dim  # +disc_vector
        self.gate = nn.Sequential(nn.Linear(gate_input_dim, fusion_dim), nn.Sigmoid())

        # ---------- classifier ----------
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(proj_dim, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # -------- utilities --------
    @staticmethod
    def freeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        self.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        B = attention_mask.shape[0]
        reduced = torch.zeros(B, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            reduced[i, :length.item()] = 1
        return reduced

    # -------- forward --------
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B = text_input_ids.shape[0]
        dev = text_input_ids.device

        # ===== 1. text encoding =====
        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)  # (B, T_text, D)
        text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        # ===== 2. audio encoding =====
        audio_frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kw = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                      output_hidden_states=True, return_dict=True)
        if audio_frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kw)
        else:
            audio_out = self.audio_encoder(**enc_kw)

        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)  # (B, T_audio, D)
        reduced_mask = None
        if audio_attention_mask is not None:
            reduced_mask = self._compute_reduced_audio_mask(audio_attention_mask, audio_hidden.shape[1])
        audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        # ===== 3. PBCF: Cross-Modal Bridge =====
        if self.use_cross_attn:
            text_cross, audio_cross, disc = self.bridge(
                text_seq, audio_seq, text_emb, audio_emb,
                text_mask=text_attention_mask, audio_mask=reduced_mask,
            )
            if self.use_discrepancy:
                disc_signal = disc
            else:
                disc_signal = torch.zeros(B, self.proj_dim, device=dev)
        else:
            disc_signal = torch.zeros(B, self.proj_dim, device=dev)

        # ===== 4. prosody =====
        if not self.use_prosody or prosody_features is None:
            prosody_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)

        # ===== 5. SpeechCraft =====
        if speechcraft_features is None:
            sc_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            sc_emb = self.sc_mlp(torch.cat([
                self.pitch_embed(speechcraft_features[:, 0]),
                self.energy_embed(speechcraft_features[:, 1]),
                self.speed_embed(speechcraft_features[:, 2]),
            ], dim=-1))

        # ===== 6. frame acoustic =====
        if not self.use_frame_acoustic or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            fa = self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        # ===== 7. discrepancy-modulated gated fusion =====
        fused = torch.cat([audio_emb, prosody_emb, text_emb, sc_emb, fa_emb], dim=-1)
        gate_input = torch.cat([fused, disc_signal], dim=-1)
        gate_weights = self.gate(gate_input)
        fused = fused * gate_weights

        # ===== 8. classify =====
        logits = self.classifier(fused)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


class PBCFSRMoEModel(nn.Module):
    """SR 增强模型: 在 SRMoEModel 架构上加入 PBCF 跨模态韵律桥接。

    融合前用 CrossModalBridge 做文本-音频跨模态交互 + 不一致性检测，
    然后不一致信号参与门控，再接 PQP 上下文建模 + MOE 分类。
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 14,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_pqp: bool = True,
        use_moe: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
        use_hierarchical: bool = False,
        layer_loss_weight: float = 0.3,
        use_cross_attn: bool = True,
        use_discrepancy: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_pqp = use_pqp
        self.use_moe = use_moe
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic
        self.use_hierarchical = use_hierarchical
        self.layer_loss_weight = layer_loss_weight
        self.use_cross_attn = use_cross_attn
        self.use_discrepancy = use_discrepancy

        # ---------- encoders ----------
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        # ---------- SR projections ----------
        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        # ---------- PQP projections ----------
        self.pqp_text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.pqp_audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        # ---------- prosody ----------
        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        self.pqp_prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )

        # ---------- SpeechCraft ----------
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))

        # ---------- frame acoustic ----------
        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)

        self.pqp_frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.pqp_frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.pqp_frame_attn_pool = AttentionPooling(proj_dim)

        # ---------- attention pooling ----------
        self.audio_attn_pool = AttentionPooling(proj_dim)
        self.pqp_audio_attn_pool = AttentionPooling(proj_dim)

        # ---------- PBCF: Cross-Modal Bridge ----------
        self.bridge = CrossModalBridge(proj_dim, num_heads, dropout)

        # ---------- PQP context cross-attention ----------
        self.cross_attn = PQPContextAttention(proj_dim, num_heads)
        fusion_dim = proj_dim * 5
        self.context_proj = nn.Sequential(
            nn.Linear(proj_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(),
        )

        # ---------- discrepancy-modulated gating ----------
        gate_input_dim = fusion_dim + proj_dim  # +disc_vector
        self.gate = nn.Sequential(nn.Linear(gate_input_dim, fusion_dim), nn.Sigmoid())

        # ---------- MOE classifier ----------
        if use_moe:
            self.moe_classifier = MOEClassifier(fusion_dim, num_experts=4, top_k=2, dropout=dropout)
            self.simple_classifier = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(proj_dim, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(proj_dim, num_labels),
            )

        # ---------- hierarchical head ----------
        self.register_buffer("label_to_layer", torch.tensor(SR_LABEL_ID_TO_LAYER_ID, dtype=torch.long))
        if use_hierarchical:
            self.layer_head = nn.Sequential(
                nn.Linear(fusion_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(proj_dim, SR_NUM_LAYERS),
            )
        else:
            self.layer_head = None

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # -------- utilities --------
    @staticmethod
    def freeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        self.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        B = attention_mask.shape[0]
        reduced = torch.zeros(B, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            reduced[i, :length.item()] = 1
        return reduced

    # -------- forward --------
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        pqp_text_input_ids: Optional[torch.Tensor] = None,
        pqp_text_attention_mask: Optional[torch.Tensor] = None,
        pqp_audio_input_values: Optional[torch.Tensor] = None,
        pqp_audio_attention_mask: Optional[torch.Tensor] = None,
        pqp_prosody_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B_size = text_input_ids.shape[0]
        dev = text_input_ids.device

        # ===== 1. SR text encoding =====
        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        # ===== 2. SR audio encoding =====
        audio_frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kw = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                      output_hidden_states=True, return_dict=True)
        if audio_frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kw)
        else:
            audio_out = self.audio_encoder(**enc_kw)

        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)
        reduced_mask = None
        if audio_attention_mask is not None:
            reduced_mask = self._compute_reduced_audio_mask(audio_attention_mask, audio_hidden.shape[1])
        audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        # ===== 3. PBCF: Cross-Modal Bridge =====
        if self.use_cross_attn:
            _, _, disc = self.bridge(
                text_seq, audio_seq, text_emb, audio_emb,
                text_mask=text_attention_mask, audio_mask=reduced_mask,
            )
            if self.use_discrepancy:
                disc_signal = disc
            else:
                disc_signal = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            disc_signal = torch.zeros(B_size, self.proj_dim, device=dev)

        # ===== 4. prosody =====
        if not self.use_prosody or prosody_features is None:
            prosody_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)

        # ===== 5. SpeechCraft =====
        if speechcraft_features is not None and speechcraft_features.abs().sum() > 0:
            sc_emb = self.sc_mlp(torch.cat([
                self.pitch_embed(speechcraft_features[:, 0]),
                self.energy_embed(speechcraft_features[:, 1]),
                self.speed_embed(speechcraft_features[:, 2]),
            ], dim=-1))
        else:
            sc_emb = torch.zeros(B_size, self.proj_dim, device=dev)

        # ===== 6. frame acoustic =====
        if not self.use_frame_acoustic or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            fa = self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        # ===== 7. discrepancy-modulated gated fusion =====
        fused = torch.cat([audio_emb, prosody_emb, text_emb, sc_emb, fa_emb], dim=-1)
        gate_input = torch.cat([fused, disc_signal], dim=-1)
        gate_weights = self.gate(gate_input)
        fused = fused * gate_weights

        # ===== 8. PQP context (same as SRMoEModel) =====
        if self.use_pqp and pqp_audio_input_values is not None:
            pqp_has_data = True
            if pqp_text_attention_mask is not None:
                pqp_has_data = pqp_text_attention_mask.sum() > 0

            if pqp_has_data:
                if pqp_text_input_ids is not None:
                    with torch.no_grad():
                        pqp_text_out = self.text_encoder(
                            input_ids=pqp_text_input_ids, attention_mask=pqp_text_attention_mask, return_dict=True)
                    pqp_text_seq = self.pqp_text_proj(pqp_text_out.last_hidden_state)
                else:
                    pqp_text_seq = torch.zeros(B_size, 1, self.proj_dim, device=dev)

                pqp_enc_kw = dict(input_values=pqp_audio_input_values, attention_mask=pqp_audio_attention_mask,
                                  output_hidden_states=True, return_dict=True)
                with torch.no_grad():
                    pqp_audio_out = self.audio_encoder(**pqp_enc_kw)

                pqp_hidden = torch.stack(pqp_audio_out.hidden_states, dim=0)
                pqp_audio_hidden = (pqp_hidden * w.view(-1, 1, 1, 1)).sum(dim=0)
                pqp_audio_seq = self.pqp_audio_proj(pqp_audio_hidden)

                pqp_reduced_mask = None
                if pqp_audio_attention_mask is not None:
                    pqp_reduced_mask = self._compute_reduced_audio_mask(
                        pqp_audio_attention_mask, pqp_audio_hidden.shape[1])
                pqp_audio_emb = self.pqp_audio_attn_pool(pqp_audio_seq, pqp_reduced_mask)

                if pqp_prosody_features is not None and pqp_prosody_features.abs().sum() > 0:
                    pqp_prosody_emb = self.pqp_prosody_mlp(pqp_prosody_features)
                else:
                    pqp_prosody_emb = torch.zeros_like(pqp_audio_emb)

                if pqp_frame_acoustic_features is not None and pqp_frame_acoustic_features.shape[1] > 0:
                    pqp_fa = self.pqp_frame_acoustic_conv(
                        pqp_frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
                    pqp_fa = self.pqp_frame_acoustic_norm(pqp_fa)
                    pqp_fa_emb = self.pqp_frame_attn_pool(pqp_fa, pqp_frame_acoustic_mask)
                else:
                    pqp_fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)

                pqp_fused = torch.cat([pqp_audio_emb, pqp_prosody_emb, pqp_text_seq.mean(1), sc_emb, pqp_fa_emb], dim=-1)

                sr_query = fused[:, :self.proj_dim].unsqueeze(1)
                pqp_key_val = pqp_fused[:, :self.proj_dim].unsqueeze(1)
                context_emb = self.cross_attn(sr_query, pqp_key_val, None)
                context_proj = self.context_proj(context_emb)
                fused = fused + context_proj

        # ===== 9. classify =====
        balance_loss = 0.0
        if self.use_moe:
            logits, balance_loss = self.moe_classifier(fused)
        else:
            logits = self.classifier(fused)

        output = {"logits": logits}

        # ===== 10. hierarchical head =====
        if self.layer_head is not None:
            layer_logits = self.layer_head(fused)
            output["layer_logits"] = layer_logits
            if labels is not None:
                layer_ids = self.label_to_layer[labels]
                output["layer_loss"] = self.loss_fn(layer_logits, layer_ids)

        if labels is not None:
            loss = self.loss_fn(logits, labels) + 0.01 * balance_loss
            if self.layer_head is not None and "layer_loss" in output:
                loss = loss + self.layer_loss_weight * output["layer_loss"]
            output["loss"] = loss
        return output


# ===========================================================================
# DRBF: Discrepancy-Routed Bilinear Fusion
# ===========================================================================

class BilinearModalityPool(nn.Module):
    """Low-rank bilinear pooling: (B,D) x (B,D) -> (B,D_out) via proj_a(a) * proj_b(b)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj_a = nn.Linear(dim_in, dim_out)
        self.proj_b = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj_a(a) * self.proj_b(b))


class DiscrepancyRoutedFusion(nn.Module):
    """5 modality pairs with bilinear pooling, routed by discrepancy; residual from concat."""

    def __init__(self, proj_dim: int, num_pairs: int = 5, dropout: float = 0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_pairs = num_pairs
        self.pairs = nn.ModuleList([
            BilinearModalityPool(proj_dim, proj_dim) for _ in range(num_pairs)
        ])
        self.router = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_pairs),
        )
        fusion_dim = proj_dim * 5
        self.residual_proj = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        prosody_emb: torch.Tensor,
        sc_emb: torch.Tensor,
        fa_emb: torch.Tensor,
        disc_vector: torch.Tensor,
    ) -> torch.Tensor:
        pair_0 = self.pairs[0](text_emb, audio_emb)
        pair_1 = self.pairs[1](text_emb, prosody_emb)
        pair_2 = self.pairs[2](audio_emb, prosody_emb)
        pair_3 = self.pairs[3](audio_emb, fa_emb)
        pair_4 = self.pairs[4](text_emb, sc_emb)
        stacked = torch.stack([pair_0, pair_1, pair_2, pair_3, pair_4], dim=1)
        route_weights = F.softmax(self.router(disc_vector), dim=-1)
        bilinear_fused = (stacked * route_weights.unsqueeze(-1)).sum(dim=1)
        concat_all = torch.cat([audio_emb, prosody_emb, text_emb, sc_emb, fa_emb], dim=-1)
        residual = self.residual_proj(concat_all)
        out = self.out_proj(torch.cat([bilinear_fused, residual], dim=-1))
        return out


class DRBFPQPModel(nn.Module):
    """PQP 新框架: 编码器 + CrossModalBridge + DiscrepancyRoutedFusion + 分类器。"""

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 2,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))

        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)
        self.audio_attn_pool = AttentionPooling(proj_dim)

        self.bridge = CrossModalBridge(proj_dim, num_heads, dropout)
        self.drbf = DiscrepancyRoutedFusion(proj_dim, num_pairs=5, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def freeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        self.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        B = attention_mask.shape[0]
        reduced = torch.zeros(B, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            reduced[i, :length.item()] = 1
        return reduced

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B = text_input_ids.shape[0]
        dev = text_input_ids.device

        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        audio_frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kw = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                      output_hidden_states=True, return_dict=True)
        if audio_frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kw)
        else:
            audio_out = self.audio_encoder(**enc_kw)
        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)
        reduced_mask = None
        if audio_attention_mask is not None:
            reduced_mask = self._compute_reduced_audio_mask(audio_attention_mask, audio_hidden.shape[1])
        audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        _, _, disc_vector = self.bridge(
            text_seq, audio_seq, text_emb, audio_emb,
            text_mask=text_attention_mask, audio_mask=reduced_mask,
        )

        if not self.use_prosody or prosody_features is None:
            prosody_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)

        if speechcraft_features is None:
            sc_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            sc_emb = self.sc_mlp(torch.cat([
                self.pitch_embed(speechcraft_features[:, 0]),
                self.energy_embed(speechcraft_features[:, 1]),
                self.speed_embed(speechcraft_features[:, 2]),
            ], dim=-1))

        if not self.use_frame_acoustic or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            fa = self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        fused = self.drbf(text_emb, audio_emb, prosody_emb, sc_emb, fa_emb, disc_vector)
        logits = self.classifier(fused)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


class DRBFSRMoEModel(nn.Module):
    """SR 新框架: 编码器 + CrossModalBridge + DiscrepancyRoutedFusion + PQP 上下文 + MOE + 层级头。"""

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 14,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_pqp: bool = True,
        use_moe: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
        use_hierarchical: bool = False,
        layer_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_pqp = use_pqp
        self.use_moe = use_moe
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic
        self.use_hierarchical = use_hierarchical
        self.layer_loss_weight = layer_loss_weight

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.pqp_text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.pqp_audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        self.pqp_prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )

        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))

        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)
        self.pqp_frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.pqp_frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.pqp_frame_attn_pool = AttentionPooling(proj_dim)
        self.audio_attn_pool = AttentionPooling(proj_dim)
        self.pqp_audio_attn_pool = AttentionPooling(proj_dim)

        self.bridge = CrossModalBridge(proj_dim, num_heads, dropout)
        self.drbf = DiscrepancyRoutedFusion(proj_dim, num_pairs=5, dropout=dropout)

        self.cross_attn = PQPContextAttention(proj_dim, num_heads)
        self.context_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
        )

        if use_moe:
            self.moe_classifier = MOEClassifier(proj_dim, num_experts=4, top_k=2, dropout=dropout)
            self.simple_classifier = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2), nn.LayerNorm(proj_dim // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(proj_dim // 2, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2), nn.LayerNorm(proj_dim // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(proj_dim // 2, num_labels),
            )

        self.register_buffer("label_to_layer", torch.tensor(SR_LABEL_ID_TO_LAYER_ID, dtype=torch.long))
        if use_hierarchical:
            self.layer_head = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(proj_dim // 2, SR_NUM_LAYERS),
            )
        else:
            self.layer_head = None

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def freeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        self.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        B = attention_mask.shape[0]
        reduced = torch.zeros(B, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            reduced[i, :length.item()] = 1
        return reduced

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        pqp_text_input_ids: Optional[torch.Tensor] = None,
        pqp_text_attention_mask: Optional[torch.Tensor] = None,
        pqp_audio_input_values: Optional[torch.Tensor] = None,
        pqp_audio_attention_mask: Optional[torch.Tensor] = None,
        pqp_prosody_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B_size = text_input_ids.shape[0]
        dev = text_input_ids.device

        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        audio_frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kw = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                      output_hidden_states=True, return_dict=True)
        if audio_frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kw)
        else:
            audio_out = self.audio_encoder(**enc_kw)
        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)
        reduced_mask = None
        if audio_attention_mask is not None:
            reduced_mask = self._compute_reduced_audio_mask(audio_attention_mask, audio_hidden.shape[1])
        audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        _, _, disc_vector = self.bridge(
            text_seq, audio_seq, text_emb, audio_emb,
            text_mask=text_attention_mask, audio_mask=reduced_mask,
        )

        if not self.use_prosody or prosody_features is None:
            prosody_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)

        if speechcraft_features is not None and speechcraft_features.abs().sum() > 0:
            sc_emb = self.sc_mlp(torch.cat([
                self.pitch_embed(speechcraft_features[:, 0]),
                self.energy_embed(speechcraft_features[:, 1]),
                self.speed_embed(speechcraft_features[:, 2]),
            ], dim=-1))
        else:
            sc_emb = torch.zeros(B_size, self.proj_dim, device=dev)

        if not self.use_frame_acoustic or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            fa = self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        fused = self.drbf(text_emb, audio_emb, prosody_emb, sc_emb, fa_emb, disc_vector)

        if self.use_pqp and pqp_audio_input_values is not None:
            pqp_has_data = pqp_text_attention_mask is not None and pqp_text_attention_mask.sum() > 0
            if pqp_has_data and pqp_text_input_ids is not None:
                with torch.no_grad():
                    pqp_text_out = self.text_encoder(
                        input_ids=pqp_text_input_ids, attention_mask=pqp_text_attention_mask, return_dict=True)
                pqp_text_seq = self.pqp_text_proj(pqp_text_out.last_hidden_state)
                pqp_enc_kw = dict(input_values=pqp_audio_input_values, attention_mask=pqp_audio_attention_mask,
                                  output_hidden_states=True, return_dict=True)
                with torch.no_grad():
                    pqp_audio_out = self.audio_encoder(**pqp_enc_kw)
                pqp_hidden = torch.stack(pqp_audio_out.hidden_states, dim=0)
                pqp_audio_hidden = (pqp_hidden * w.view(-1, 1, 1, 1)).sum(dim=0)
                pqp_audio_seq = self.pqp_audio_proj(pqp_audio_hidden)
                pqp_reduced_mask = None
                if pqp_audio_attention_mask is not None:
                    pqp_reduced_mask = self._compute_reduced_audio_mask(
                        pqp_audio_attention_mask, pqp_audio_hidden.shape[1])
                pqp_audio_emb = self.pqp_audio_attn_pool(pqp_audio_seq, pqp_reduced_mask)
                if pqp_prosody_features is not None and pqp_prosody_features.abs().sum() > 0:
                    pqp_prosody_emb = self.pqp_prosody_mlp(pqp_prosody_features)
                else:
                    pqp_prosody_emb = torch.zeros_like(pqp_audio_emb)
                if pqp_frame_acoustic_features is not None and pqp_frame_acoustic_features.shape[1] > 0:
                    pqp_fa = self.pqp_frame_acoustic_conv(
                        pqp_frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
                    pqp_fa = self.pqp_frame_acoustic_norm(pqp_fa)
                    pqp_fa_emb = self.pqp_frame_attn_pool(pqp_fa, pqp_frame_acoustic_mask)
                else:
                    pqp_fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)
                pqp_fused = torch.cat([pqp_audio_emb, pqp_prosody_emb, pqp_text_seq.mean(1), sc_emb, pqp_fa_emb], dim=-1)
                sr_query = fused.unsqueeze(1)
                pqp_key_val = pqp_fused[:, :self.proj_dim].unsqueeze(1)
                context_emb = self.cross_attn(sr_query, pqp_key_val, None)
                context_proj = self.context_proj(context_emb)
                fused = fused + context_proj

        balance_loss = 0.0
        if self.use_moe:
            logits, balance_loss = self.moe_classifier(fused)
        else:
            logits = self.classifier(fused)

        output = {"logits": logits}
        if self.layer_head is not None:
            layer_logits = self.layer_head(fused)
            output["layer_logits"] = layer_logits
            if labels is not None:
                layer_ids = self.label_to_layer[labels]
                output["layer_loss"] = self.loss_fn(layer_logits, layer_ids)

        if labels is not None:
            loss = self.loss_fn(logits, labels) + 0.01 * balance_loss
            if self.layer_head is not None and "layer_loss" in output:
                loss = loss + self.layer_loss_weight * output["layer_loss"]
            output["loss"] = loss
        return output


# ===========================================================================
# CDD-Net: Consistency-Discrepancy Disentangled Network
# ===========================================================================

class GuidedDisentanglement(nn.Module):
    """Split embedding into consistency and discrepancy subspaces using disc_vector as gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.cons_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.disc_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, emb: torch.Tensor, disc_vector: torch.Tensor):
        g = self.gate(torch.cat([emb, disc_vector], dim=-1))
        consistent = self.cons_proj(emb * g)
        discrepant = self.disc_proj(emb * (1 - g))
        return consistent, discrepant


class ConsistencyStream(nn.Module):
    """MLP on concatenated text_cons and audio_cons -> (B, D)."""

    def __init__(self, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_cons: torch.Tensor, audio_cons: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([text_cons, audio_cons], dim=-1))


class DiscrepancyStream(nn.Module):
    """MLP on concat(text_disc, audio_disc, prosody, sc, fa, disc_vector) -> (B, D)."""

    def __init__(self, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 6, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_disc: torch.Tensor,
        audio_disc: torch.Tensor,
        prosody_emb: torch.Tensor,
        sc_emb: torch.Tensor,
        fa_emb: torch.Tensor,
        disc_vector: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([text_disc, audio_disc, prosody_emb, sc_emb, fa_emb, disc_vector], dim=-1)
        return self.mlp(x)


class AdaptiveCombiner(nn.Module):
    """alpha-weighted fusion of consistency and discrepancy streams.

    output_dim = proj_dim * 3 (alpha*cons, (1-alpha)*disc, disc_vector).
    """

    def __init__(self, proj_dim: int):
        super().__init__()
        self.proj_dim = proj_dim
        self.output_dim = proj_dim * 3
        self.alpha_mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2), nn.GELU(),
            nn.Linear(proj_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(
        self,
        consistency_out: torch.Tensor,
        discrepancy_out: torch.Tensor,
        disc_vector: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alpha_mlp(disc_vector)
        weighted_cons = alpha * consistency_out
        weighted_disc = (1 - alpha) * discrepancy_out
        return torch.cat([weighted_cons, weighted_disc, disc_vector], dim=-1)


class CDReconstructor(nn.Module):
    """Consistency-Discrepancy Reconstruction: cons + disc -> reconstruct original. Forces complementary split."""

    def __init__(self, proj_dim: int):
        super().__init__()
        self.recon = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(
        self,
        cons_out: torch.Tensor,
        disc_repr: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        recon = self.recon(torch.cat([cons_out, disc_repr], dim=-1))
        return F.mse_loss(recon, target.detach())


def supcon_loss(reprs: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss: same-class pull together, different-class push apart."""
    B = reprs.shape[0]
    if B < 2:
        return reprs.new_zeros(1)
    reprs = F.normalize(reprs, p=2, dim=1)
    logits = torch.mm(reprs, reprs.t()) / temperature
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask = mask.float()
    mask = mask * (1 - torch.eye(B, device=reprs.device, dtype=reprs.dtype))
    exp_logits = torch.exp(logits) * (1 - torch.eye(B, device=reprs.device, dtype=reprs.dtype))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    per_sample = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    return -per_sample.mean()


def discrepancy_guided_supcon_loss(
    reprs: torch.Tensor,
    labels: torch.Tensor,
    disc_scores: torch.Tensor,
    temperature: float = 0.07,
    disc_weight: float = 0.5,
) -> torch.Tensor:
    """Supervised contrastive with discrepancy-pattern weighting: same-class + similar disc pattern -> stronger pull."""
    B = reprs.shape[0]
    if B < 2:
        return reprs.new_zeros(1)
    reprs = F.normalize(reprs, p=2, dim=1)
    logits = torch.mm(reprs, reprs.t()) / temperature

    label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    label_mask = label_mask * (1 - torch.eye(B, device=reprs.device, dtype=reprs.dtype))

    if disc_scores.dim() == 1:
        disc_scores = disc_scores.unsqueeze(1)
    disc_norm = F.normalize(disc_scores.float(), p=2, dim=1)
    disc_sim = torch.mm(disc_norm, disc_norm.t())
    disc_sim = (disc_sim + 1) / 2

    combined_mask = label_mask * (1 - disc_weight + disc_weight * disc_sim.to(label_mask.dtype))

    exp_logits = torch.exp(logits) * (1 - torch.eye(B, device=reprs.device, dtype=reprs.dtype))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    per_sample = (combined_mask * log_prob).sum(1) / (combined_mask.sum(1) + 1e-8)
    return -per_sample.mean()


def dual_space_contrastive_loss(
    text_cons: torch.Tensor,
    audio_cons: torch.Tensor,
    cons_out: torch.Tensor,
    disc_repr: torch.Tensor,
) -> tuple:
    """DSCL: L_align (cross-modal cons alignment) + L_sep (inter-space separation)."""
    align_loss = (1 - F.cosine_similarity(text_cons, audio_cons, dim=1)).mean()
    sep_loss = F.cosine_similarity(cons_out, disc_repr, dim=1).pow(2).mean()
    return align_loss, sep_loss


class CDDPQPModel(nn.Module):
    """PQP CDD-Net: encoders + (CrossModalBridgeV2 TLDL or Bridge+GuidedDisentanglement) + dual streams + AdaptiveCombiner + classifier."""

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 2,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
        supcon_temperature: float = 0.07,
        use_token_disc: bool = True,
        use_dual_contrastive: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic
        self.supcon_temperature = supcon_temperature
        self.use_token_disc = use_token_disc
        self.use_dual_contrastive = use_dual_contrastive

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))

        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)
        self.audio_attn_pool = AttentionPooling(proj_dim)

        if use_token_disc:
            self.bridge_v2 = CrossModalBridgeV2(proj_dim, num_heads, dropout)
            self.bridge = None
            self.text_disent = None
            self.audio_disent = None
        else:
            self.bridge = CrossModalBridge(proj_dim, num_heads, dropout)
            self.bridge_v2 = None
            self.text_disent = GuidedDisentanglement(proj_dim)
            self.audio_disent = GuidedDisentanglement(proj_dim)
        self.cons_stream = ConsistencyStream(proj_dim, dropout)
        self.disc_stream = DiscrepancyStream(proj_dim, dropout)
        self.combiner = AdaptiveCombiner(proj_dim)

        combiner_out_dim = self.combiner.output_dim  # proj_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(combiner_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def freeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        self.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        B = attention_mask.shape[0]
        reduced = torch.zeros(B, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            reduced[i, :length.item()] = 1
        return reduced

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B = text_input_ids.shape[0]
        dev = text_input_ids.device

        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        audio_frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kw = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                      output_hidden_states=True, return_dict=True)
        if audio_frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kw)
        else:
            audio_out = self.audio_encoder(**enc_kw)
        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)
        reduced_mask = None
        if audio_attention_mask is not None:
            reduced_mask = self._compute_reduced_audio_mask(audio_attention_mask, audio_hidden.shape[1])
        audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        if self.use_token_disc:
            bridge_out = self.bridge_v2(
                text_seq, audio_seq, text_emb, audio_emb,
                text_mask=text_attention_mask, audio_mask=reduced_mask,
            )
            text_cons = bridge_out["text_cons"]
            text_disc = bridge_out["text_disc"]
            audio_cons = bridge_out["audio_cons"]
            audio_disc = bridge_out["audio_disc"]
            disc_vector = bridge_out["disc_vector"]
            text_disc_scores = bridge_out["text_disc_scores"]
            audio_disc_scores = bridge_out["audio_disc_scores"]
        else:
            _, _, disc_vector = self.bridge(
                text_seq, audio_seq, text_emb, audio_emb,
                text_mask=text_attention_mask, audio_mask=reduced_mask,
            )
            text_cons, text_disc = self.text_disent(text_emb, disc_vector)
            audio_cons, audio_disc = self.audio_disent(audio_emb, disc_vector)
            text_disc_scores = None
            audio_disc_scores = None

        if not self.use_prosody or prosody_features is None:
            prosody_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)

        if speechcraft_features is None:
            sc_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            sc_emb = self.sc_mlp(torch.cat([
                self.pitch_embed(speechcraft_features[:, 0]),
                self.energy_embed(speechcraft_features[:, 1]),
                self.speed_embed(speechcraft_features[:, 2]),
            ], dim=-1))

        if not self.use_frame_acoustic or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B, self.proj_dim, device=dev)
        else:
            fa = self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        cons_out = self.cons_stream(text_cons, audio_cons)
        disc_repr = self.disc_stream(text_disc, audio_disc, prosody_emb, sc_emb, fa_emb, disc_vector)
        fused = self.combiner(cons_out, disc_repr, disc_vector)
        logits = self.classifier(fused)

        output = {"logits": logits, "disc_repr": disc_repr}
        if text_disc_scores is not None:
            output["text_disc_scores"] = text_disc_scores
        if audio_disc_scores is not None:
            output["audio_disc_scores"] = audio_disc_scores
        ortho_loss = (
            (text_cons * text_disc).sum(dim=-1).pow(2).mean()
            + (audio_cons * audio_disc).sum(dim=-1).pow(2).mean()
        )
        output["ortho_loss"] = ortho_loss
        if self.use_dual_contrastive:
            align_loss, sep_loss = dual_space_contrastive_loss(text_cons, audio_cons, cons_out, disc_repr)
            output["align_loss"] = align_loss
            output["sep_loss"] = sep_loss
        else:
            output["align_loss"] = disc_repr.new_zeros(1)
            output["sep_loss"] = disc_repr.new_zeros(1)
        if labels is not None:
            output["contrastive_loss"] = supcon_loss(disc_repr, labels, self.supcon_temperature)
            ce_loss = self.loss_fn(logits, labels)
            output["loss"] = ce_loss
        else:
            output["contrastive_loss"] = disc_repr.new_zeros(1)
        return output


class CDDSRMoEModel(nn.Module):
    """SR CDD-Net: encoders + (BridgeV2 TLDL or Bridge+GuidedDisentanglement) + dual streams + combiner + PQP context + MOE + layer head."""

    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        audio_model_name: str = "TencentGameMate/chinese-wav2vec2-base",
        num_labels: int = 14,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        label_smoothing: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        use_pqp: bool = True,
        use_moe: bool = True,
        use_prosody: bool = True,
        use_frame_acoustic: bool = True,
        use_hierarchical: bool = False,
        layer_loss_weight: float = 0.3,
        supcon_temperature: float = 0.07,
        use_token_disc: bool = True,
        use_dual_contrastive: bool = True,
        use_dgcp: bool = True,
        use_rfr: bool = True,
        rfr_gate_tau: float = 1.0,
        rfr_beta_init: float = 1.0,
        use_tone_aware_tldl: bool = False,
        tone_mask_gamma: float = 0.5,
        tone_mask_temp: float = 1.0,
        tone_var_dim: int = 1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_pqp = use_pqp
        self.use_moe = use_moe
        self.use_prosody = use_prosody
        self.use_frame_acoustic = use_frame_acoustic
        self.use_hierarchical = use_hierarchical
        self.layer_loss_weight = layer_loss_weight
        self.supcon_temperature = supcon_temperature
        self.use_token_disc = use_token_disc
        self.use_dual_contrastive = use_dual_contrastive
        self.use_dgcp = use_dgcp
        self.use_rfr = use_rfr
        self.rfr_gate_tau = rfr_gate_tau
        self.use_tone_aware_tldl = use_tone_aware_tldl
        self.tone_mask_gamma = tone_mask_gamma
        self.tone_mask_temp = tone_mask_temp
        self.tone_var_dim = tone_var_dim

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.freeze_module(self.text_encoder)
        if freeze_audio_encoder:
            self.freeze_module(self.audio_encoder)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        num_audio_layers = self.audio_encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_audio_layers))

        self.text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.pqp_text_proj = nn.Sequential(nn.Linear(text_hidden, proj_dim), nn.LayerNorm(proj_dim))
        self.pqp_audio_proj = nn.Sequential(nn.Linear(audio_hidden, proj_dim), nn.LayerNorm(proj_dim))

        self.prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        self.pqp_prosody_mlp = nn.Sequential(
            nn.LayerNorm(PROSODY_FEAT_DIM),
            nn.Linear(PROSODY_FEAT_DIM, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim),
        )
        sc_embed_dim = 16
        self.pitch_embed = nn.Embedding(3, sc_embed_dim)
        self.energy_embed = nn.Embedding(3, sc_embed_dim)
        self.speed_embed = nn.Embedding(3, sc_embed_dim)
        self.sc_mlp = nn.Sequential(nn.Linear(sc_embed_dim * 3, proj_dim), nn.GELU(), nn.Dropout(dropout))

        self.frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.frame_attn_pool = AttentionPooling(proj_dim)
        self.pqp_frame_acoustic_conv = nn.Sequential(
            nn.Conv1d(FRAME_ACOUSTIC_DIM, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, proj_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.pqp_frame_acoustic_norm = nn.LayerNorm(proj_dim)
        self.pqp_frame_attn_pool = AttentionPooling(proj_dim)
        self.audio_attn_pool = AttentionPooling(proj_dim)
        self.pqp_audio_attn_pool = AttentionPooling(proj_dim)

        if use_token_disc:
            self.bridge_v2 = CrossModalBridgeV2(
                proj_dim,
                num_heads,
                dropout,
                use_tone_aware_mask=use_tone_aware_tldl,
                tone_mask_gamma=tone_mask_gamma,
                tone_mask_temp=tone_mask_temp,
                tone_var_dim=tone_var_dim,
            )
            self.bridge = None
            self.text_disent = None
            self.audio_disent = None
        else:
            self.bridge = CrossModalBridge(proj_dim, num_heads, dropout)
            self.bridge_v2 = None
            self.text_disent = GuidedDisentanglement(proj_dim)
            self.audio_disent = GuidedDisentanglement(proj_dim)
        self.cons_stream = ConsistencyStream(proj_dim, dropout)
        self.disc_stream = DiscrepancyStream(proj_dim, dropout)
        self.combiner = AdaptiveCombiner(proj_dim)
        self.cd_reconstructor = CDReconstructor(proj_dim)

        combiner_out_dim = self.combiner.output_dim  # proj_dim * 3
        self.combiner_proj = nn.Sequential(
            nn.Linear(combiner_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.rfr_proj = nn.Sequential(
            nn.Linear(proj_dim * 5, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rfr_gate_linear = nn.Linear(proj_dim * 2, 1)
        self.rfr_beta = nn.Parameter(torch.tensor(float(rfr_beta_init)))

        self.cross_attn = PQPContextAttention(proj_dim, num_heads)
        self.context_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
        )
        if use_moe:
            self.moe_classifier = MOEClassifier(proj_dim, num_experts=4, top_k=2, dropout=dropout)
            self.simple_classifier = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2), nn.LayerNorm(proj_dim // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(proj_dim // 2, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2), nn.LayerNorm(proj_dim // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(proj_dim // 2, num_labels),
            )
        self.register_buffer("label_to_layer", torch.tensor(SR_LABEL_ID_TO_LAYER_ID, dtype=torch.long))
        if use_hierarchical:
            self.layer_head = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(proj_dim // 2, SR_NUM_LAYERS),
            )
        else:
            self.layer_head = None
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def freeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @staticmethod
    def unfreeze_module(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = True

    def partial_unfreeze_audio(self, num_layers: int = 4):
        self.freeze_module(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - num_layers, total):
            self.unfreeze_module(self.audio_encoder.encoder.layers[i])

    def set_freeze(self, freeze_text: bool, freeze_audio: bool):
        self.freeze_module(self.text_encoder)
        if freeze_audio:
            self.freeze_module(self.audio_encoder)
        else:
            self.partial_unfreeze_audio(num_layers=4)

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _compute_reduced_audio_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=-1).long()
        output_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        B = attention_mask.shape[0]
        reduced = torch.zeros(B, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            reduced[i, :length.item()] = 1
        return reduced

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        speechcraft_features: Optional[torch.Tensor] = None,
        frame_acoustic_features: Optional[torch.Tensor] = None,
        frame_acoustic_mask: Optional[torch.Tensor] = None,
        pqp_text_input_ids: Optional[torch.Tensor] = None,
        pqp_text_attention_mask: Optional[torch.Tensor] = None,
        pqp_audio_input_values: Optional[torch.Tensor] = None,
        pqp_audio_attention_mask: Optional[torch.Tensor] = None,
        pqp_prosody_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_features: Optional[torch.Tensor] = None,
        pqp_frame_acoustic_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B_size = text_input_ids.shape[0]
        dev = text_input_ids.device

        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_emb = self._masked_mean_pool(text_seq, text_attention_mask)

        audio_frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kw = dict(input_values=audio_input_values, attention_mask=audio_attention_mask,
                      output_hidden_states=True, return_dict=True)
        if audio_frozen:
            with torch.no_grad():
                audio_out = self.audio_encoder(**enc_kw)
        else:
            audio_out = self.audio_encoder(**enc_kw)
        hidden_states = torch.stack(audio_out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0)
        audio_hidden = (hidden_states * w.view(-1, 1, 1, 1)).sum(dim=0)
        audio_seq = self.audio_proj(audio_hidden)
        reduced_mask = None
        if audio_attention_mask is not None:
            reduced_mask = self._compute_reduced_audio_mask(audio_attention_mask, audio_hidden.shape[1])
        audio_emb = self.audio_attn_pool(audio_seq, reduced_mask)

        tone_var = None
        if self.use_tone_aware_tldl and prosody_features is not None and prosody_features.dim() == 2:
            tone_var = prosody_features[:, self.tone_var_dim]

        if self.use_token_disc:
            bridge_out = self.bridge_v2(
                text_seq, audio_seq, text_emb, audio_emb,
                text_mask=text_attention_mask, audio_mask=reduced_mask,
                tone_var=tone_var,
            )
            text_cons = bridge_out["text_cons"]
            text_disc = bridge_out["text_disc"]
            audio_cons = bridge_out["audio_cons"]
            audio_disc = bridge_out["audio_disc"]
            disc_vector = bridge_out["disc_vector"]
            text_disc_scores = bridge_out["text_disc_scores"]
            audio_disc_scores = bridge_out["audio_disc_scores"]
        else:
            _, _, disc_vector = self.bridge(
                text_seq, audio_seq, text_emb, audio_emb,
                text_mask=text_attention_mask, audio_mask=reduced_mask,
            )
            text_cons, text_disc = self.text_disent(text_emb, disc_vector)
            audio_cons, audio_disc = self.audio_disent(audio_emb, disc_vector)
            text_disc_scores = None
            audio_disc_scores = None

        if not self.use_prosody or prosody_features is None:
            prosody_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            prosody_emb = self.prosody_mlp(prosody_features)

        if speechcraft_features is not None and speechcraft_features.abs().sum() > 0:
            sc_emb = self.sc_mlp(torch.cat([
                self.pitch_embed(speechcraft_features[:, 0]),
                self.energy_embed(speechcraft_features[:, 1]),
                self.speed_embed(speechcraft_features[:, 2]),
            ], dim=-1))
        else:
            sc_emb = torch.zeros(B_size, self.proj_dim, device=dev)

        if not self.use_frame_acoustic or frame_acoustic_features is None or frame_acoustic_features.shape[1] == 0:
            fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)
        else:
            fa = self.frame_acoustic_conv(frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
            fa = self.frame_acoustic_norm(fa)
            fa_emb = self.frame_attn_pool(fa, frame_acoustic_mask)

        cons_out = self.cons_stream(text_cons, audio_cons)
        disc_repr = self.disc_stream(text_disc, audio_disc, prosody_emb, sc_emb, fa_emb, disc_vector)
        recon_target = (text_emb + audio_emb) / 2
        recon_loss = self.cd_reconstructor(cons_out, disc_repr, recon_target)
        fused_wide = self.combiner(cons_out, disc_repr, disc_vector)
        cdd_fused = self.combiner_proj(fused_wide)

        raw_cat = torch.cat([audio_emb, prosody_emb, text_emb, sc_emb, fa_emb], dim=-1)
        rfr_fused = self.rfr_proj(raw_cat)
        if self.use_rfr:
            gate_input = torch.cat([cdd_fused, rfr_fused], dim=-1)
            gate_logit = self.rfr_gate_linear(gate_input)
            g = torch.sigmoid(gate_logit / self.rfr_gate_tau)
            beta = torch.clamp(self.rfr_beta, 0.1, 2.0)
            fused = g * cdd_fused + (1 - g) * (beta * rfr_fused)
        else:
            fused = cdd_fused

        if self.use_pqp and pqp_audio_input_values is not None:
            pqp_has_data = pqp_text_attention_mask is not None and pqp_text_attention_mask.sum() > 0
            if pqp_has_data and pqp_text_input_ids is not None:
                with torch.no_grad():
                    pqp_text_out = self.text_encoder(
                        input_ids=pqp_text_input_ids, attention_mask=pqp_text_attention_mask, return_dict=True)
                pqp_text_seq = self.pqp_text_proj(pqp_text_out.last_hidden_state)
                pqp_enc_kw = dict(input_values=pqp_audio_input_values, attention_mask=pqp_audio_attention_mask,
                                  output_hidden_states=True, return_dict=True)
                with torch.no_grad():
                    pqp_audio_out = self.audio_encoder(**pqp_enc_kw)
                pqp_hidden = torch.stack(pqp_audio_out.hidden_states, dim=0)
                pqp_audio_hidden = (pqp_hidden * w.view(-1, 1, 1, 1)).sum(dim=0)
                pqp_audio_seq = self.pqp_audio_proj(pqp_audio_hidden)
                pqp_reduced_mask = None
                if pqp_audio_attention_mask is not None:
                    pqp_reduced_mask = self._compute_reduced_audio_mask(
                        pqp_audio_attention_mask, pqp_audio_hidden.shape[1])
                pqp_audio_emb = self.pqp_audio_attn_pool(pqp_audio_seq, pqp_reduced_mask)
                if pqp_prosody_features is not None and pqp_prosody_features.abs().sum() > 0:
                    pqp_prosody_emb = self.pqp_prosody_mlp(pqp_prosody_features)
                else:
                    pqp_prosody_emb = torch.zeros_like(pqp_audio_emb)
                if pqp_frame_acoustic_features is not None and pqp_frame_acoustic_features.shape[1] > 0:
                    pqp_fa = self.pqp_frame_acoustic_conv(
                        pqp_frame_acoustic_features.transpose(1, 2)).transpose(1, 2)
                    pqp_fa = self.pqp_frame_acoustic_norm(pqp_fa)
                    pqp_fa_emb = self.pqp_frame_attn_pool(pqp_fa, pqp_frame_acoustic_mask)
                else:
                    pqp_fa_emb = torch.zeros(B_size, self.proj_dim, device=dev)
                pqp_fused = torch.cat([pqp_audio_emb, pqp_prosody_emb, pqp_text_seq.mean(1), sc_emb, pqp_fa_emb], dim=-1)
                sr_query = fused.unsqueeze(1)
                pqp_key_val = pqp_fused[:, :self.proj_dim].unsqueeze(1)
                context_emb = self.cross_attn(sr_query, pqp_key_val, None)
                context_proj = self.context_proj(context_emb)
                fused = fused + context_proj

        balance_loss = 0.0
        if self.use_moe:
            logits, balance_loss = self.moe_classifier(fused)
        else:
            logits = self.classifier(fused)

        output = {"logits": logits, "disc_repr": disc_repr, "recon_loss": recon_loss}
        if text_disc_scores is not None:
            output["text_disc_scores"] = text_disc_scores
        if audio_disc_scores is not None:
            output["audio_disc_scores"] = audio_disc_scores
        ortho_loss = (
            (text_cons * text_disc).sum(dim=-1).pow(2).mean()
            + (audio_cons * audio_disc).sum(dim=-1).pow(2).mean()
        )
        output["ortho_loss"] = ortho_loss
        if self.use_dual_contrastive:
            align_loss, sep_loss = dual_space_contrastive_loss(
                text_cons, audio_cons,
                cons_out, disc_repr,
            )
            output["align_loss"] = align_loss
            output["sep_loss"] = sep_loss
        else:
            output["align_loss"] = disc_repr.new_zeros(1)
            output["sep_loss"] = disc_repr.new_zeros(1)
        if labels is not None:
            if self.use_dgcp:
                disc_scores_for_con = (
                    text_disc_scores.detach()
                    if text_disc_scores is not None
                    else disc_repr.new_zeros(disc_repr.shape[0], 1)
                )
                output["contrastive_loss"] = discrepancy_guided_supcon_loss(
                    disc_repr,
                    labels,
                    disc_scores_for_con,
                    self.supcon_temperature,
                    disc_weight=0.5,
                )
            else:
                output["contrastive_loss"] = supcon_loss(disc_repr, labels, self.supcon_temperature)
        else:
            output["contrastive_loss"] = disc_repr.new_zeros(1)

        if self.layer_head is not None:
            layer_logits = self.layer_head(fused)
            output["layer_logits"] = layer_logits
            if labels is not None:
                layer_ids = self.label_to_layer[labels]
                output["layer_loss"] = self.loss_fn(layer_logits, layer_ids)

        if labels is not None:
            ce_loss = self.loss_fn(logits, labels) + 0.01 * balance_loss
            if self.layer_head is not None and "layer_loss" in output:
                ce_loss = ce_loss + self.layer_loss_weight * output["layer_loss"]
            output["loss"] = ce_loss
        return output
