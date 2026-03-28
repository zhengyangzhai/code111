"""model_sr_ccmt.py — CCMT-style cascaded cross-modal transformer for SR 14-class

CCMT (Cascaded Cross-Modal Transformer) reproduced under our SR setting.
Structure: text self-attention → cascaded cross-modal (audio attends to text) → pool → classifier.
Uses BERT + Wav2Vec2 encoders; text + audio only (no prosody/SC/FA in backbone).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model

from model import CrossModalTransformerLayer, MultiModalPQPModel


def _masked_mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


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


class CCMTSRModel(nn.Module):
    """CCMT-style cascaded cross-modal transformer for SR 14-class.

    Stage 1: Text self-attention → text_context
    Stage 2: Audio attends to text_context (cascaded cross-modal)
    Stage 3: Pool fused representation → 14-class classifier

    Output: logits, loss (no contrastive/recon/CDD-specific losses).
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
        num_text_self_layers: int = 2,
        num_cross_layers: int = 2,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # ---- Encoders ----
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

        # ---- Stage 1: Text self-attention (context modeling) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.text_self_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_text_self_layers)

        # ---- Stage 2: Cascaded cross-modal (audio attends to text) ----
        self.audio_with_text = nn.ModuleList([
            CrossModalTransformerLayer(proj_dim, num_heads, dropout) for _ in range(num_cross_layers)
        ])

        # ---- Stage 3: Pooling + classifier ----
        self.audio_attn_pool = AttentionPooling(proj_dim)
        # Concatenate text_emb + audio_emb (both after cross-modal flow)
        fusion_dim = proj_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(proj_dim, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _compute_reduced_audio_mask(self, attention_mask: torch.Tensor, hidden_seq_len: int) -> torch.Tensor:
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
        # Accept batch fields for compatibility; CCMT ignores prosody/SC/FA in backbone.
        B = text_input_ids.shape[0]
        dev = text_input_ids.device

        # 1. Text sequence
        with torch.no_grad():
            text_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        text_seq = self.text_proj(text_out.last_hidden_state)  # (B, L_text, D)

        # 2. Text self-attention → text_context
        # TransformerEncoder expects (B, L, D); need to pass src_key_padding_mask for padding
        # key_padding_mask: True = ignore
        text_pad = ~text_attention_mask.bool() if text_attention_mask is not None else None
        text_context = self.text_self_transformer(text_seq, src_key_padding_mask=text_pad)  # (B, L_text, D)

        # 3. Audio sequence
        frozen = not any(p.requires_grad for p in self.audio_encoder.parameters())
        enc_kwargs = dict(
            input_values=audio_input_values,
            attention_mask=audio_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
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
            reduced_mask = self._compute_reduced_audio_mask(audio_attention_mask, audio_hidden.shape[1])
        else:
            reduced_mask = None

        # 4. Cascaded cross-modal: audio attends to text_context
        fused_audio = audio_seq
        for layer in self.audio_with_text:
            fused_audio = layer(fused_audio, text_context, reduced_mask, text_attention_mask)

        # 5. Pool
        text_emb = _masked_mean_pool(text_context, text_attention_mask)  # (B, D)
        audio_emb = self.audio_attn_pool(fused_audio, reduced_mask)  # (B, D)

        # 6. Classify (concat text + audio)
        fused = torch.cat([text_emb, audio_emb], dim=-1)
        logits = self.classifier(fused)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output
