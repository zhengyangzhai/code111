"""Quick smoke test: verify CCMT model forward pass and output format.
Run: python test_ccmt_smoke.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from model_sr_ccmt import CCMTSRModel

def main():
    model = CCMTSRModel(
        text_model_name="bert-base-chinese",
        audio_model_name="TencentGameMate/chinese-wav2vec2-base",
        num_labels=14,
        proj_dim=64,
        num_heads=2,
        dropout=0.1,
        num_text_self_layers=1,
        num_cross_layers=1,
    )
    B, L = 2, 32
    out = model(
        text_input_ids=torch.randint(0, 1000, (B, L)),
        text_attention_mask=torch.ones(B, L),
        audio_input_values=torch.randn(B, 16000) * 0.1,
        audio_attention_mask=torch.ones(B, 100),
        labels=torch.randint(0, 14, (B,)),
    )
    assert "logits" in out and "loss" in out
    assert out["logits"].shape == (B, 14)
    assert "contrastive_loss" not in out and "recon_loss" not in out
    print("[OK] CCMTSRModel forward: logits + loss only (no CDD-specific keys)")

if __name__ == "__main__":
    main()
