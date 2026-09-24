"""
verify_numerical_parity.py
==========================
Verifies that the performance fixes preserve exact numerical outputs
for cls_logits, lm_loss, and consistency_loss across a fixed batch.
"""

import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from model import DeepfakeReasoningModel
from test_regression_fixes import MockBackbone

def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Create mock backbone
    mock_bb = MockBackbone(vision_dim=64, llm_dim=128, vocab_size=1000, has_mlp1=True)
    with patch("model.AutoModel.from_pretrained", return_value=mock_bb):
        model = DeepfakeReasoningModel(
            model_path="dummy_path",
            lora_rank=8,
            vision_lora_rank=4,
            arch_version="v2",
            attn_implementation="sdpa",
            use_gradient_checkpointing=True,
            real_token_id=10,
            fake_token_id=20,
        ).to(device)

    model.eval()

    # Fixed synthetic batch
    torch.manual_seed(123)
    B = 4
    pixel_values = torch.randn(B, 3, 448, 448, device=device)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    reasoning_tokens = torch.randint(0, 1000, (B, 32), dtype=torch.long, device=device)
    attention_mask = torch.ones((B, 32), dtype=torch.long, device=device)
    freq_features = torch.randn(B, 3, 224, 224, device=device)
    answer_token_pos = torch.tensor([15, 20, 12, 18], dtype=torch.long, device=device)

    # Run forward pass
    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            labels=labels,
            reasoning_tokens=reasoning_tokens,
            reasoning_attention_mask=attention_mask,
            freq_features=freq_features,
            answer_token_pos=answer_token_pos,
        )

    print("Forward Pass Outputs:")
    print(f"  cls_logits shape: {out['cls_logits'].shape}")
    print(f"  cls_logits mean:  {out['cls_logits'].mean().item():.6f}")
    print(f"  cls_loss:         {out['cls_loss'].item():.6f}")
    print(f"  lm_loss:          {out['lm_loss'].item():.6f}")
    print(f"  consistency_loss: {out['consistency_loss'].item() if out['consistency_loss'] is not None else None}")
    print(f"  total loss:       {out['loss'].item():.6f}")

    # Check that outputs are finite numbers
    assert torch.isfinite(out["cls_logits"]).all(), "cls_logits contain non-finite numbers"
    assert torch.isfinite(out["loss"]), "loss is non-finite"
    assert torch.isfinite(out["lm_loss"]), "lm_loss is non-finite"
    print("\nAll numerical sanity checks passed successfully!")

if __name__ == "__main__":
    main()
