"""
test_regression_fixes.py
========================
Verification suite for DeepfakeReasoningModel v2 performance fixes:
  1. Module syntax & import verification
  2. Construction with attn_implementation & use_gradient_checkpointing
  3. Single-pass vision feature reuse numerical parity check
  4. Forward & Backward joint-stage pass check
  5. State dict compatibility & strict loading check
  6. CLI argument parser validation in train.py and train_dpo.py
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Test imports
import model as model_mod
from model import (
    DeepfakeReasoningModel,
    AttentionPool,
    FrequencyBranch,
    ClassificationHead,
    CustomProjector,
)
import train as train_mod
import train_dpo as train_dpo_mod


class MockVisionOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockLMOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class MockVisionModel(nn.Module):
    def __init__(self, vision_dim=64):
        super().__init__()
        self.vision_dim = vision_dim
        self.qkv = nn.Linear(vision_dim, vision_dim)
        self.proj = nn.Linear(vision_dim, vision_dim)
        self.fc1 = nn.Linear(vision_dim, vision_dim)
        self.fc2 = nn.Linear(vision_dim, vision_dim)

    def forward(self, pixel_values):
        B = pixel_values.size(0)
        N = 16  # 16 patches + 1 CLS token = 17
        hs = torch.randn(B, N + 1, self.vision_dim, dtype=torch.bfloat16, device=pixel_values.device)
        return MockVisionOutput(hs)


class MockLanguageModel(nn.Module):
    def __init__(self, llm_dim=128, vocab_size=1000):
        super().__init__()
        self.llm_dim = llm_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, llm_dim)
        self.q_proj = nn.Linear(llm_dim, llm_dim)
        self.lm_head = nn.Linear(llm_dim, vocab_size)
        self.config = MagicMock()
        self.config._attn_implementation = "sdpa"
        self.gradient_checkpointing_enabled = False
        self.input_require_grads_enabled = False

    def get_input_embeddings(self):
        return self.embed

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed(input_ids)
        B, seq_len, _ = inputs_embeds.shape
        logits = torch.randn(B, seq_len, self.vocab_size, dtype=torch.bfloat16, device=inputs_embeds.device, requires_grad=True)
        loss = logits.sum() * 0.01
        return MockLMOutput(loss, logits)


class MockBackbone(nn.Module):
    def __init__(self, vision_dim=64, llm_dim=128, vocab_size=1000, has_mlp1=True):
        super().__init__()
        self.vision_model = MockVisionModel(vision_dim)
        self.language_model = MockLanguageModel(llm_dim, vocab_size)
        self.config = MagicMock()
        self.config.vision_config.hidden_size = vision_dim
        self.config.llm_config.hidden_size = llm_dim
        self.config.llm_config.vocab_size = vocab_size
        self.config._attn_implementation = "sdpa"
        self.downsample_ratio = 0.5

        if has_mlp1:
            self.mlp1 = nn.Sequential(
                nn.Linear(vision_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim)
            )

    def pixel_shuffle(self, hs, scale_factor=0.5):
        return hs


def build_mock_v2_model(has_mlp1=True, attn_impl="sdpa", use_grad_ckpt=True):
    mock_bb = MockBackbone(vision_dim=64, llm_dim=128, vocab_size=1000, has_mlp1=has_mlp1)
    with patch("model.AutoModel.from_pretrained", return_value=mock_bb):
        model = DeepfakeReasoningModel(
            model_path="dummy_path",
            lora_rank=8,
            vision_lora_rank=4,
            arch_version="v2",
            attn_implementation=attn_impl,
            use_gradient_checkpointing=use_grad_ckpt,
            real_token_id=10,
            fake_token_id=20,
        )
    return model


class TestRegressionFixes(unittest.TestCase):

    def test_01_model_init_and_attributes(self):
        """Verify model initializes with new performance parameters."""
        model = build_mock_v2_model(attn_impl="sdpa", use_grad_ckpt=True)
        self.assertEqual(model.attn_implementation, "sdpa")
        self.assertTrue(model.use_gradient_checkpointing)
        self.assertEqual(model._attn_implementation_resolved, "sdpa")
        self.assertTrue(model.backbone.language_model.gradient_checkpointing_enabled)
        self.assertTrue(model.backbone.language_model.input_require_grads_enabled)
        print("  [Pass] Model initialized correctly with SDPA and Gradient Checkpointing.")

    def test_02_gradient_checkpointing_toggle(self):
        """Verify gradient checkpointing can be toggled off."""
        model = build_mock_v2_model(attn_impl="sdpa", use_grad_ckpt=False)
        self.assertFalse(model.use_gradient_checkpointing)
        self.assertFalse(model.backbone.language_model.gradient_checkpointing_enabled)
        print("  [Pass] Gradient checkpointing toggle verified.")

    def test_03_native_visual_tokens_patch_reuse_parity(self):
        """Verify _get_native_visual_tokens_from_patches produces identical tokens to manual pass."""
        model = build_mock_v2_model(has_mlp1=True)
        B, N, D = 2, 16, 64
        patch_tokens = torch.randn(B, N, D, dtype=torch.bfloat16)
        
        # Output via patch reuse method
        out_from_patches = model._get_native_visual_tokens_from_patches(patch_tokens)
        
        # Ground truth: direct pixel_shuffle + mlp1
        hs = model.backbone.pixel_shuffle(patch_tokens, scale_factor=model.backbone.downsample_ratio)
        mlp1_dtype = next(model.backbone.mlp1.parameters()).dtype
        expected = model.backbone.mlp1(hs.to(mlp1_dtype))
        
        self.assertEqual(out_from_patches.shape, expected.shape)
        self.assertTrue(torch.allclose(out_from_patches.float(), expected.float(), atol=1e-5))
        print("  [Pass] Numerical parity between patch token reuse and direct mlp1 verified.")

    def test_04_fallback_mock_patch_reuse_parity(self):
        """Verify fallback path without mlp1 works seamlessly."""
        model = build_mock_v2_model(has_mlp1=False)
        model.eval()
        B, N, D = 2, 16, 64
        cls_token = torch.randn(B, D, dtype=torch.bfloat16)
        patch_tokens = torch.randn(B, N, D, dtype=torch.bfloat16)
        
        out = model._get_native_visual_tokens_from_patches(patch_tokens, cls_token=cls_token)
        expected = model.custom_projector(cls_token.float(), patch_tokens.mean(dim=1).float())
        
        self.assertEqual(out.shape, (B, 2, model.llm_dim))
        self.assertTrue(torch.allclose(out.float(), expected.float(), atol=1e-5))
        print("  [Pass] Fallback custom projector patch reuse parity verified.")

    def test_05_forward_and_backward_joint_pass(self):
        """Verify joint forward & backward pass executes cleanly without runtime errors."""
        model = build_mock_v2_model(has_mlp1=True, use_grad_ckpt=True)
        B = 2
        pixel_values = torch.randn(B, 3, 448, 448)
        labels = torch.tensor([0, 1], dtype=torch.long)
        reasoning_tokens = torch.tensor([[1, 2, 10, 3], [1, 2, 20, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(reasoning_tokens)
        freq_features = torch.randn(B, 3, 224, 224)
        answer_token_pos = torch.tensor([2, 2], dtype=torch.long)

        out = model(
            pixel_values=pixel_values,
            labels=labels,
            reasoning_tokens=reasoning_tokens,
            reasoning_attention_mask=attention_mask,
            freq_features=freq_features,
            answer_token_pos=answer_token_pos,
        )

        self.assertIn("cls_logits", out)
        self.assertIn("loss", out)
        self.assertIn("cls_loss", out)
        self.assertIn("lm_loss", out)
        self.assertIn("consistency_loss", out)
        self.assertIsNotNone(out["loss"])
        self.assertIsNotNone(out["lm_loss"])

        # Backward pass check
        out["loss"].backward()
        
        # Verify gradients exist on trainable parameters
        trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertGreater(len(trainable_grads), 0)
        print("  [Pass] Joint forward & backward pass verified with valid gradients.")

    def test_06_state_dict_compatibility(self):
        """Verify state_dict keys and shapes are 100% compatible for checkpoint resumption."""
        model1 = build_mock_v2_model(attn_impl="sdpa", use_grad_ckpt=True)
        model2 = build_mock_v2_model(attn_impl="eager", use_grad_ckpt=False)

        sd1 = model1.state_dict()
        sd2 = model2.state_dict()

        self.assertEqual(sd1.keys(), sd2.keys())
        for k in sd1:
            self.assertEqual(sd1[k].shape, sd2[k].shape, f"Shape mismatch on key {k}")

        # Test strict load
        model2.load_state_dict(sd1, strict=True)
        print("  [Pass] State dict strict=True checkpoint compatibility verified.")

    def test_07_cli_arguments_train_and_dpo(self):
        """Verify train.py, train_dpo.py, eval.py, and inference.py argument parsers accept new acceleration flags."""
        with patch("sys.argv", ["train.py", "--attn_implementation", "sdpa", "--no_gradient_checkpointing"]):
            args = train_mod.parse_args()
            self.assertEqual(args.attn_implementation, "sdpa")
            self.assertTrue(args.no_gradient_checkpointing)

        with patch("sys.argv", ["train_dpo.py", "--dataset_root", "dummy", "--json_root", "dummy", "--checkpoint", "dummy.pth", "--attn_implementation", "sdpa"]):
            dpo_args = train_dpo_mod.parse_args()
            self.assertEqual(dpo_args.attn_implementation, "sdpa")
            self.assertTrue(dpo_args.use_gradient_checkpointing)

        import eval as eval_mod
        with patch("sys.argv", ["eval.py", "--checkpoint", "dummy.pth", "--attn_implementation", "sdpa"]):
            eval_args = eval_mod.parse_args()
            self.assertEqual(eval_args.attn_implementation, "sdpa")

        import inference as inf_mod
        with patch("sys.argv", ["inference.py", "--attn_implementation", "sdpa"]):
            inf_args = inf_mod.parse_args()
            self.assertEqual(inf_args.attn_implementation, "sdpa")

        print("  [Pass] CLI argument parsers verified for train.py, train_dpo.py, eval.py, and inference.py.")

    def test_08_4d_pixel_shuffle_internvl_convention(self):
        """Verify _apply_pixel_shuffle_and_mlp1 correctly handles standard 4D (B, H, W, C) pixel_shuffle."""
        model = build_mock_v2_model(has_mlp1=True)
        # Mock 4D pixel_shuffle (standard OpenGVLab / HuggingFace InternVL behavior)
        def mock_4d_pixel_shuffle(x, scale_factor=0.5):
            B, H, W, C = x.shape
            # Downsample H, W by 0.5 -> (B, H/2, W/2, C*4)
            return x.view(B, int(H * scale_factor), int(W * scale_factor), int(C / (scale_factor ** 2)))

        model.backbone.pixel_shuffle = mock_4d_pixel_shuffle
        model.backbone.downsample_ratio = 0.5
        # 16 patches = 4x4 grid, C = 64 -> pixel_shuffle returns (B, 2, 2, 256) -> flattened to (B, 4, 256)
        # We need mlp1 to map 256 -> llm_dim
        model.backbone.mlp1 = nn.Linear(256, model.llm_dim)

        B, N, D = 2, 16, 64
        patch_tokens = torch.randn(B, N, D, dtype=torch.bfloat16)

        out = model._get_native_visual_tokens_from_patches(patch_tokens)
        self.assertEqual(out.shape, (B, 4, model.llm_dim))
        print("  [Pass] Standard 4D pixel_shuffle (InternVL convention) handled successfully.")


if __name__ == "__main__":
    unittest.main()
