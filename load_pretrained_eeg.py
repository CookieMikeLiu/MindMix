#!/usr/bin/env python3
"""
Load Pre-trained MindMix EEG Encoder

This script shows the exact model class and key-mapping logic needed to
instantiate the EEG encoder from the released MindMix fusion checkpoint.

Usage:
    python load_pretrained_eeg.py
    python load_pretrained_eeg.py --checkpoint pretrain_fusion_checkpoints/best_model_loss_0.0909.pth
"""

import argparse
import os

import torch
from einops import rearrange

from modeling_finetune_2 import labram_base_patch200_200


def build_mindmix_eeg_encoder(checkpoint_path: str, device: str = "cpu"):
    """
    Build the MindMix EEG encoder from a fusion pre-training checkpoint.

    Args:
        checkpoint_path: Path to the MindMix checkpoint, e.g.
            ``pretrain_fusion_checkpoints/best_model_loss_0.0909.pth``.
        device: Device to place the model on (``"cpu"`` or ``"cuda"``).

    Returns:
        nn.Module: EEG encoder that accepts ``[B, n_channels, 400]`` and
        outputs ``[B, 200]`` features.
    """
    # ------------------------------------------------------------------
    # 1. Instantiate the EEG backbone (NeuralTransformer)
    # ------------------------------------------------------------------
    # These hyper-parameters must match the ones used during MindMix
    # pre-training (see MindMix_clip_pretrain.py).
    backbone = labram_base_patch200_200(
        pretrained=False,
        num_classes=0,          # Remove the classification head
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        use_mean_pooling=True,  # Output: [B, 200]
        use_rel_pos_bias=True,
        use_abs_pos_emb=True,
        qkv_bias=True,
    )
    # Replace the head with Identity so we get pure features
    backbone.head = torch.nn.Identity()

    # ------------------------------------------------------------------
    # 2. Load the checkpoint
    # ------------------------------------------------------------------
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please ensure you have cloned the repository with Git LFS:\n"
            "  git lfs pull"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # The fusion script saves the whole CLIPModel under this key
    state_dict = ckpt.get("model_state_dict", ckpt)

    # ------------------------------------------------------------------
    # 3. Extract EEG encoder weights
    # ------------------------------------------------------------------
    # In CLIPModel the EEG branch is stored as:
    #   eeg_model        -> EEGEncoder wrapper
    #   eeg_model.model  -> NeuralTransformer (the actual backbone)
    # Therefore the checkpoint keys look like:
    #   eeg_model.model.patch_embed.conv1.weight
    #   eeg_model.model.blocks.0.attn.qkv.weight
    #   ...
    # We strip the "eeg_model.model." prefix so they match the bare
    # NeuralTransformer state_dict keys.
    eeg_weights = {
        k.replace("eeg_model.model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("eeg_model.model.")
    }

    if not eeg_weights:
        raise RuntimeError(
            "No EEG encoder weights found in the checkpoint. "
            "Make sure you are loading a MindMix fusion checkpoint, "
            "not the raw LaBraM backbone checkpoint."
        )

    # ------------------------------------------------------------------
    # 4. Load weights into the backbone
    # ------------------------------------------------------------------
    missing, unexpected = backbone.load_state_dict(eeg_weights, strict=False)
    if missing:
        print(f"[Warning] Missing keys ({len(missing)}):", missing[:5])
    if unexpected:
        print(f"[Warning] Unexpected keys ({len(unexpected)}):", unexpected[:5])

    # ------------------------------------------------------------------
    # 5. Wrap with the same rearrange logic used in EEGEncoder
    # ------------------------------------------------------------------
    class EEGEncoder(torch.nn.Module):
        """
        Wrapper that reshapes flat EEG input ``[B, N, 400]`` into
        ``[B, N, 2, 200]`` before feeding the backbone.
        """
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch_size, n_channels, 400]
            x = rearrange(x, "B N (A T) -> B N A T", A=2, T=200)
            return self.model(x)  # -> [batch_size, 200]

    encoder = EEGEncoder(backbone).to(device).eval()
    return encoder


def main():
    parser = argparse.ArgumentParser(
        description="Load the pre-trained MindMix EEG encoder"
    )
    parser.add_argument(
        "--checkpoint",
        default="pretrain_fusion_checkpoints/best_model_loss_0.0909.pth",
        help="Path to the MindMix fusion checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu or cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for the dummy inference",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Number of EEG channels",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MindMix Pre-trained EEG Encoder Loader")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {args.device}")
    print(f"Input shape: [{args.batch_size}, {args.channels}, 400]")
    print("-" * 60)

    encoder = build_mindmix_eeg_encoder(args.checkpoint, device=args.device)

    # Dummy inference
    dummy_eeg = torch.randn(args.batch_size, args.channels, 400).to(args.device)
    with torch.no_grad():
        features = encoder(dummy_eeg)

    print(f"Output shape: {features.shape}")  # [B, 200]
    print("-" * 60)
    print("[OK] EEG encoder loaded and inference successful.")
    print("=" * 60)


if __name__ == "__main__":
    main()
