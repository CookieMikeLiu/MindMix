#!/usr/bin/env python3
"""
Quick Start Example for MindMix

This script demonstrates basic usage of MindMix for EEG-audio multimodal learning.
"""

import torch
import numpy as np
from universal_models import CLARA, ClipLoss


def create_dummy_data(batch_size=8, eeg_channels=64, eeg_samples=400, audio_dim=768):
    """Create dummy EEG and audio data for demonstration."""
    # EEG data: (batch, channels, time)
    eeg_data = torch.randn(batch_size, eeg_channels, eeg_samples)
    
    # Audio embeddings: (batch, embed_dim)
    audio_data = torch.randn(batch_size, audio_dim)
    
    return eeg_data, audio_data


def demo_clara_module():
    """Demonstrate the CLARA fusion module."""
    print("=" * 60)
    print("Demo 1: CLARA (Cross-Attention Low-Rank Alignment) Module")
    print("=" * 60)
    
    # Create CLARA module
    clara = CLARA(
        embed_dim=256,
        num_heads=4,
        ffn_hidden_factor=2,
        low_rank_factor=0.5,
        dropout_rate=0.1
    )
    
    # Create dummy EEG embeddings (after encoder)
    batch_size = 8
    eeg_embed = torch.randn(batch_size, 256)
    audio_embed = torch.randn(batch_size, 256)
    
    # Forward pass through CLARA
    eeg_out, audio_out = clara(eeg_embed, audio_embed)
    
    print(f"Input EEG shape: {eeg_embed.shape}")
    print(f"Input Audio shape: {audio_embed.shape}")
    print(f"Output EEG shape: {eeg_out.shape}")
    print(f"Output Audio shape: {audio_out.shape}")
    print("[OK] CLARA module successfully fuses EEG and audio representations\n")


def demo_clip_loss():
    """Demonstrate the CLIP contrastive loss."""
    print("=" * 60)
    print("Demo 2: CLIP Contrastive Loss")
    print("=" * 60)
    
    # Create CLIP loss
    clip_loss = ClipLoss(linear=None, center=False, initial_temp=0.07)
    
    # Create dummy embeddings
    batch_size = 8
    eeg_embed = torch.randn(batch_size, 256)
    audio_embed = torch.randn(batch_size, 256)
    
    # Compute loss
    loss = clip_loss(eeg_embed, audio_embed)
    
    # Get similarity scores
    scores = clip_loss.get_scores(eeg_embed, audio_embed)
    probabilities = clip_loss.get_probabilities(eeg_embed, audio_embed)
    
    print(f"EEG embeddings shape: {eeg_embed.shape}")
    print(f"Audio embeddings shape: {audio_embed.shape}")
    print(f"Contrastive loss: {loss.item():.4f}")
    print(f"Similarity scores shape: {scores.shape}")
    print(f"Learned temperature (1/logit_scale): {1/clip_loss.logit_scale.exp().item():.4f}")
    print("[OK] CLIP loss aligns EEG and audio in shared embedding space\n")


def demo_inference_pipeline():
    """Demonstrate a complete inference pipeline."""
    print("=" * 60)
    print("Demo 3: Complete Inference Pipeline")
    print("=" * 60)
    
    batch_size = 4
    num_candidates = 10
    
    # Simulate EEG query
    eeg_query = torch.randn(1, 256)
    
    # Simulate audio candidate library
    audio_candidates = torch.randn(num_candidates, 256)
    
    # Compute similarities
    clip_loss = ClipLoss(initial_temp=0.07)
    
    # Expand EEG query to match candidates
    eeg_query_expanded = eeg_query.expand(num_candidates, -1)
    
    # Get scores
    scores = clip_loss.get_scores(eeg_query_expanded, audio_candidates)
    probabilities = clip_loss.get_probabilities(eeg_query_expanded, audio_candidates)
    
    # Get top-k matches
    top_k = 3
    top_scores, top_indices = torch.topk(probabilities[0], k=top_k)
    
    print(f"EEG query shape: {eeg_query.shape}")
    print(f"Audio candidates shape: {audio_candidates.shape}")
    print(f"\nTop-{top_k} matching audio candidates:")
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        print(f"  Rank {i+1}: Candidate {idx.item()} (score: {score.item():.4f})")
    
    print("\n[OK] Inference pipeline successfully retrieves matching audio from EEG\n")


def print_model_summary():
    """Print a summary of MindMix architecture."""
    print("=" * 60)
    print("MindMix Model Architecture Summary")
    print("=" * 60)
    
    summary = """
Stage 1: EEG Foundation Pre-training
  - Data: >3,000 hours of EEG recordings
  - Model: LaBraM (Large Brain Model)
  - Task: Learn generalized EEG representations
  
Stage 2: Neural-Acoustic Alignment
  - Data: >100 hours of paired EEG-audio
  - Module: CLARA (Cross-Attention Low-Rank Alignment)
  - Task: Align EEG and audio embeddings
  
Downstream Tasks:
  1. Auditory Attention Decoding (KUL, DTU datasets)
  2. Auditory Emotion Recognition (EEG4EMO dataset)
  3. Cross-Modal Retrieval (EEG <-> Audio)
    """
    print(summary)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("MindMix Quick Start Demo")
    print("=" * 60 + "\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demos
    demo_clara_module()
    demo_clip_loss()
    demo_inference_pipeline()
    print_model_summary()
    
    print("=" * 60)
    print("Demo completed! See README.md for full usage instructions.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
