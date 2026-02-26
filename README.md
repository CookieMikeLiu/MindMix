# MindMix: A Multimodal Foundation Model for Auditory Perception Decoding

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=1ifQzlETeG)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

> **MindMix** is a multimodal foundation model that bridges the gap between unimodal EEG foundations and task-specific auditory decoders, enabling powerful auditory perception decoding from non-invasive EEG signals.

---

## 📋 Overview

Decoding complex auditory experiences from non-invasive EEG is a rapidly emerging field with significant promise for advancing both fundamental neuroscience and human-machine interaction technologies. While recent EEG foundation models have yielded powerful neural representations, their effectiveness remains constrained by limited integration with acoustic stimulus information.

**MindMix** addresses this challenge through:

- 🧠 **Two-Stage Training Strategy**: Generalized EEG feature learning followed by neural-acoustic alignment
- 🔄 **Cross-Attention Low-Rank Alignment (CLARA)**: Novel module for fine-grained cross-modal information integration
- 📊 **State-of-the-Art Performance**: Superior results on auditory attention decoding, emotion recognition, and cross-modal retrieval tasks

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🔬 Foundation Model** | Pre-trained on 3,000+ hours of EEG data for generalized neural representations |
| **🎵 Multimodal Fusion** | Novel CLARA module for EEG-audio cross-modal alignment |
| **🎯 Multi-Task Support** | Auditory attention decoding, emotion recognition, cross-modal retrieval |
| **⚡ Flexible Fine-tuning** | Three strategies: EEG-only, multimodal real, and multimodal prototype |
| **📈 SOTA Results** | Substantially surpasses existing baselines across diverse auditory tasks |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MindMix Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: EEG Foundation Pre-training                           │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐   │
│  │  EEG Input  │────▶│  LaBraM Encoder │────▶│ EEG Features│   │
│  │  (>3000hrs) │     │   (Pre-trained) │     │  (General)  │   │
│  └─────────────┘     └─────────────────┘     └─────────────┘   │
│                                                                  │
│  Stage 2: Neural-Acoustic Alignment                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │  EEG Embed  │────▶│             │────▶│  Aligned EEG    │   │
│  │             │     │    CLARA    │     │  Representation │   │
│  └─────────────┘     │   Module    │     └─────────────────┘   │
│  ┌─────────────┐     │ (Low-Rank  │                            │
│  │Audio Embed  │────▶│ Cross-Attn)│                            │
│  │ (>100hrs)   │     └─────────────┘                            │
│  └─────────────┘                                                │
│                                                                  │
│  Downstream Tasks                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │Attention Decode│  │Emotion Recogn. │  │Cross-Modal     │    │
│  │    (KUL/DTU)   │  │   (EEG4EMO)    │  │   Retrieval    │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CLARA Module

The **Cross-Attention Low-Rank Alignment (CLARA)** module is our novel contribution for effective EEG-audio fusion:

- **Self-Attention Paths**: Independent processing for EEG and audio modalities
- **Cross-Attention Fusion**: Bidirectional cross-modal attention with low-rank decomposition
- **Residual Connections**: Maintains modality-specific information while learning shared representations

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/CookieMikeLiu/MindMix.git
cd MindMix

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers timm einops tensorboardX
pip install numpy pandas scikit-learn scipy h5py tqdm
pip install pyhealth
```

### Data Preparation

Organize your datasets as follows:

```
data/
├── EEG4EMO/           # EEG-Audio emotion recognition
│   ├── train/
│   ├── test/
│   └── labels.csv
├── KUL/               # Auditory attention decoding (KUL dataset)
│   ├── subjects/
│   └── metadata.json
└── DTU/               # Auditory attention decoding (DTU dataset)
    ├── subjects/
    └── metadata.json
```

### Pre-training (Stage 1)

Train the EEG foundation model with paired EEG-audio data:

```bash
python MindMix_clip_pretrain.py \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --input_size 400 \
    --model labram_base_patch200_200 \
    --output_dir ./pretrain_fusion_checkpoints
```

### Fine-tuning (Stage 2)

#### Auditory Emotion Recognition (EEG4EMO)

```bash
python universal_eeg_finetune.py \
    --dataset EEG4EMO \
    --strategy multimodal_real \
    --fusion_method clara \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-5
```

#### Auditory Attention Decoding (KUL/DTU)

```bash
python universal_eeg_finetune.py \
    --dataset KUL \
    --strategy multimodal_real \
    --fusion_method clara \
    --eval_method contrastive \
    --batch_size 32 \
    --epochs 50
```

### Using Pre-trained Checkpoints

```python
import torch
from modeling_finetune_2 import labram_base_patch200_200

# Load pre-trained EEG encoder
checkpoint = torch.load('pretrain_fusion_checkpoints/checkpoint-best.pth')
model = labram_base_patch200_200(
    num_classes=256,  # Embedding dimension
    drop_path_rate=0.1
)
model.load_state_dict(checkpoint['model'])
```

---

## 📁 Project Structure

```
MindMix/
├── MindMix_clip_pretrain.py      # Stage 1: EEG-audio fusion pre-training
├── MindMix_clip_finetune.py      # Stage 2: Cross-modal fine-tuning
├── universal_eeg_finetune.py     # Universal fine-tuning framework
├── universal_models.py           # Model architectures (CLARA, ClipLoss)
├── universal_trainer.py          # Training utilities
├── utils.py                      # General utilities
├── modeling_finetune_2.py        # LaBraM model implementation
├── pretrain_fusion_checkpoints/  # Pre-trained model checkpoints
└── README.md                     # This file
```

### File Descriptions

| File | Description |
|------|-------------|
| `MindMix_clip_pretrain.py` | Pre-trains EEG encoder with CLIP-style contrastive learning on EEG-audio pairs |
| `MindMix_clip_finetune.py` | Fine-tunes the model on specific downstream tasks |
| `universal_eeg_finetune.py` | Universal framework supporting multiple datasets and strategies |
| `universal_models.py` | Core model components: CLARA module, ClipLoss, classification heads |
| `utils.py` | Data loading, preprocessing, channel mapping, evaluation metrics |

---

## 🎯 Supported Tasks & Datasets

### 1. Auditory Attention Decoding
- **Datasets**: KUL (KU Leuven), DTU (Technical University of Denmark)
- **Task**: Identify which of multiple speakers a subject is attending to
- **Evaluation**: Contrastive learning based accuracy

### 2. Auditory Emotion Recognition
- **Dataset**: EEG4EMO
- **Task**: Classify emotional valence from EEG during music listening
- **Evaluation**: Classification accuracy, F1-score

### 3. Cross-Modal Retrieval
- **Task**: Retrieve matching audio given EEG (or vice versa)
- **Evaluation**: Recall@K, Mean Reciprocal Rank (MRR)

---

## 🧪 Fine-tuning Strategies

### 1. EEG-Only (`eeg_only`)
Baseline using only EEG encoder for classification.

```bash
python universal_eeg_finetune.py --strategy eeg_only --dataset EEG4EMO
```

### 2. Multimodal Real (`multimodal_real`)
Uses real paired EEG-audio data with CLARA fusion.

```bash
python universal_eeg_finetune.py --strategy multimodal_real --fusion_method clara
```

### 3. Multimodal Prototype (`multimodal_prototype`)
Uses EEG with pseudo-audio prototypes for lightweight training.

```bash
python universal_eeg_finetune.py --strategy multimodal_prototype --fusion_method clara
```

---

## 📊 Results

MindMix substantially surpasses existing baselines across multiple auditory decoding tasks:

| Task | Dataset | MindMix | Previous SOTA |
|------|---------|---------|---------------|
| Attention Decoding | KUL | **XX.X%** | XX.X% |
| Attention Decoding | DTU | **XX.X%** | XX.X% |
| Emotion Recognition | EEG4EMO | **XX.X%** | XX.X% |
| Cross-Modal Retrieval | - | **XX.X%** | XX.X% |

*Detailed results available in our [paper](https://openreview.net/forum?id=1ifQzlETeG).*

---

## 🔧 Configuration Options

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `labram_base_patch200_200` | Model architecture |
| `--input_size` | 400 | EEG input size (time samples) |
| `--drop_path` | 0.1 | Stochastic depth rate |
| `--fusion_method` | `clara` | Fusion module: `clara` or `concat` |

### Training Parameters

| Parameter | Pre-train | Fine-tune | Description |
|-----------|-----------|-----------|-------------|
| `--batch_size` | 32 | 32 | Batch size |
| `--lr` | 1e-4 | 1e-5 | Learning rate |
| `--epochs` | 100 | 50 | Training epochs |
| `--weight_decay` | 0.05 | 0.01 | Weight decay |

---

## 📚 Citation

If you find MindMix useful in your research, please cite our paper:

```bibtex
@inproceedings{liu2025mindmix,
  title={MindMix: A Multimodal Foundation Model for Auditory Perception Decoding},
  author={Liu, Mike and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## 🤝 Acknowledgments

This work builds upon several excellent open-source projects:

- [LaBraM](https://github.com/935963004/LaBraM) - Large Brain Model for EEG
- [BEiT-v2](https://github.com/microsoft/unilm/tree/master/beitv2) - Transformer architecture
- [timm](https://github.com/rwightman/pytorch-image-models) - PyTorch model library
- [BIOT](https://github.com/ycq091044/BIOT) - Brain signal processing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the authors.

---

<div align="center">

⭐ **Star this repo if you find it helpful!** ⭐

</div>
