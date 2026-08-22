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
- 🔄 **Cross-Attention Low-Rank Alignment (CALRA)**: Novel module for fine-grained cross-modal information integration
- 📊 **State-of-the-Art Performance**: Superior results on auditory attention decoding, emotion recognition, and cross-modal retrieval tasks

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🔬 Foundation Model** | Pre-trained on 3,000+ hours of EEG data for generalized neural representations |
| **🎵 Multimodal Fusion** | Novel CALRA module for EEG-audio cross-modal alignment |
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
│  │  EEG Input  │────▶│  EEG Encoder    │────▶│ EEG Features│   │
│  │  (>3000hrs) │     │   (Pre-trained) │     │  (General)  │   │
│  └─────────────┘     └─────────────────┘     └─────────────┘   │
│                                                                  │
│  Stage 2: Neural-Acoustic Alignment                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │  EEG Embed  │────▶│             │────▶│  Aligned EEG    │   │
│  │             │     │    CALRA    │     │  Representation │   │
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

### CALRA Module

The **Cross-Attention Low-Rank Alignment (CALRA)** module is our novel contribution for effective EEG-audio fusion:

- **Self-Attention Paths**: Independent processing for EEG and audio modalities
- **Cross-Attention Fusion**: Bidirectional cross-modal attention with low-rank decomposition
- **Residual Connections**: Maintains modality-specific information while learning shared representations

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository and fetch Git LFS checkpoint files
git lfs install
git clone https://github.com/CookieMikeLiu/MindMix.git
cd MindMix
git lfs pull

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Downstream datasets are not included in this repository. Prepare the
preprocessed files locally and pass their directory with `--data_path`.

```
Dataset/
├── EEG4EMO/
│   └── preprocessed_pair/
│       └── <subject_id>/
│           ├── <subject_id>_labels.csv
│           └── *video*.pkl
├── KUL_trail/
│   ├── <subject_id>.pkl
│   └── ...
└── DTU_trail/
    ├── <subject_id>.pkl
    └── ...
```

For KUL/DTU, each subject `.pkl` can be a pandas `DataFrame` or a list of
records with `eeg`, `target_audio`, `attended_label`, and either
`negetive_audio` (legacy spelling) or `negative_audio`.

### Neural-Acoustic Alignment (Stage 2)

The released MindMix script trains the paired EEG-audio alignment stage on top
of an EEG backbone. The backbone can be initialized with `--finetune`; for
example, `checkpoints/v2.4_large.pth` is our EEG-only pre-trained checkpoint,
while `checkpoints/labram-base.pth` is the default initialization used by the
released scripts.

```bash
python MindMix_clip_pretrain.py \
    --data_path <path_to_paired_EEG_audio_pretraining_data> \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --input_size 400 \
    --model labram_base_patch200_200 \
    --output_dir ./pretrain_fusion_checkpoints
```

### Downstream Fine-tuning (Stage 3)

#### Auditory Emotion Recognition (EEG4EMO)

```bash
python universal_eeg_finetune.py \
    --dataset EEG4EMO \
    --strategy multimodal_real \
    --fusion_method calra \
    --pretrained_model pretrain_fusion_checkpoints/best_model_loss_0.0909.pth \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-5
```

#### Auditory Attention Decoding (KUL/DTU)

```bash
python universal_eeg_finetune.py \
    --dataset KUL \
    --data_path <path_to_KUL_trail> \
    --strategy multimodal_real \
    --fusion_method calra \
    --pretrained_model pretrain_fusion_checkpoints/best_model_loss_0.0909.pth \
    --batch_size 32 \
    --epochs 50
```

For DTU, use `--dataset DTU --data_path <path_to_DTU_trail>` with the same
multimodal settings.

### Using Pre-trained Checkpoints

We provide the pre-trained MindMix fusion checkpoint at:
- `pretrain_fusion_checkpoints/best_model_loss_0.0909.pth`

If this file is only a small Git LFS pointer after cloning, run `git lfs pull`
from the repository root.

For the reported MindMix downstream results, the model is initialized from our
MindMix EEG-audio pre-training checkpoint via `--pretrained_model`. The files in
`checkpoints/` are EEG-backbone checkpoints used for initialization or ablation,
not the final cross-modal MindMix checkpoint. In the downstream scripts,
`--finetune` first builds/loads the EEG backbone, and `--pretrained_model` then
loads the released MindMix EEG, audio, projection, and fusion weights.

| Path | Role |
|------|------|
| `pretrain_fusion_checkpoints/best_model_loss_0.0909.pth` | Released MindMix EEG-audio pre-training checkpoint. This is the checkpoint used by `--pretrained_model` for downstream fine-tuning and corresponds to the reported MindMix results. |
| `checkpoints/v2.4_large.pth` | Our EEG-only pre-trained backbone checkpoint. It can be used for EEG-backbone initialization or ablations, but it is not a replacement for the released MindMix EEG-audio checkpoint. |
| `checkpoints/labram-base.pth` | LaBraM Base checkpoint used as the default backbone initialization in the released scripts. |

To extract and use the **EEG encoder** from this checkpoint:

```python
import torch
from modeling_finetune_2 import labram_base_patch200_200
from einops import rearrange
from utils import load_trusted_checkpoint

# 1. Instantiate the EEG backbone
#    num_classes=0 removes the classification head; output is [B, 200]
eeg_encoder = labram_base_patch200_200(
    pretrained=False,
    num_classes=0,
    drop_path_rate=0.1,
    use_mean_pooling=True,
    use_rel_pos_bias=True,
    use_abs_pos_emb=True,
    init_values=0.1,
    qkv_bias=True,
)

# 2. Load the MindMix fusion checkpoint
ckpt = load_trusted_checkpoint(
    'pretrain_fusion_checkpoints/best_model_loss_0.0909.pth',
    map_location='cpu',
)
state_dict = ckpt['model_state_dict']  # key name used during training

# 3. Extract EEG encoder weights
#    In the fusion checkpoint, EEG backbone weights are stored under
#    the prefix "eeg_model.model.*". Strip that prefix before loading.
eeg_weights = {
    k.replace('eeg_model.model.', ''): v
    for k, v in state_dict.items()
    if k.startswith('eeg_model.model.')
}

# 4. Load into the backbone
eeg_encoder.load_state_dict(eeg_weights, strict=False)
eeg_encoder.eval()

# 5. Wrap with the same rearrange logic used during training
class EEGEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # x: [B, n_channels, 400] -> [B, n_channels, 2, 200]
        x = rearrange(x, 'B N (A T) -> B N A T', A=2, T=200)
        return self.model(x)

encoder = EEGEncoder(eeg_encoder)

# Example inference
dummy_eeg = torch.randn(2, 64, 400)  # [batch, 64 channels, 400 time points]
with torch.no_grad():
    feat = encoder(dummy_eeg)  # [2, 200]
print("EEG feature shape:", feat.shape)
```

For a complete runnable script, see [`load_pretrained_eeg.py`](load_pretrained_eeg.py).

### Sanity Check Demo

To verify the installation and inspect the core CALRA and contrastive-learning
components without preparing a real dataset, run:

```bash
python quick_start.py
```

This demo uses synthetic tensors only. It is intended as a lightweight
reader-facing example rather than a training or evaluation script.

---

## 📁 Project Structure

```
MindMix/
|-- checkpoints/                  # EEG backbone checkpoints used by --finetune
|   |-- labram-base.pth           # Default LaBraM Base checkpoint for released scripts
|   `-- v2.4_large.pth            # Our EEG-only pre-trained backbone checkpoint
|-- pretrain_fusion_checkpoints/  # Released MindMix pre-training checkpoints
|   `-- best_model_loss_0.0909.pth
|-- MindMix_clip_pretrain.py      # Stage 2: EEG-audio alignment pre-training
|-- MindMix_clip_finetune.py      # Task-specific AAD fine-tuning script
|-- universal_eeg_finetune.py     # Recommended downstream fine-tuning entry point
|-- universal_models.py           # Universal downstream model wrappers and CALRA/ClipLoss
|-- universal_trainer.py          # Universal training and evaluation loops
|-- modeling_finetune_2.py        # EEG backbone implementation used by MindMix
|-- load_pretrained_eeg.py        # Example: extract EEG encoder from fusion checkpoint
|-- quick_start.py                # Synthetic smoke-test/demo script
|-- utils.py                      # Checkpoint loading and shared utilities
|-- requirements.txt              # Python dependencies
`-- README.md
```

### File Descriptions

| File or directory | Description |
|-------------------|-------------|
| `pretrain_fusion_checkpoints/` | Released MindMix EEG-audio pre-training checkpoint directory. `best_model_loss_0.0909.pth` is the default `--pretrained_model` used for downstream fine-tuning. |
| `checkpoints/` | EEG-backbone checkpoints used by `--finetune`. `labram-base.pth` is the default released-script initialization; `v2.4_large.pth` is our EEG-only pre-trained backbone checkpoint for initialization/ablation experiments. |
| `MindMix_clip_pretrain.py` | Pre-trains EEG encoder with CLIP-style contrastive learning on EEG-audio pairs |
| `MindMix_clip_finetune.py` | Task-specific fine-tuning script retained for AAD experiments |
| `universal_eeg_finetune.py` | Recommended universal fine-tuning framework supporting multiple datasets and strategies |
| `universal_models.py` | Core model components: CALRA module, ClipLoss, classification heads |
| `universal_trainer.py` | Training and evaluation loops for EEG-only, multimodal real, and multimodal prototype strategies |
| `modeling_finetune_2.py` | EEG backbone architecture code used to instantiate the encoder before loading Stage-1 or MindMix checkpoints; the released commands instantiate `labram_base_patch200_200` |
| `utils.py` | Data loading, preprocessing, channel mapping, evaluation metrics |
| `quick_start.py` | Lightweight demo using synthetic tensors to verify installation and core modules |
| `load_pretrained_eeg.py` | Runnable example for extracting and using the EEG encoder from a MindMix checkpoint |
| `requirements.txt` | Python package requirements |

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
Uses real paired EEG-audio data with CALRA fusion.

```bash
python universal_eeg_finetune.py --strategy multimodal_real --fusion_method calra
```

### 3. Multimodal Prototype (`multimodal_prototype`)
Uses EEG with pseudo-audio prototypes for lightweight training.

```bash
python universal_eeg_finetune.py --strategy multimodal_prototype --fusion_method calra
```

---

## 📊 Results

MindMix substantially surpasses existing baselines across multiple auditory decoding tasks. The released checkpoint and scripts reproduce the auditory attention decoding protocol reported in the paper:

| Task | Dataset | MindMix | Previous SOTA |
|------|---------|---------|---------------|
| Attention Decoding | KUL | **99.82%** | 94.81% |
| Attention Decoding | DTU | **99.93%** | 84.56% |
| Emotion Recognition | HR-EEG4EMO | **88.78%** | 82.74% |
| Cross-Modal Retrieval | MAD-EEG (Duo Acc.) | **94.75%** | 94.25% |

*Detailed results available in our [paper](https://openreview.net/forum?id=1ifQzlETeG).*
---

## 🔧 Configuration Options

### Model Parameters

`--model` selects the EEG backbone architecture used inside MindMix. The
released MindMix EEG-audio pre-trained checkpoint and README commands use
`labram_base_patch200_200`, whose EEG features are 200-dimensional before
projection into the 256-dimensional fusion space. `checkpoints/v2.4_large.pth`
is our EEG-only pre-trained backbone checkpoint; downstream users should still
load `pretrain_fusion_checkpoints/best_model_loss_0.0909.pth` with
`--pretrained_model` to reproduce the released MindMix results.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `labram_base_patch200_200` | EEG backbone architecture identifier used to instantiate the EEG encoder |
| `--input_size` | 400 | EEG input size (time samples) |
| `--drop_path` | 0.1 | Stochastic depth rate |
| `--fusion_method` | `calra` | Fusion module: `calra`, `cross_attention`, `simple_fusion`, `bidirectional_fusion`, or `calra_enhanced` |
| `--finetune` | `checkpoints/labram-base.pth` | EEG backbone initialization checkpoint; `checkpoints/v2.4_large.pth` is our EEG-only pre-trained checkpoint for backbone initialization/ablation |
| `--pretrained_model` | `pretrain_fusion_checkpoints/best_model_loss_0.0909.pth` | Released MindMix EEG-audio pre-training checkpoint used for downstream fine-tuning |
| `--use_auditory_type` | disabled | Optional auditory-type-specific CALRA aligners; leave disabled when a downstream dataset does not provide auditory type labels |

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
@inproceedings{liu2026mindmix,
  title={MindMix: A Multimodal Foundation Model for Auditory Perception Decoding},
  author={Liu, Mike and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
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
