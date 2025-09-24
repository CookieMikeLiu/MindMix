import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用tensorflow警告（如果有的话）

# 禁用特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", message=".*?torch.distributed.*?")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import random
import numpy as np
from modeling_finetune_2 import labram_base_patch200_200
from einops import rearrange
from torch.utils.data import TensorDataset
from timm.models import create_model
import argparse
import utils
from collections import OrderedDict
from transformers import Wav2Vec2Model
from tqdm import tqdm
import json
from sklearn.model_selection import KFold

# 禁用transformers的警告
import transformers
transformers.logging.set_verbosity_error()

# 设置torch的警告级别
torch.set_warn_always(False)


def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning script for EEG-Audio cross-modal learning', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)  # 微调使用较小的batch size
    parser.add_argument('--epochs', default=50, type=int)  # 微调使用较少的epochs
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=400, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,  # 微调使用更小的weight decay
                        help='weight decay (default: 0.01)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',  # 微调使用更小的学习率
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-7)')

    parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',  # 微调需要较短的warmup
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params  
    parser.add_argument('--pretrained_model', default='./pretrain_fusion_checkpoints/best_model_loss_0.0909.pth',  # 预训练模型路径
                        help='Path to the pretrained CLIP model checkpoint')
    parser.add_argument('--finetune', default='./checkpoints/labram-base.pth',  # EEG encoder的初始checkpoint
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./finetune_results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)  # 微调默认保存

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)  # 微调使用适中的workers
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV')

    # 微调特定参数
    parser.add_argument('--data_path', default='./Dataset/KUL/paired_data/EEG_audio', type=str,
                        help='Path to the downstream task dataset')
    parser.add_argument('--dataset_type', default='KUL', type=str, choices=['KUL', 'DTU'],
                        help='Type of dataset: KUL or DTU')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature for contrastive learning')
    parser.add_argument('--n_folds', default=5, type=int,
                        help='Number of folds for cross-validation')
    parser.add_argument('--fusion_method', default='clara', type=str, 
                        choices=['cross_attention', 'simple_fusion', 'bidirectional_fusion', 'clara_enhanced', 'clara'],
                        help='Cross-modal fusion method to use')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


ch_names = [
            "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1",\
            "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7",\
            "PO3", "O1", "IZ", "OZ", "POZ", "PZ", "CPZ", "FPZ", "FP2", "AF8", "AF4", "AFZ", "FZ",\
            "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ", "C2", "C4", "C6", "T8",\
            "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2"
        ]

# 固定随机种子
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )

    return model


class EEGEncoder(nn.Module):
    def __init__(self, args, device):
        super(EEGEncoder, self).__init__()
        self.args = args
        self.device = device
        self.model = self.load_model()  # Load the model during initialization
        self.fc_layer = None  # Initialize the fully connected layer as None

    def load_model(self):
        model = get_models(self.args)

        if self.args.finetune:
            if self.args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.args.finetune, map_location='cpu')

            print("Load checkpoint from %s" % self.args.finetune)
            checkpoint_model = self.get_checkpoint_model(checkpoint)

            self.adjust_checkpoint_keys(checkpoint_model, model)
            utils.load_state_dict(model, checkpoint_model, prefix=self.args.model_prefix)

        model.to(self.device)
        return model

    def forward(self, eeg_data):
        # Forward pass through the model
        eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', A=2, T=200)
        output = self.model(eeg_data)  # This is on self.device (GPU)

        return output

    def get_checkpoint_model(self, checkpoint):
        checkpoint_model = None
        for model_key in self.args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Loaded state_dict by model_key = %s" % model_key)
                break
        return checkpoint_model if checkpoint_model is not None else checkpoint

    def adjust_checkpoint_keys(self, checkpoint_model, model):
        if checkpoint_model is not None and self.args.model_filter_name != '':
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]

            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        for key in list(checkpoint_model.keys()):
            if "relative_position_index" in key:
                checkpoint_model.pop(key)


class DownstreamDataset(Dataset):
    """下游任务数据集类"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        eeg = sample['eeg']
        target_audio = sample['target_audio']
        negative_audio = sample['negetive_audio']
        label = sample['attended_label']
        
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        target_audio_tensor = torch.tensor(target_audio, dtype=torch.float32)
        negative_audio_tensor = torch.tensor(negative_audio, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return eeg_tensor, target_audio_tensor, negative_audio_tensor, label_tensor


def load_subject_data(folder_path):
    """加载每个subject的数据到独立DataFrame"""
    subject_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            subject_id = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            data = pd.read_pickle(file_path)
            if isinstance(data, list):
                data = pd.DataFrame(data)
            subject_data[subject_id] = data
    return subject_data


class ClipLoss(nn.Module):
    """CLIP contrastive loss for EEG and audio embeddings."""
    def __init__(self, linear=None, center=False, initial_temp=0.07):
        super().__init__()
        if linear is not None and hasattr(nn, 'LazyLinear'):
            self.linear_est = nn.LazyLinear(linear)
        elif linear is not None:
            print("Warning: nn.LazyLinear not found, linear projection in ClipLoss might not work as expected without in_features.")
            self.linear_est = None
        else:
            self.linear_est = None
        self.linear_gt = self.linear_est
        self.center = center
        
        # 使用 logit_scale，初始值基于 initial_temp
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / initial_temp))

    def get_scores(self, estimates, candidates):
        if self.linear_est is not None:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)

        if self.center:
            estimates = estimates - estimates.mean(dim=1, keepdim=True)
            candidates = candidates - candidates.mean(dim=1, keepdim=True)
        
        estimates = F.normalize(estimates, p=2, dim=-1)
        candidates = F.normalize(candidates, p=2, dim=-1)

        scores = torch.mm(estimates, candidates.T) * self.logit_scale.exp()
        return scores

    def get_probabilities(self, estimates, candidates):
        scores = self.get_scores(estimates, candidates)
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate):
        assert estimate.size(0) <= candidate.size(0), "Need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(estimate.size(0), device=estimate.device)
        return F.cross_entropy(scores, target)


class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 定义query/key/value投影
        self.eeg_query = nn.Linear(dim, dim)
        self.audio_key = nn.Linear(dim, dim)
        self.audio_value = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, eeg, audio):
        """
        输入:
            eeg: [B, D] (D=256)
            audio: [B, D]
        输出:
            attended_eeg: [B, D]
        """
        B = eeg.size(0)
        
        # 将EEG作为query，音频作为key/value
        q = self.eeg_query(eeg).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        k = self.audio_key(audio).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        v = self.audio_value(audio).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        
        # 计算注意力分数
        attn_scores = torch.einsum('bhd,bhd->bh', q, k) / np.sqrt(self.head_dim)  # [B, H]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H]
        
        # 加权聚合
        attended = torch.einsum('bh,bhd->bhd', attn_weights, v)  # [B, H, d]
        attended = attended.reshape(B, -1)  # [B, D]
        
        # 残差连接
        return self.out_proj(attended) + eeg  # [B, D]


class CrossModalFusion_Simple(nn.Module):
    """简化的跨模态融合 - 基于局部操作的低秩分解"""
    def __init__(self, embed_dim=256, num_heads=4, low_rank_factor=0.5, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.low_rank_dim = max(1, int(embed_dim * low_rank_factor))
        
        # 保持与CrossModalAttention相同的结构，但添加低秩分解
        self.eeg_query = nn.Linear(embed_dim, self.low_rank_dim)
        self.audio_key = nn.Linear(embed_dim, self.low_rank_dim)
        self.audio_value = nn.Linear(embed_dim, embed_dim)
        
        # 低秩投影回原始维度
        self.low_rank_proj = nn.Linear(self.low_rank_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, eeg, audio):
        """
        保持与CrossModalAttention完全相同的局部操作逻辑
        """
        B = eeg.size(0)
        
        # 低秩投影 - 但保持局部操作
        q = self.eeg_query(eeg)      # [B, low_rank_dim]
        k = self.audio_key(audio)    # [B, low_rank_dim]
        v = self.audio_value(audio)  # [B, embed_dim]
        
        # 关键：保持element-wise操作，不使用全局注意力
        # 计算局部相似度（每个样本独立）
        similarity = torch.sum(q * k, dim=-1, keepdim=True)  # [B, 1]
        attn_weights = torch.sigmoid(similarity)  # [B, 1] 简单的门控
        
        # 应用注意力权重
        attended = attn_weights * v  # [B, embed_dim]
        
        # 低秩特征投影
        low_rank_features = self.low_rank_proj(q * k)  # [B, embed_dim]
        
        # 组合特征
        combined = attended + low_rank_features
        
        # 输出投影和残差连接（与CrossModalAttention完全一致）
        output = self.out_proj(combined)
        return self.dropout(output) + eeg  # 保持相同的残差连接


class CrossModalFusion_Bidirectional(nn.Module):
    """双向跨模态融合 - 基于简化融合的双向扩展"""
    def __init__(self, embed_dim=256, num_heads=4, low_rank_factor=0.5, dropout_rate=0.1, audio_alignment_weight=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.low_rank_dim = max(1, int(embed_dim * low_rank_factor))
        self.audio_alignment_weight = audio_alignment_weight  # 控制audio对齐强度
        
        # EEG分支 (主要对齐方向，与CrossModalFusion_Simple相同的成功结构)
        self.eeg_query = nn.Linear(embed_dim, self.low_rank_dim)
        self.audio_key = nn.Linear(embed_dim, self.low_rank_dim)
        self.audio_value = nn.Linear(embed_dim, embed_dim)
        self.eeg_low_rank_proj = nn.Linear(self.low_rank_dim, embed_dim)
        self.eeg_out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Audio分支 (次要对齐方向)
        self.audio_query = nn.Linear(embed_dim, self.low_rank_dim)
        self.eeg_key = nn.Linear(embed_dim, self.low_rank_dim)
        self.eeg_value = nn.Linear(embed_dim, embed_dim)
        self.audio_low_rank_proj = nn.Linear(self.low_rank_dim, embed_dim)
        self.audio_out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, eeg, audio):
        """
        双向跨模态融合，基于CrossModalFusion_Simple的成功模式
        """
        B = eeg.size(0)
        
        # === 主要方向：EEG -> Audio (完全复制CrossModalFusion_Simple的成功逻辑) ===
        eeg_q = self.eeg_query(eeg)      # [B, low_rank_dim]
        audio_k = self.audio_key(audio)  # [B, low_rank_dim]
        audio_v = self.audio_value(audio)  # [B, embed_dim]
        
        # 保持CrossModalFusion_Simple的成功模式：局部相似度计算
        eeg_audio_similarity = torch.sum(eeg_q * audio_k, dim=-1, keepdim=True)  # [B, 1]
        eeg_audio_weights = torch.sigmoid(eeg_audio_similarity)  # [B, 1]
        
        # EEG特征对齐
        eeg_attended = eeg_audio_weights * audio_v  # [B, embed_dim]
        eeg_low_rank_features = self.eeg_low_rank_proj(eeg_q * audio_k)  # [B, embed_dim]
        eeg_combined = eeg_attended + eeg_low_rank_features
        eeg_projected = self.eeg_out_proj(eeg_combined)
        aligned_eeg = self.dropout(eeg_projected) + eeg  # 与CrossModalFusion_Simple相同的残差连接
        
        # === 次要方向：Audio -> EEG (控制对齐强度) ===
        audio_q = self.audio_query(audio)  # [B, low_rank_dim]
        eeg_k = self.eeg_key(eeg)          # [B, low_rank_dim]
        eeg_v = self.eeg_value(eeg)        # [B, embed_dim]
        
        # 同样的局部相似度计算
        audio_eeg_similarity = torch.sum(audio_q * eeg_k, dim=-1, keepdim=True)  # [B, 1]
        audio_eeg_weights = torch.sigmoid(audio_eeg_similarity)  # [B, 1]
        
        # Audio特征对齐
        audio_attended = audio_eeg_weights * eeg_v  # [B, embed_dim]
        audio_low_rank_features = self.audio_low_rank_proj(audio_q * eeg_k)  # [B, embed_dim]
        audio_combined = audio_attended + audio_low_rank_features
        audio_projected = self.audio_out_proj(audio_combined)
        audio_aligned_full = self.dropout(audio_projected) + audio
        
        # 关键：控制audio对齐强度，保持对比学习稳定性
        aligned_audio = audio_aligned_full * self.audio_alignment_weight + audio * (1 - self.audio_alignment_weight)
        
        return aligned_eeg, aligned_audio


class CLARA_Enhanced(nn.Module):
    """增强版CLARA - 基于成功模式的自注意力+FFN+稳定交互"""
    def __init__(self, embed_dim=256, num_heads=4, ffn_hidden_factor=2, low_rank_factor=0.5, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 使用较小的FFN扩张倍数，避免过拟合
        ffn_hidden_dim = embed_dim * ffn_hidden_factor
        self.low_rank_dim = max(1, int(embed_dim * low_rank_factor))
        
        # === EEG模态处理链 ===
        # 简化的自注意力 - 使用点积注意力替代MultiheadAttention
        self.eeg_self_query = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_key = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_value = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_proj = nn.Linear(embed_dim, embed_dim)
        self.eeg_norm1 = nn.LayerNorm(embed_dim)
        
        # EEG的FFN（简化版本）
        self.eeg_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.eeg_norm2 = nn.LayerNorm(embed_dim)
        
        # === Audio模态处理链 ===
        # Audio的简化自注意力
        self.audio_self_query = nn.Linear(embed_dim, embed_dim)
        self.audio_self_key = nn.Linear(embed_dim, embed_dim)
        self.audio_self_value = nn.Linear(embed_dim, embed_dim)
        self.audio_self_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_norm1 = nn.LayerNorm(embed_dim)
        
        # Audio的FFN
        self.audio_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.audio_norm2 = nn.LayerNorm(embed_dim)
        
        # === 跨模态交互机制 (基于CrossModalFusion_Simple的成功模式) ===
        self.eeg_query = nn.Linear(embed_dim, self.low_rank_dim)
        self.audio_key = nn.Linear(embed_dim, self.low_rank_dim)
        self.audio_value = nn.Linear(embed_dim, embed_dim)
        self.eeg_low_rank_proj = nn.Linear(self.low_rank_dim, embed_dim)
        
        # 双向交互
        self.audio_query = nn.Linear(embed_dim, self.low_rank_dim)
        self.eeg_key = nn.Linear(embed_dim, self.low_rank_dim)
        self.eeg_value = nn.Linear(embed_dim, embed_dim)
        self.audio_low_rank_proj = nn.Linear(self.low_rank_dim, embed_dim)
        
        # 最终输出投影
        self.eeg_final_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_final_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def _simplified_self_attention(self, x, query_proj, key_proj, value_proj, out_proj):
        """简化的自注意力机制 - 适合单个特征向量"""
        B = x.size(0)
        
        # 投影到query, key, value
        q = query_proj(x).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        k = key_proj(x).view(B, self.num_heads, self.head_dim)    # [B, H, d]
        v = value_proj(x).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        
        # 计算自注意力分数 (局部操作，每个样本独立)
        attn_scores = torch.sum(q * k, dim=-1) / np.sqrt(self.head_dim)  # [B, H]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H]
        
        # 应用注意力权重
        attended = torch.einsum('bh,bhd->bhd', attn_weights, v)  # [B, H, d]
        attended = attended.reshape(B, -1)  # [B, embed_dim]
        
        # 输出投影
        return out_proj(attended)
    
    def forward(self, eeg, audio):
        """
        增强版CLARA处理流程：自注意力 -> 跨模态交互 -> FFN
        """
        B = eeg.size(0)
        
        # === 第一阶段：自注意力处理 ===
        # EEG自注意力
        eeg_self_attn = self._simplified_self_attention(
            eeg, self.eeg_self_query, self.eeg_self_key, 
            self.eeg_self_value, self.eeg_self_proj
        )
        eeg_after_self = self.eeg_norm1(eeg + self.dropout(eeg_self_attn))
        
        # Audio自注意力
        audio_self_attn = self._simplified_self_attention(
            audio, self.audio_self_query, self.audio_self_key,
            self.audio_self_value, self.audio_self_proj
        )
        audio_after_self = self.audio_norm1(audio + self.dropout(audio_self_attn))
        
        # === 第二阶段：跨模态交互 (使用CrossModalFusion_Simple的成功模式) ===
        # EEG -> Audio 交互
        eeg_q = self.eeg_query(eeg_after_self)      # [B, low_rank_dim]
        audio_k = self.audio_key(audio_after_self)  # [B, low_rank_dim]
        audio_v = self.audio_value(audio_after_self)  # [B, embed_dim]
        
        # 保持CrossModalFusion_Simple的成功模式：局部相似度计算
        eeg_audio_similarity = torch.sum(eeg_q * audio_k, dim=-1, keepdim=True)  # [B, 1]
        eeg_audio_weights = torch.sigmoid(eeg_audio_similarity)  # [B, 1]
        
        # EEG特征增强
        eeg_attended = eeg_audio_weights * audio_v  # [B, embed_dim]
        eeg_low_rank_features = self.eeg_low_rank_proj(eeg_q * audio_k)  # [B, embed_dim]
        eeg_cross_modal = eeg_attended + eeg_low_rank_features
        
        # Audio -> EEG 交互 (轻微对齐)
        audio_q = self.audio_query(audio_after_self)  # [B, low_rank_dim]
        eeg_k = self.eeg_key(eeg_after_self)          # [B, low_rank_dim]
        eeg_v = self.eeg_value(eeg_after_self)        # [B, embed_dim]
        
        audio_eeg_similarity = torch.sum(audio_q * eeg_k, dim=-1, keepdim=True)  # [B, 1]
        audio_eeg_weights = torch.sigmoid(audio_eeg_similarity)  # [B, 1]
        
        audio_attended = audio_eeg_weights * eeg_v  # [B, embed_dim]
        audio_low_rank_features = self.audio_low_rank_proj(audio_q * eeg_k)  # [B, embed_dim]
        audio_cross_modal = audio_attended + audio_low_rank_features
        
        # === 第三阶段：FFN处理 ===
        # EEG FFN
        eeg_ffn_input = eeg_after_self + self.dropout(eeg_cross_modal)
        eeg_ffn_out = self.eeg_ffn(eeg_ffn_input)
        eeg_final = self.eeg_norm2(eeg_ffn_input + self.dropout(eeg_ffn_out))
        
        # Audio FFN (轻微修改，保持对比学习稳定性)
        audio_ffn_input = audio_after_self + self.dropout(audio_cross_modal) * 0.3  # 减少audio修改幅度
        audio_ffn_out = self.audio_ffn(audio_ffn_input)
        audio_final = self.audio_norm2(audio_ffn_input + self.dropout(audio_ffn_out))
        
        # 最终控制audio对齐强度
        audio_final = audio_final * 0.3 + audio * 0.7  # 保持大部分原始audio特征
        
        # 最终投影
        aligned_eeg = self.eeg_final_proj(eeg_final) + eeg  # 保持残差连接
        aligned_audio = self.audio_final_proj(audio_final) + audio * 0.7  # 控制audio修改
        
        return aligned_eeg, aligned_audio


class CLARA(nn.Module):
    """CLARA (Cross-modal Low-rank Alignment) - 真正的Shared Low-Rank Alignment机制"""
    def __init__(self, embed_dim=256, num_heads=4, ffn_hidden_factor=2, low_rank_factor=0.5, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 使用较小的FFN扩张倍数，避免过拟合
        ffn_hidden_dim = embed_dim * ffn_hidden_factor
        self.low_rank_dim = max(1, int(embed_dim * low_rank_factor))
        
        # === EEG模态处理链 ===
        # 简化的自注意力 - Multi-Headed Attention
        self.eeg_self_query = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_key = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_value = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_proj = nn.Linear(embed_dim, embed_dim)
        self.eeg_norm1 = nn.LayerNorm(embed_dim)
        
        # EEG的FFN（Feed Forward）
        self.eeg_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.eeg_norm2 = nn.LayerNorm(embed_dim)
        
        # === Audio模态处理链 ===
        # Audio的简化自注意力 - Multi-Headed Attention
        self.audio_self_query = nn.Linear(embed_dim, embed_dim)
        self.audio_self_key = nn.Linear(embed_dim, embed_dim)
        self.audio_self_value = nn.Linear(embed_dim, embed_dim)
        self.audio_self_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_norm1 = nn.LayerNorm(embed_dim)
        
        # Audio的FFN（Feed Forward）
        self.audio_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.audio_norm2 = nn.LayerNorm(embed_dim)
        
        # === 关键：Shared Low-Rank Alignment (CLARA的核心机制) ===
        # WU: 投影到共享低秩空间（图片中的上行箭头）
        self.W_U_eeg = nn.Linear(embed_dim, self.low_rank_dim)     # EEG -> 共享空间
        self.W_U_audio = nn.Linear(embed_dim, self.low_rank_dim)   # Audio -> 共享空间
        
        # H: 共享交互层（图片中间的 f|H 部分）
        self.shared_interaction = nn.Sequential(
            nn.Linear(self.low_rank_dim, self.low_rank_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # WD: 从共享空间投影回模态空间（图片中的下行箭头）
        self.W_D_eeg = nn.Linear(self.low_rank_dim, embed_dim)     # 共享空间 -> EEG
        self.W_D_audio = nn.Linear(self.low_rank_dim, embed_dim)   # 共享空间 -> Audio
        
        # 最终输出投影
        self.eeg_final_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_final_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def _simplified_self_attention(self, x, query_proj, key_proj, value_proj, out_proj):
        """简化的自注意力机制 - 对应图片中的Multi-Headed Attention"""
        B = x.size(0)
        
        # 投影到query, key, value
        q = query_proj(x).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        k = key_proj(x).view(B, self.num_heads, self.head_dim)    # [B, H, d]
        v = value_proj(x).view(B, self.num_heads, self.head_dim)  # [B, H, d]
        
        # 计算自注意力分数 (局部操作，每个样本独立)
        attn_scores = torch.sum(q * k, dim=-1) / np.sqrt(self.head_dim)  # [B, H]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H]
        
        # 应用注意力权重
        attended = torch.einsum('bh,bhd->bhd', attn_weights, v)  # [B, H, d]
        attended = attended.reshape(B, -1)  # [B, embed_dim]
        
        # 输出投影
        return out_proj(attended)
    
    def forward(self, eeg, audio):
        """
        CLARA架构：Multi-Headed Attention -> Shared Low-Rank Alignment -> Feed Forward
        完全对应CLARA论文图片中的流程
        """
        B = eeg.size(0)
        
        # === 第一阶段：Multi-Headed Attention + Add & Norm ===
        # EEG自注意力
        eeg_self_attn = self._simplified_self_attention(
            eeg, self.eeg_self_query, self.eeg_self_key, 
            self.eeg_self_value, self.eeg_self_proj
        )
        eeg_after_self = self.eeg_norm1(eeg + self.dropout(eeg_self_attn))  # Add & Norm
        
        # Audio自注意力
        audio_self_attn = self._simplified_self_attention(
            audio, self.audio_self_query, self.audio_self_key,
            self.audio_self_value, self.audio_self_proj
        )
        audio_after_self = self.audio_norm1(audio + self.dropout(audio_self_attn))  # Add & Norm
        
        # === 第二阶段：Shared Low-Rank Alignment (CLARA的核心机制) ===
        # 步骤1: WU - 投影到共享低秩空间
        eeg_U = self.W_U_eeg(eeg_after_self)     # [B, low_rank_dim]
        audio_U = self.W_U_audio(audio_after_self)  # [B, low_rank_dim]
        
        # 步骤2: H - 在共享空间中进行交互
        # Element-wise product表示跨模态交互
        interaction_H = eeg_U * audio_U          # [B, low_rank_dim] 
        # 通过共享交互层
        interaction_H = self.shared_interaction(interaction_H)  # [B, low_rank_dim]
        
        # 步骤3: WD - 投影回各自模态空间  
        eeg_feedback = self.W_D_eeg(interaction_H)    # [B, embed_dim]
        audio_feedback = self.W_D_audio(interaction_H)  # [B, embed_dim]
        
        # === 第三阶段：Feed Forward + Add & Norm ===
        # EEG FFN with shared low-rank feedback
        eeg_ffn_input = eeg_after_self + self.dropout(eeg_feedback)  # 添加共享对齐反馈
        eeg_ffn_out = self.eeg_ffn(eeg_ffn_input)
        eeg_final = self.eeg_norm2(eeg_ffn_input + self.dropout(eeg_ffn_out))  # Add & Norm
        
        # Audio FFN with shared low-rank feedback (控制强度)
        audio_ffn_input = audio_after_self + self.dropout(audio_feedback) * 0.3  # 减少audio修改幅度
        audio_ffn_out = self.audio_ffn(audio_ffn_input)
        audio_final = self.audio_norm2(audio_ffn_input + self.dropout(audio_ffn_out))  # Add & Norm
        
        # 最终控制audio对齐强度，保持对比学习稳定性
        audio_final = audio_final * 0.3 + audio * 0.7  # 保持大部分原始audio特征
        
        # 最终投影
        aligned_eeg = self.eeg_final_proj(eeg_final) + eeg  # 保持残差连接
        aligned_audio = self.audio_final_proj(audio_final) + audio * 0.7  # 控制audio修改
        
        return aligned_eeg, aligned_audio


class CLIPModel(nn.Module):
    def __init__(self, eeg_model, audio_model, fusion_method='cross_attention'):
        super(CLIPModel, self).__init__()
        self.eeg_model = eeg_model
        self.audio_model = audio_model
        self.fusion_method = fusion_method
        
        # 修改投影层维度为256以匹配注意力
        self.eeg_proj = nn.Sequential(
            nn.Linear(200, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        
        # 初始化融合模块
        if fusion_method == 'cross_attention':
            self.fusion_module = CrossModalAttention(dim=256)
        elif fusion_method == 'simple_fusion':
            self.fusion_module = CrossModalFusion_Simple(embed_dim=256)
        elif fusion_method == 'bidirectional_fusion':
            self.fusion_module = CrossModalFusion_Bidirectional(embed_dim=256)
        elif fusion_method == 'clara_enhanced':
            self.fusion_module = CLARA_Enhanced(embed_dim=256)
        elif fusion_method == 'clara':
            self.fusion_module = CLARA(embed_dim=256)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}. Choose from: 'cross_attention', 'simple_fusion', 'bidirectional_fusion', 'clara_enhanced', 'clara'")
        
        # 最终分类器
        self.classifier = nn.Linear(256, 1)

    def forward(self, eeg, target_audio, negative_audio=None):
        # EEG特征提取
        eeg_emb = self.eeg_model(eeg)
        eeg_emb = self.eeg_proj(eeg_emb)  # [B,256]
        
        # 音频特征提取
        target_audio_emb = self.audio_model(target_audio).last_hidden_state.mean(1)    
        target_audio_emb = self.audio_proj(target_audio_emb)  # [B,256]
        
        # 根据融合方法进行跨模态交互
        if self.fusion_method == 'cross_attention':
            # 原始CrossModalAttention - 单向对齐
            fused_eeg = self.fusion_module(eeg_emb, target_audio_emb)  # [B,256]
            final_audio_emb = target_audio_emb  # 保持原始音频特征
        elif self.fusion_method == 'simple_fusion':
            # 简化跨模态融合 - 单向对齐，保持音频特征稳定
            fused_eeg = self.fusion_module(eeg_emb, target_audio_emb)  # [B,256]
            final_audio_emb = target_audio_emb  # 保持原始音频特征用于对比学习
        elif self.fusion_method == 'bidirectional_fusion':
            # 双向跨模态融合版本
            aligned_eeg, aligned_audio = self.fusion_module(eeg_emb, target_audio_emb)
            fused_eeg = aligned_eeg  # [B,256]
            final_audio_emb = aligned_audio  # 使用双向对齐后的音频特征
        elif self.fusion_method == 'clara_enhanced':
            # CLARA增强版本
            fused_eeg, aligned_audio = self.fusion_module(eeg_emb, target_audio_emb)
            final_audio_emb = aligned_audio  # 使用增强对齐后的音频特征
        elif self.fusion_method == 'clara':
            # CLARA (真正的Shared Low-Rank Alignment)
            fused_eeg, aligned_audio = self.fusion_module(eeg_emb, target_audio_emb)
            final_audio_emb = aligned_audio  # 使用CLARA对齐后的音频特征
        
        if negative_audio is not None:
            negative_audio_emb = self.audio_model(negative_audio).last_hidden_state.mean(1)
            negative_audio_emb = self.audio_proj(negative_audio_emb)
            
            # 处理负样本
            if self.fusion_method == 'bidirectional_fusion':
                # 双向融合需要对负样本也进行对齐
                _, aligned_negative_audio = self.fusion_module(eeg_emb, negative_audio_emb)
                negative_audio_emb = aligned_negative_audio
            elif self.fusion_method == 'clara_enhanced':
                # CLARA增强版本需要对负样本也进行对齐
                _, aligned_negative_audio = self.fusion_module(eeg_emb, negative_audio_emb)
                negative_audio_emb = aligned_negative_audio
            elif self.fusion_method == 'clara':
                # CLARA需要对负样本也进行对齐
                _, aligned_negative_audio = self.fusion_module(eeg_emb, negative_audio_emb)
                negative_audio_emb = aligned_negative_audio
            # 其他方法保持负样本音频特征不变以保持对比学习稳定性
            
            return fused_eeg, final_audio_emb, negative_audio_emb
        
        return fused_eeg, final_audio_emb

    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重"""
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重，允许部分匹配
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                print(f"Loaded: {k}")
            else:
                print(f"Skipped: {k} (shape mismatch or not found)")
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Successfully loaded {len(pretrained_dict)} layers from pretrained model")


def train_model(model, train_loader, val_loader, args, device):
    """微调训练函数"""
    model = model.to(device)
    
    # 优化器配置 - 微调使用较小的学习率
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95) if args.opt_betas is None else args.opt_betas,
        eps=args.opt_eps
    )
    
    # 设置学习率调度器
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=args.warmup_lr/args.lr,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )
    
    clip_loss_fn = ClipLoss(initial_temp=args.temperature)
    clip_loss_fn.to(device)
    
    best_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting fine-tuning for {args.epochs} epochs")
    print(f"Optimizer: {args.opt}, LR: {args.lr}, Weight Decay: {args.weight_decay}")

    for epoch in range(args.start_epoch, args.epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (eeg, target_audio, negative_audio, _) in enumerate(train_pbar):
            eeg = eeg.to(device)
            target_audio = target_audio.to(device)
            
            optimizer.zero_grad()
            
            # 计算CLIP loss
            eeg_features, audio_features = model(eeg, target_audio)
            loss = clip_loss_fn(eeg_features, audio_features)
            
            loss.backward()
            
            # 梯度裁剪
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

            # 更新进度条
            if batch_idx % 20 == 0:
                avg_loss = epoch_train_loss / num_batches
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{avg_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 学习率调度
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, device, clip_loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if args.save_ckpt:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'warmup_scheduler_state_dict': warmup_scheduler.state_dict() if args.warmup_epochs > 0 else None,
                    'best_accuracy': best_accuracy,
                    'args': args,
                }, os.path.join(args.output_dir, f'best_model_acc_{best_accuracy:.4f}.pth'))

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        print(f"Best Val Accuracy: {best_accuracy:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}\n")
    
    return model, train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, device, clip_loss_fn=None):
    """评估模型性能"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0
    
    if clip_loss_fn is None:
        clip_loss_fn = ClipLoss()
        clip_loss_fn.to(device)

    with torch.no_grad():
        for batch in test_loader:
            eeg, target_audio, negative_audio, _ = batch
            eeg = eeg.to(device)
            target_audio = target_audio.to(device)
            negative_audio = negative_audio.to(device)

            # 计算特征
            eeg_feat, target_feat, neg_feat = model(eeg, target_audio, negative_audio)

            # 计算loss
            loss = clip_loss_fn(eeg_feat, target_feat)
            total_loss += loss.item()
            num_batches += 1

            # 计算概率
            probs_segment_A = clip_loss_fn.get_probabilities(eeg_feat, target_feat)
            probs_segment_B = clip_loss_fn.get_probabilities(eeg_feat, neg_feat)

            # 计算对角线元素
            diag_A = probs_segment_A.diagonal(offset=0, dim1=0, dim2=1)
            diag_B = probs_segment_B.diagonal(offset=0, dim1=0, dim2=1)

            # 预测
            predictions = (diag_A > diag_B).long()

            # 创建标签
            labels = torch.zeros(eeg.size(0), dtype=torch.long, device=device)
            labels[:target_audio.size(0)] = 1

            # 更新正确预测数和样本总数
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # 计算平均loss和准确率
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})')
    
    return avg_loss, accuracy


def cross_validate_model(args, ds_init, data, device, subject_id=""):
    """交叉验证函数"""
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"\n=== Subject: {subject_id} | Fold {fold + 1}/{args.n_folds} ===")
        
        # 创建模型
        eeg_model = EEGEncoder(args, device)
        audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        model = CLIPModel(eeg_model, audio_model, fusion_method=args.fusion_method)
        
        # 加载预训练权重
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            model.load_pretrained_weights(args.pretrained_model)
        else:
            print("Warning: No pretrained model found, training from scratch")

        # 准备数据
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        train_dataset = DownstreamDataset(train_data)
        val_dataset = DownstreamDataset(val_data)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem
        )

        # 训练模型
        trained_model, train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, args, device
        )

        # 最终评估
        _, final_accuracy = evaluate_model(trained_model, val_loader, device)
        accuracies.append(final_accuracy)
        
        # 保存当前fold的结果
        fold_results = {
            'subject_id': subject_id,
            'fold': fold + 1,
            'final_accuracy': final_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        fold_save_path = os.path.join(args.output_dir, f'{subject_id}_fold_{fold+1}_results.json')
        with open(fold_save_path, 'w') as f:
            json.dump(fold_results, f, indent=2)

    # 计算总体结果
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    results = f"=== Subject: {subject_id} Cross-Validation Results ===\n"
    results += f"Fold Accuracies: {[f'{acc:.4f}' for acc in accuracies]}\n"
    results += f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n"
    results += f"Best Fold Accuracy: {max(accuracies):.4f}\n\n"

    print(results)

    # 保存结果到txt文件
    results_file = os.path.join(args.output_dir, "finetune_results.txt")
    with open(results_file, "a") as f:
        f.write(results)
    
    return mean_accuracy, std_accuracy, accuracies


def main():
    """主函数 - 微调版本"""
    args, ds_init = get_args()
    set_random_seed(args.seed)
    
    # 初始化分布式训练（如果需要）
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载下游任务数据
    print(f"Loading {args.dataset_type} dataset from {args.data_path}")
    subject_data = load_subject_data(args.data_path)
    
    all_results = {}
    
    # 对每个subject进行交叉验证
    for subject_id, df in subject_data.items():
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_id}")
        print(f"Data shape: {df.shape}")
        print(f"Using fusion method: {args.fusion_method}")
        if args.fusion_method == 'clara':
            print("CLARA (Cross-modal Low-rank Alignment) - True Shared Low-Rank Alignment")
        elif args.fusion_method == 'simple_fusion':
            print("Simple Cross-Modal Fusion - Best performing simple version")
        elif args.fusion_method == 'bidirectional_fusion':
            print("Bidirectional Cross-Modal Fusion - Dual alignment with controlled strength")
        elif args.fusion_method == 'clara_enhanced':
            print("CLARA Enhanced - Self-attention + FFN + stable cross-modal interaction")
        elif args.fusion_method == 'cross_attention':
            print("CrossModalAttention - Stable baseline")
        print(f"{'='*60}")
        
        mean_acc, std_acc, fold_accs = cross_validate_model(
            args, ds_init, df, device, subject_id=subject_id
        )
        
        all_results[subject_id] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': fold_accs
        }
    
    # 保存所有结果
    final_results_path = os.path.join(args.output_dir, 'all_subjects_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 计算总体统计
    all_mean_accs = [results['mean_accuracy'] for results in all_results.values()]
    overall_mean = np.mean(all_mean_accs)
    overall_std = np.std(all_mean_accs)
    
    summary = f"\n{'='*60}\n"
    summary += f"OVERALL RESULTS SUMMARY\n"
    summary += f"{'='*60}\n"
    summary += f"Number of subjects: {len(all_results)}\n"
    summary += f"Overall mean accuracy: {overall_mean:.4f} ± {overall_std:.4f}\n"
    summary += f"Best subject accuracy: {max(all_mean_accs):.4f}\n"
    summary += f"Worst subject accuracy: {min(all_mean_accs):.4f}\n"
    summary += f"{'='*60}\n"
    
    print(summary)
    
    # 保存总结到文件
    with open(os.path.join(args.output_dir, "finetune_results.txt"), "a") as f:
        f.write(summary)
    
    print("Fine-tuning completed!")


if __name__ == "__main__":
    main() 