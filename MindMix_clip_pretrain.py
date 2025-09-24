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

# 禁用transformers的警告
import transformers
transformers.logging.set_verbosity_error()

# 设置torch的警告级别
torch.set_warn_always(False)


def get_args():
    parser = argparse.ArgumentParser('LaBraM pre-training script for EEG-Audio cross-modal learning with fusion', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)  # 预训练使用更大的batch size
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)  # 更频繁的保存

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
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',  # 预训练使用较小的学习率
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',  # 预训练需要更长的warmup
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
    parser.add_argument('--finetune', default='./checkpoints/labram-base.pth',  # 预训练用labram-base
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

    parser.add_argument('--output_dir', default='./pretrain_fusion_checkpoints',
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
    parser.set_defaults(save_ckpt=True)  # 预训练默认保存

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=16, type=int)  # 预训练使用更多workers
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)  # 预训练默认使用pin_memory

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

    # 预训练特定参数
    parser.add_argument('--data_path', default='./Dataset/Pretraining/test', type=str,
                        help='Path to the dataset directory containing pkl files')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature for contrastive learning')
    parser.add_argument('--fusion_method', default='clara', type=str, 
                        choices=['cross_attention', 'simple_fusion', 'bidirectional_fusion', 'clara_enhanced', 'clara'],
                        help='Cross-modal fusion method to use')
    parser.add_argument('--use_auditory_type', action='store_true', default=False,
                        help='Whether to use auditory type-specific aligners in CLARA')

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


class PretrainDataset(Dataset):
    """预训练数据集类"""
    def __init__(self, data_path, use_auditory_type=False):
        self.data_path = data_path
        self.use_auditory_type = use_auditory_type
        self.samples = []
        # 定义可能的字段名映射
        self.field_mappings = {
            'eeg': ['eeg', 'eeg_data', 'eeg_signal'],
            'audio': ['audio', 'audio_data', 'wav', 'target_audio'],
            'auditory_type': ['auditory_type', 'audio_type', 'type']  # 新增auditory type的可能字段名
        }
        self.load_all_data()
        
    def _find_matching_field(self, data_columns, possible_fields):
        """在数据列中查找匹配的字段名"""
        for field in possible_fields:
            if field in data_columns:
                return field
        return None
        
    def load_all_data(self):
        """加载所有pkl文件中的数据"""
        print("正在加载预训练数据...")
        
        # 查找所有pkl文件
        pkl_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_files.append(os.path.join(root, file))
        
        if len(pkl_files) == 0:
            raise ValueError(f"在路径 {self.data_path} 中没有找到.pkl文件")
        
        print(f"找到 {len(pkl_files)} 个pkl文件")
        
        for pkl_file in tqdm(pkl_files, desc="加载数据文件"):
            try:
                # 加载数据
                data = pd.read_pickle(pkl_file)
                
                # 转换为DataFrame
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'eeg' in data and 'audio' in data:
                        # 如果是单个样本的字典
                        data = pd.DataFrame([data])
                    else:
                        # 尝试将字典转换为DataFrame
                        data = pd.DataFrame.from_dict(data)
                
                if not isinstance(data, pd.DataFrame):
                    print(f"跳过文件 {pkl_file}: 无法转换为DataFrame格式")
                    continue
                
                # 查找EEG和音频字段
                eeg_field = self._find_matching_field(data.columns, self.field_mappings['eeg'])
                audio_field = self._find_matching_field(data.columns, self.field_mappings['audio'])
                
                if not eeg_field or not audio_field:
                    print(f"跳过文件 {pkl_file}: 找不到EEG或音频字段")
                    print(f"可用的列: {list(data.columns)}")
                    continue
                
                # 如果启用了auditory type，尝试查找相应字段
                auditory_type_field = None
                if self.use_auditory_type:
                    auditory_type_field = self._find_matching_field(data.columns, self.field_mappings['auditory_type'])
                    if not auditory_type_field:
                        print(f"警告：文件 {pkl_file} 中未找到auditory type字段，将使用默认值0")
                
                # 处理每个样本
                valid_samples = 0
                for idx, row in data.iterrows():
                    try:
                        eeg = row[eeg_field]
                        audio = row[audio_field]
                        
                        # 确保数据是numpy数组
                        if isinstance(eeg, list):
                            eeg = np.array(eeg)
                        if isinstance(audio, list):
                            audio = np.array(audio)
                        
                        # 验证数据
                        if not isinstance(eeg, np.ndarray) or not isinstance(audio, np.ndarray):
                            continue
                            
                        if len(eeg.shape) == 0 or len(audio.shape) == 0:
                            continue
                            
                        # 构建样本字典
                        sample = {
                            'eeg': eeg.astype(np.float32),
                            'audio': audio.astype(np.float32),
                            'source_file': pkl_file
                        }
                        
                        # 如果启用了auditory type，添加到样本中
                        if self.use_auditory_type:
                            if auditory_type_field and auditory_type_field in row:
                                sample['auditory_type'] = row[auditory_type_field]
                            else:
                                sample['auditory_type'] = 0  # 默认值
                            
                        self.samples.append(sample)
                        valid_samples += 1
                        
                    except Exception as e:
                        print(f"处理样本时出错 (文件: {pkl_file}, 索引: {idx}): {str(e)}")
                        continue
                
                print(f"从文件 {pkl_file} 加载了 {valid_samples} 个有效样本")
                
            except Exception as e:
                print(f"加载文件 {pkl_file} 时出错: {str(e)}")
                continue
        
        if len(self.samples) == 0:
            raise ValueError("没有找到有效的预训练数据！请检查数据路径和格式。")
            
        print(f"成功加载总计 {len(self.samples)} 个样本")
        
        # 打印第一个样本的形状信息
        if len(self.samples) > 0:
            first_sample = self.samples[0]
            print(f"样本数据形状:")
            print(f"EEG: {first_sample['eeg'].shape}")
            print(f"Audio: {first_sample['audio'].shape}")
            if self.use_auditory_type:
                print(f"包含 auditory type 信息")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.use_auditory_type:
            return torch.tensor(sample['eeg']), torch.tensor(sample['audio']), torch.tensor(sample['auditory_type'])
        else:
            return torch.tensor(sample['eeg']), torch.tensor(sample['audio'])


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
    
    def forward(self, eeg, audio, auditory_type=None):
        """
        CLARA架构：Multi-Headed Attention -> Shared Low-Rank Alignment -> Feed Forward
        完全对应CLARA论文图片中的流程
        
        Args:
            eeg: EEG特征 [B, embed_dim]
            audio: 音频特征 [B, embed_dim]
            auditory_type: 可选的auditory type索引 [B] (如果use_auditory_type=True时使用)
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
        if self.use_auditory_type and auditory_type is not None:
            # 使用type-specific aligner
            eeg_U = torch.zeros(B, self.low_rank_dim, device=eeg.device)
            for type_idx in range(len(self.type_aligners)):
                type_mask = (auditory_type == type_idx)
                if type_mask.any():
                    eeg_U[type_mask] = self.type_aligners[type_idx](eeg_after_self[type_mask])
        else:
            # 使用标准投影
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
    def __init__(self, eeg_model, audio_model, fusion_method='cross_attention', use_auditory_type=False):
        super(CLIPModel, self).__init__()
        self.eeg_model = eeg_model
        self.audio_model = audio_model
        self.fusion_method = fusion_method
        self.use_auditory_type = use_auditory_type
        
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
            self.fusion_module = CLARA(embed_dim=256, use_auditory_type=use_auditory_type)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}. Choose from: 'cross_attention', 'simple_fusion', 'bidirectional_fusion', 'clara_enhanced', 'clara'")
        
        # 最终分类器（在预训练中不使用）


    def forward(self, eeg, target_audio, negative_audio=None, auditory_type=None):
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
            fused_eeg, aligned_audio = self.fusion_module(eeg_emb, target_audio_emb, auditory_type)
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
                _, aligned_negative_audio = self.fusion_module(eeg_emb, negative_audio_emb, auditory_type)
                negative_audio_emb = aligned_negative_audio
            # 其他方法保持负样本音频特征不变以保持对比学习稳定性
            
            return fused_eeg, final_audio_emb, negative_audio_emb
        
        return fused_eeg, final_audio_emb


def pretrain_model(model, train_loader, val_loader, args):
    """预训练函数，基于原始训练函数修改"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 优化器配置
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
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting pre-training for {args.epochs} epochs")
    print(f"Optimizer: {args.opt}, LR: {args.lr}, Weight Decay: {args.weight_decay}")

    for epoch in range(args.start_epoch, args.epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            # 根据是否使用auditory type来解包数据
            if args.use_auditory_type:
                eeg, audio, auditory_type = [b.to(device) for b in batch]
            else:
                eeg, audio = [b.to(device) for b in batch]
                auditory_type = None
                
            optimizer.zero_grad()
            
            # 在预训练中，我们使用相同的音频作为target
            if args.use_auditory_type:
                eeg_features, audio_features = model(eeg, audio, auditory_type=auditory_type)
            else:
                eeg_features, audio_features = model(eeg, audio)
                
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
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.inference_mode():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in val_pbar:
                # 根据是否使用auditory type来解包数据
                if args.use_auditory_type:
                    eeg, audio, auditory_type = [b.to(device) for b in batch]
                else:
                    eeg, audio = [b.to(device) for b in batch]
                    auditory_type = None

                if args.use_auditory_type:
                    eeg_features, audio_features = model(eeg, audio, auditory_type=auditory_type)
                else:
                    eeg_features, audio_features = model(eeg, audio)
                    
                loss = clip_loss_fn(eeg_features, audio_features)

                val_loss += loss.item()
                num_val_batches += 1
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if args.save_ckpt:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'warmup_scheduler_state_dict': warmup_scheduler.state_dict() if args.warmup_epochs > 0 else None,
                    'loss': best_loss,
                    'args': args,
                }, os.path.join(args.output_dir, f'best_model_loss_{best_loss:.4f}.pth'))

        # Regular checkpoint saving
        if args.save_ckpt and (epoch + 1) % args.save_ckpt_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict() if args.warmup_epochs > 0 else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': args,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Best Val Loss: {best_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}\n")
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """保持原有的评估函数（在预训练中可能不会用到）"""
    model.eval()
    total_correct = 0
    total_samples = 0
    clip_loss = ClipLoss()

    with torch.no_grad():
        for batch in test_loader:
            eeg, target_audio, negative_audio, _ = batch
            eeg = eeg.to(device)
            target_audio = target_audio.to(device)
            negative_audio = negative_audio.to(device)

            # 计算特征
            eeg_feat, target_feat, neg_feat = model(eeg, target_audio, negative_audio)

            # 计算概率
            probs_segment_A = clip_loss.get_probabilities(eeg_feat, target_feat)
            probs_segment_B = clip_loss.get_probabilities(eeg_feat, neg_feat)

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

    # 输出结果
    if total_samples > 0:
        print(f'Total Accuracy: {total_correct / total_samples * 100:.2f}%')
    else:
        print('No samples to evaluate.')
    
    return total_correct / total_samples


def main():
    """主函数 - 预训练版本"""
    args, ds_init = get_args()
    set_random_seed(args.seed)
    
    # 初始化分布式训练（如果需要）
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集和数据加载器
    dataset = PretrainDataset(args.data_path, use_auditory_type=args.use_auditory_type)

    # 添加数据shuffle
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # 使用已设置的随机种子
    shuffled_dataset = torch.utils.data.Subset(dataset, indices)

    # 划分训练集和验证集
    train_size = int(0.9 * len(shuffled_dataset))
    val_size = len(shuffled_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        shuffled_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
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
    
    # 创建模型
    eeg_model = EEGEncoder(args, device)
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model = CLIPModel(eeg_model, audio_model, fusion_method=args.fusion_method, use_auditory_type=args.use_auditory_type)
    
    print(f"Using fusion method: {args.fusion_method}")
    if args.fusion_method == 'clara':
        print("CLARA (Cross-modal Low-rank Alignment) - True Shared Low-Rank Alignment")
        if args.use_auditory_type:
            print("Using auditory type-specific aligners")
        else:
            print("Using standard alignment without auditory type")
    elif args.fusion_method == 'simple_fusion':
        print("Simple Cross-Modal Fusion - Best performing simple version")
    elif args.fusion_method == 'bidirectional_fusion':
        print("Bidirectional Cross-Modal Fusion - Dual alignment with controlled strength")
    elif args.fusion_method == 'clara_enhanced':
        print("CLARA Enhanced - Self-attention + FFN + stable cross-modal interaction")
    elif args.fusion_method == 'cross_attention':
        print("CrossModalAttention - Stable baseline")
    
    # 开始预训练
    trained_model, train_losses, val_losses = pretrain_model(
        model, train_loader, val_loader, args
    )
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Pre-training completed!")


if __name__ == "__main__":
    main() 