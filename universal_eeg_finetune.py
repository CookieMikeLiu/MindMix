"""
通用EEG微调分类框架

支持多种数据集：
- EEG4EMO: EEG-Audio配对情绪分类数据
- KUL: EEG-Audio注意力分类数据（支持对比学习评估）
- DTU: EEG-Audio注意力分类数据（支持对比学习评估）
- 可扩展到其他数据集

支持多种微调策略：
1. eeg_only: 仅使用EEG编码器进行分类
2. multimodal_real: 使用真实的EEG-Audio数据进行多模态微调
   - 自动检测数据格式（3元组 vs 4元组）
   - 4元组数据使用类似MindMix_clip_finetune.py的对比学习评估
3. multimodal_prototype: 使用EEG+伪音频原型进行多模态微调

Usage:
    # EEG4EMO情绪分类
    python universal_eeg_finetune.py --dataset EEG4EMO --strategy multimodal_real --fusion_method clara
    
    # KUL/DTU对比学习评估
    python universal_eeg_finetune.py --dataset KUL --strategy multimodal_real --fusion_method clara
    
    # 对比学习评估示例
    python run_contrastive_example.py
"""

import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", message=".*?torch.distributed.*?")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import random
import argparse
import json
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
from transformers import Wav2Vec2Model
from modeling_finetune_2 import labram_base_patch200_200
from einops import rearrange
from timm.models import create_model
import utils
from collections import OrderedDict

# 禁用transformers的警告
import transformers
transformers.logging.set_verbosity_error()
torch.set_warn_always(False)

# Early Stopping参数
EARLY_STOPPING_PATIENCE = 50

def get_args():
    parser = argparse.ArgumentParser('Universal EEG Fine-tuning Framework', add_help=False)
    
    # 基础训练参数
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')  # 从1e-5增加到5e-5
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup epochs')  # 从3增加到5
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='warmup learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='minimum learning rate')
    parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping')
    
    # 数据集参数
    parser.add_argument('--dataset', default='EEG4EMO', type=str, 
                        choices=['EEG4EMO', 'KUL', 'DTU', 'ESAA'],
                        help='Dataset to use')
    parser.add_argument('--data_path', default='Dataset\EEG4EMO\preprocessed_pair', type=str,
                        help='Path to dataset')
    
    # 微调策略参数
    parser.add_argument('--strategy', default='eeg_only', type=str,
                        choices=['eeg_only', 'multimodal_real', 'multimodal_prototype'],
                        help='Fine-tuning strategy')
    parser.add_argument('--fusion_method', default='clara', type=str,
                        choices=['cross_attention', 'simple_fusion', 'bidirectional_fusion', 'clara_enhanced', 'clara'],
                        help='Cross-modal fusion method')
    
    # 模型参数
    parser.add_argument('--pretrained_model', default='checkpoints\\checkpoint.pth',
                        help='Path to pretrained model (for eeg_only: EEG encoder, for multimodal: fusion model)')
    parser.add_argument('--finetune', default='checkpoints/labram-base.pth',
                        help='EEG encoder checkpoint')
    parser.add_argument('--use_auditory_type', action='store_true', default=False,
                        help='Whether to use auditory type-specific aligners')
    
    # 其他参数
    parser.add_argument('--n_folds', default=5, type=int, help='Number of CV folds')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--output_dir', default='./universal_finetune_results', help='Output directory')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--pin_mem', action='store_true', default=True, help='Pin memory')
    parser.add_argument('--save_ckpt', action='store_true', default=True, help='Save checkpoints')
    
    # LaBraM模型参数
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, help='Model name')
    parser.add_argument('--input_size', default=400, type=int, help='EEG input size')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--qkv_bias', action='store_true', default=True)
    parser.add_argument('--rel_pos_bias', action='store_true', default=True)
    parser.add_argument('--abs_pos_emb', action='store_true', default=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true', default=True)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    
    return parser.parse_args()


def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DatasetLoader:
    """通用数据集加载器"""
    
    @staticmethod
    def load_eeg4emo(data_path):
        """加载EEG4EMO数据集"""
        subject_data = {}
        
        for subject_folder in os.listdir(data_path):
            subject_path = os.path.join(data_path, subject_folder)
            if os.path.isdir(subject_path):
                print(f"Loading subject: {subject_folder}")
                
                # 读取标签CSV文件
                label_file = os.path.join(subject_path, f"{subject_folder}_labels.csv")
                if not os.path.exists(label_file):
                    print(f"  Warning: Label file not found: {label_file}")
                    continue
                
                labels_df = pd.read_csv(label_file)
                emotion_labels = labels_df['Emotion_Label'].values
                print(f"  Found {len(emotion_labels)} video labels")
                
                all_subject_data = []
                
                for filename in os.listdir(subject_path):
                    if filename.endswith('.pkl') and 'video' in filename:
                        file_path = os.path.join(subject_path, filename)
                        try:
                            # 提取video ID
                            video_id = None
                            if 'video_' in filename:
                                try:
                                    video_id = int(filename.split('video_')[1].split('_')[0])
                                except:
                                    continue
                            
                            # 从CSV获取对应的标签（video_id从1开始，数组索引从0开始）
                            if video_id is None or video_id < 1 or video_id > len(emotion_labels):
                                print(f"  Invalid video_id {video_id} for {filename}")
                                continue
                            
                            emotion_label = emotion_labels[video_id - 1]  # 转换为数组索引
                            
                            # 跳过标签为0的数据
                            if emotion_label == 0.0:
                                print(f"  Skipping {filename}: label=0")
                                continue
                            
                            with open(file_path, 'rb') as f:
                                data = pickle.load(f)
                            
                            if isinstance(data, list) and len(data) > 0:
                                sample = data[0]
                                if 'eeg' in sample and 'audio' in sample:
                                    processed_samples = []
                                    for sample in data:
                                        processed_sample = {
                                            'eeg': sample['eeg'].astype(np.float32),
                                            'audio': sample['audio'].astype(np.float32),
                                            'video_id': video_id,
                                            'label': int(emotion_label)  # 使用CSV中的实际标签
                                        }
                                        processed_samples.append(processed_sample)
                                    
                                    df_samples = pd.DataFrame(processed_samples)
                                    print(f"  {filename}: {len(df_samples)} samples (label: {int(emotion_label)})")
                                    all_subject_data.append(df_samples)
                                        
                        except Exception as e:
                            print(f"  Error loading {filename}: {str(e)}")
                            continue
                
                if all_subject_data:
                    subject_df = pd.concat(all_subject_data, ignore_index=True)
                    print(f"  Total samples: {len(subject_df)}")
                    print(f"  Label distribution: {subject_df['label'].value_counts().to_dict()}")
                    subject_data[subject_folder] = subject_df
                    
        return subject_data
    
    @staticmethod
    def load_kul_dtu(data_path):
        """加载KUL/DTU数据集"""
        subject_data = {}
        for filename in os.listdir(data_path):
            if filename.endswith('.pkl'):
                subject_id = os.path.splitext(filename)[0]
                file_path = os.path.join(data_path, filename)
                data = pd.read_pickle(file_path)
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                subject_data[subject_id] = data
        return subject_data
    
    @staticmethod
    def load_esaa(data_path):
        """加载ESAA数据集"""
        subject_data = {}
        
        # 按subject分组文件
        subject_files = {}
        for filename in os.listdir(data_path):
            if filename.endswith('.pkl'):
                parts = filename.split('_')
                subject_id = parts[0]  # S1, S2, etc.
                if subject_id not in subject_files:
                    subject_files[subject_id] = []
                subject_files[subject_id].append(filename)
        
        print(f"Found {len(subject_files)} ESAA subjects")
        
        for subject_id, files in subject_files.items():
            print(f"Processing subject {subject_id} with {len(files)} files")
            
            all_subject_data = []
            
            for filename in files:
                file_path = os.path.join(data_path, filename)
                try:
                    # 尝试加载数据
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    except ValueError as e:
                        if "unsupported pickle protocol" in str(e):
                            print(f"  Warning: Skipping {filename} due to pickle protocol issue")
                            continue
                        else:
                            raise e
                    
                    # 处理数据格式 - ESAA和KUL/DTU格式相同
                    if isinstance(data, dict):
                        # 如果是字典格式，检查是否包含必要的键
                        if 'eeg' in data and 'target_audio' in data and 'negative_audio' in data:
                            # 使用stimuli_label作为标签
                            label = data.get('stimuli_label', 0)
                            sample_data = {
                                'eeg': data['eeg'],
                                'target_audio': data['target_audio'],
                                'negative_audio': data['negative_audio'],
                                'stimuli_label': label,  # 使用stimuli_label作为标签
                                'subject_id': data.get('subject_id', subject_id),
                                'event_num': data.get('event_num', 0),
                                'segment_index': data.get('segment_index', 0),
                            }
                            all_subject_data.append(pd.DataFrame([sample_data]))
                        else:
                            print(f"  Warning: {filename} does not contain required keys (eeg, target_audio, negative_audio)")
                    elif isinstance(data, pd.DataFrame):
                        # 如果已经是DataFrame，检查必要的列
                        if 'eeg' in data.columns and 'target_audio' in data.columns and 'negative_audio' in data.columns:
                            all_subject_data.append(data)
                        else:
                            print(f"  Warning: {filename} DataFrame does not contain required columns")
                    elif isinstance(data, list):
                        # 如果是列表，转换为DataFrame
                        df = pd.DataFrame(data)
                        if 'eeg' in df.columns and 'target_audio' in df.columns and 'negative_audio' in df.columns:
                            all_subject_data.append(df)
                        else:
                            print(f"  Warning: {filename} list data does not contain required columns")
                    else:
                        print(f"  Warning: Unknown data format in {filename}: {type(data)}")
                        
                except Exception as e:
                    print(f"  Error loading {filename}: {str(e)}")
                    continue
            
            if all_subject_data:
                # 合并该subject的所有数据
                subject_df = pd.concat(all_subject_data, ignore_index=True)
                print(f"  Subject {subject_id}: {len(subject_df)} total samples")
                if 'stimuli_label' in subject_df.columns:
                    print(f"  Label distribution: {subject_df['stimuli_label'].value_counts().to_dict()}")
                subject_data[subject_id] = subject_df
            else:
                print(f"  Warning: No valid data for subject {subject_id}")
        
        return subject_data
    
    @staticmethod
    def load_dataset(dataset_name, data_path):
        """根据数据集名称加载相应数据"""
        if dataset_name == 'EEG4EMO':
            return DatasetLoader.load_eeg4emo(data_path)
        elif dataset_name in ['KUL', 'DTU']:
            return DatasetLoader.load_kul_dtu(data_path)
        elif dataset_name == 'ESAA':
            return DatasetLoader.load_esaa(data_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")


class UniversalDataset(Dataset):
    """通用数据集类"""
    
    def __init__(self, data, dataset_type='EEG4EMO', strategy='multimodal_real'):
        self.data = data
        self.dataset_type = dataset_type
        self.strategy = strategy
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        if self.dataset_type == 'EEG4EMO':
            eeg = sample['eeg']
            label = sample['label']
            binary_label = 1 if label == 1 else 0  # 转换为0/1标签
            
            eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
            label_tensor = torch.tensor(binary_label, dtype=torch.long)
            
            if self.strategy in ['multimodal_real', 'multimodal_prototype']:
                # 多模态策略需要返回音频数据
                audio = sample['audio']
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
                return eeg_tensor, audio_tensor, label_tensor
            else:
                # 纯EEG策略
                return eeg_tensor, label_tensor
                
        elif self.dataset_type in ['KUL', 'DTU']:
            eeg = sample['eeg']
            target_audio = sample['target_audio']
            negative_audio = sample['negetive_audio']
            label = sample['attended_label']
            
            eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
            target_audio_tensor = torch.tensor(target_audio, dtype=torch.float32)
            negative_audio_tensor = torch.tensor(negative_audio, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            return eeg_tensor, target_audio_tensor, negative_audio_tensor, label_tensor
        
        elif self.dataset_type == 'ESAA':
            eeg = sample['eeg']
            target_audio = sample['target_audio']
            negative_audio = sample['negative_audio']
            label = sample['stimuli_label']
            
            eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
            target_audio_tensor = torch.tensor(target_audio, dtype=torch.float32)
            negative_audio_tensor = torch.tensor(negative_audio, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            return eeg_tensor, target_audio_tensor, negative_audio_tensor, label_tensor
        
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")


# 从原始文件导入必要的模型组件
ch_names = [
    "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1",
    "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7",
    "PO3", "O1", "IZ", "OZ", "POZ", "PZ", "CPZ", "FPZ", "FP2", "AF8", "AF4", "AFZ", "FZ",
    "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ", "C2", "C4", "C6", "T8",
    "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2"
]


def get_models(args):
    """创建EEG编码器模型"""
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=0,  # 不需要分类头
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
    """EEG编码器"""
    def __init__(self, args, device):
        super(EEGEncoder, self).__init__()
        self.args = args
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        model = get_models(self.args)

        if self.args.finetune:
            if self.args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.args.finetune, map_location='cpu')

            print("Load EEG encoder checkpoint from %s" % self.args.finetune)
            checkpoint_model = self.get_checkpoint_model(checkpoint)
            self.adjust_checkpoint_keys(checkpoint_model, model)
            utils.load_state_dict(model, checkpoint_model, prefix=self.args.model_prefix)

        model.to(self.device)
        return model

    def forward(self, eeg_data):
        # 将EEG数据reshape为模型期望的格式
        eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', A=2, T=200)
        output = self.model(eeg_data)
        return output

    def get_checkpoint_model(self, checkpoint):
        """从checkpoint中提取模型状态字典"""
        checkpoint_model = None
        for model_key in self.args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Loaded state_dict by model_key = %s" % model_key)
                break
        return checkpoint_model if checkpoint_model is not None else checkpoint

    def adjust_checkpoint_keys(self, checkpoint_model, model):
        """调整checkpoint中的键名以匹配当前模型"""
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


def cross_validate_subject(args, subject_data, subject_id, device):
    """对单个subject进行交叉验证"""
    from sklearn.model_selection import KFold
    from universal_models import create_model
    from universal_trainer import create_trainer
    
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    accuracies = []
    f1_scores = []
    
    print(f"\n=== Subject: {subject_id} ===")
    print(f"Data shape: {subject_data.shape}")
    print(f"Strategy: {args.strategy}")
    
    # 定义collate函数来确保数据在正确设备上
    def collate_fn(batch):
        if args.dataset == 'EEG4EMO':
            if args.strategy in ['multimodal_real', 'multimodal_prototype']:
                eeg_list, audio_list, label_list = zip(*batch)
                eeg = torch.stack(eeg_list).to(device)
                audio = torch.stack(audio_list).to(device)
                labels = torch.stack(label_list).to(device)
                return eeg, audio, labels
            else:
                eeg_list, label_list = zip(*batch)
                eeg = torch.stack(eeg_list).to(device)
                labels = torch.stack(label_list).to(device)
                return eeg, labels
        elif args.dataset == 'ESAA':
            # ESAA和KUL/DTU格式相同，都是对比学习任务
            eeg_list, target_audio_list, negative_audio_list, label_list = zip(*batch)
            eeg = torch.stack(eeg_list).to(device)
            target_audio = torch.stack(target_audio_list).to(device)
            negative_audio = torch.stack(negative_audio_list).to(device)
            labels = torch.stack(label_list).to(device)
            return eeg, target_audio, negative_audio, labels
        else:  # KUL/DTU
            eeg_list, target_audio_list, negative_audio_list, label_list = zip(*batch)
            eeg = torch.stack(eeg_list).to(device)
            target_audio = torch.stack(target_audio_list).to(device)
            negative_audio = torch.stack(negative_audio_list).to(device)
            labels = torch.stack(label_list).to(device)
            return eeg, target_audio, negative_audio, labels
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(subject_data)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        
        # 创建EEG编码器
        eeg_encoder = EEGEncoder(args, device)
        
        # 根据策略创建音频编码器
        audio_encoder = None
        if args.strategy in ['multimodal_real', 'multimodal_prototype']:
            audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            audio_encoder.to(device)  # 确保音频编码器在正确设备上
        
        # 创建模型
        model = create_model(
            args.strategy, 
            eeg_encoder, 
            audio_encoder, 
            args.fusion_method, 
            num_classes=2, 
            use_auditory_type=args.use_auditory_type
        )
        model.to(device)  # 确保整个模型在正确设备上
        
        # 加载预训练权重
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            model.load_pretrained_weights(args.pretrained_model)
            print(f"Loaded pretrained weights from {args.pretrained_model}")
        
        # 准备数据
        train_data = subject_data.iloc[train_idx]
        val_data = subject_data.iloc[val_idx]
        
        train_dataset = UniversalDataset(train_data, args.dataset, args.strategy)
        val_dataset = UniversalDataset(val_data, args.dataset, args.strategy)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # 减少worker数量避免设备问题
            pin_memory=False,  # 关闭pin_memory避免设备问题
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # 减少worker数量避免设备问题
            pin_memory=False,  # 关闭pin_memory避免设备问题
            collate_fn=collate_fn
        )
        
        # 创建训练器并训练
        trainer = create_trainer(args.strategy, model, device, args)
        best_accuracy, best_f1 = trainer.train(train_loader, val_loader)
        
        accuracies.append(best_accuracy)
        f1_scores.append(best_f1)
        
        print(f"Fold {fold + 1} Results: Acc={best_accuracy:.4f}, F1={best_f1:.4f}")
    
    # 计算总体结果
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    results = f"\n=== Subject {subject_id} Cross-Validation Results ===\n"
    results += f"Strategy: {args.strategy}\n"
    results += f"Fold Accuracies: {[f'{acc:.4f}' for acc in accuracies]}\n"
    results += f"Fold F1 Scores: {[f'{f1:.4f}' for f1 in f1_scores]}\n"
    results += f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n"
    results += f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n"
    results += f"Best Fold Accuracy: {max(accuracies):.4f}\n"
    results += f"Best Fold F1: {max(f1_scores):.4f}\n"
    
    print(results)
    
    # 保存结果
    results_filename = f"results_{args.dataset}_{args.strategy}_MindMix.txt"
    results_file = os.path.join(args.output_dir, results_filename)
    with open(results_file, "a") as f:
        f.write(results + "\n")
    
    return mean_accuracy, std_accuracy, mean_f1, std_f1


def main():
    """主函数"""
    args = get_args()
    set_random_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    print(f"Loading {args.dataset} dataset from {args.data_path}")
    subject_data = DatasetLoader.load_dataset(args.dataset, args.data_path)
    
    if len(subject_data) == 0:
        print("No valid subjects found! Please check the data path and format.")
        return
    
    print(f"\nLoaded {len(subject_data)} subjects")
    print(f"Strategy: {args.strategy}")
    if args.strategy != 'eeg_only':
        print(f"Fusion method: {args.fusion_method}")
    
    # 对每个subject进行交叉验证
    all_results = {}
    
    for subject_id, df in subject_data.items():
        if len(df) < args.n_folds:
            print(f"Warning: Subject {subject_id} has only {len(df)} samples, skipping...")
            continue
        
        mean_acc, std_acc, mean_f1, std_f1 = cross_validate_subject(
            args, df, subject_id, device
        )
        
        all_results[subject_id] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_f1': mean_f1,
            'std_f1': std_f1
        }
    
    if len(all_results) == 0:
        print("No subjects were successfully processed!")
        return
    
    # 计算总体统计
    all_mean_accs = [results['mean_accuracy'] for results in all_results.values()]
    all_mean_f1s = [results['mean_f1'] for results in all_results.values()]
    overall_mean_acc = np.mean(all_mean_accs)
    overall_std_acc = np.std(all_mean_accs)
    overall_mean_f1 = np.mean(all_mean_f1s)
    overall_std_f1 = np.std(all_mean_f1s)
    
    summary = f"\n{'='*60}\n"
    summary += f"OVERALL RESULTS SUMMARY\n"
    summary += f"{'='*60}\n"
    summary += f"Dataset: {args.dataset}\n"
    summary += f"Strategy: {args.strategy}\n"
    if args.strategy != 'eeg_only':
        summary += f"Fusion method: {args.fusion_method}\n"
    summary += f"Number of subjects: {len(all_results)}\n"
    summary += f"Overall mean accuracy: {overall_mean_acc:.4f} ± {overall_std_acc:.4f}\n"
    summary += f"Overall mean F1 score: {overall_mean_f1:.4f} ± {overall_std_f1:.4f}\n"
    summary += f"Best subject accuracy: {max(all_mean_accs):.4f}\n"
    summary += f"Best subject F1: {max(all_mean_f1s):.4f}\n"
    summary += f"{'='*60}\n"
    
    print(summary)
    
    # 保存总结和详细结果
    results_filename = f"results_{args.dataset}_{args.strategy}_MindMix.txt"
    with open(os.path.join(args.output_dir, results_filename), "a") as f:
        f.write(summary)
    
    import json
    json_filename = f"all_results_{args.dataset}_{args.strategy}_MindMix.json"
    with open(os.path.join(args.output_dir, json_filename), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Universal EEG fine-tuning completed!")


if __name__ == "__main__":
    main() 