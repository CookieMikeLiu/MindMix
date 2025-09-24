"""
通用EEG微调模型定义

包含三种微调策略的模型：
1. EEGOnlyModel: 纯EEG分类模型
2. MultimodalRealModel: 使用真实EEG-Audio数据的多模态模型
3. MultimodalPrototypeModel: 使用伪音频原型的多模态模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Model


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


class CLARA(nn.Module):
    """CLARA (Cross-modal Low-rank Alignment) 融合模块"""
    def __init__(self, embed_dim=256, num_heads=4, ffn_hidden_factor=2, low_rank_factor=0.5, dropout_rate=0.1, use_auditory_type=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_auditory_type = use_auditory_type
        
        ffn_hidden_dim = embed_dim * ffn_hidden_factor
        self.low_rank_dim = max(1, int(embed_dim * low_rank_factor))
        
        # EEG模态处理链
        self.eeg_self_query = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_key = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_value = nn.Linear(embed_dim, embed_dim)
        self.eeg_self_proj = nn.Linear(embed_dim, embed_dim)
        self.eeg_norm1 = nn.LayerNorm(embed_dim)
        
        self.eeg_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.eeg_norm2 = nn.LayerNorm(embed_dim)
        
        # Audio模态处理链
        self.audio_self_query = nn.Linear(embed_dim, embed_dim)
        self.audio_self_key = nn.Linear(embed_dim, embed_dim)
        self.audio_self_value = nn.Linear(embed_dim, embed_dim)
        self.audio_self_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_norm1 = nn.LayerNorm(embed_dim)
        
        self.audio_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.audio_norm2 = nn.LayerNorm(embed_dim)
        
        # Shared Low-Rank Alignment
        if self.use_auditory_type:
            self.type_aligners = nn.ModuleList([
                nn.Linear(embed_dim, self.low_rank_dim)
                for _ in range(3)  # 假设有3种auditory type
            ])
        else:
            self.W_U_eeg = nn.Linear(embed_dim, self.low_rank_dim)
            
        self.W_U_audio = nn.Linear(embed_dim, self.low_rank_dim)
        
        self.shared_interaction = nn.Sequential(
            nn.Linear(self.low_rank_dim, self.low_rank_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.W_D_eeg = nn.Linear(self.low_rank_dim, embed_dim)
        self.W_D_audio = nn.Linear(self.low_rank_dim, embed_dim)
        
        self.eeg_final_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_final_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def _simplified_self_attention(self, x, query_proj, key_proj, value_proj, out_proj):
        B = x.size(0)
        
        q = query_proj(x).view(B, self.num_heads, self.head_dim)
        k = key_proj(x).view(B, self.num_heads, self.head_dim)
        v = value_proj(x).view(B, self.num_heads, self.head_dim)
        
        attn_scores = torch.sum(q * k, dim=-1) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        attended = torch.einsum('bh,bhd->bhd', attn_weights, v)
        attended = attended.reshape(B, -1)
        
        return out_proj(attended)
    
    def forward(self, eeg, audio, auditory_type=None):
        B = eeg.size(0)
        
        # Multi-Headed Attention
        eeg_self_attn = self._simplified_self_attention(
            eeg, self.eeg_self_query, self.eeg_self_key, 
            self.eeg_self_value, self.eeg_self_proj
        )
        eeg_after_self = self.eeg_norm1(eeg + self.dropout(eeg_self_attn))
        
        audio_self_attn = self._simplified_self_attention(
            audio, self.audio_self_query, self.audio_self_key,
            self.audio_self_value, self.audio_self_proj
        )
        audio_after_self = self.audio_norm1(audio + self.dropout(audio_self_attn))
        
        # Shared Low-Rank Alignment
        if self.use_auditory_type and auditory_type is not None:
            eeg_U = torch.zeros(B, self.low_rank_dim, device=eeg.device)
            for type_idx in range(len(self.type_aligners)):
                type_mask = (auditory_type == type_idx)
                if type_mask.any():
                    eeg_U[type_mask] = self.type_aligners[type_idx](eeg_after_self[type_mask])
        else:
            eeg_U = self.W_U_eeg(eeg_after_self)
            
        audio_U = self.W_U_audio(audio_after_self)
        
        interaction_H = eeg_U * audio_U
        interaction_H = self.shared_interaction(interaction_H)
        
        eeg_feedback = self.W_D_eeg(interaction_H)
        audio_feedback = self.W_D_audio(interaction_H)
        
        # Feed Forward
        eeg_ffn_input = eeg_after_self + self.dropout(eeg_feedback)
        eeg_ffn_out = self.eeg_ffn(eeg_ffn_input)
        eeg_final = self.eeg_norm2(eeg_ffn_input + self.dropout(eeg_ffn_out))
        
        audio_ffn_input = audio_after_self + self.dropout(audio_feedback) * 0.3
        audio_ffn_out = self.audio_ffn(audio_ffn_input)
        audio_final = self.audio_norm2(audio_ffn_input + self.dropout(audio_ffn_out))
        
        audio_final = audio_final * 0.3 + audio * 0.7
        
        aligned_eeg = self.eeg_final_proj(eeg_final) + eeg
        aligned_audio = self.audio_final_proj(audio_final) + audio * 0.7
        
        return aligned_eeg, aligned_audio


class EEGOnlyModel(nn.Module):
    """策略1: 纯EEG分类模型"""
    def __init__(self, eeg_encoder, num_classes=2):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        
        self.eeg_proj = nn.Sequential(
            nn.Linear(200, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, eeg):
        eeg_emb = self.eeg_encoder(eeg)
        eeg_features = self.eeg_proj(eeg_emb)
        logits = self.classifier(eeg_features)
        return logits
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重"""
        print(f"Loading pretrained EEG encoder weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的checkpoint结构
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("使用 'model_state_dict' 键")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("使用 'model' 键")
        else:
            state_dict = checkpoint
            print("直接使用checkpoint作为状态字典")
        
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        # 打印checkpoint中的键名以便调试
        print(f"Checkpoint中的键名 (前20个): {list(state_dict.keys())[:20]}")
        print(f"模型期望的键名 (前20个): {list(model_dict.keys())[:20]}")
        
        # 统计checkpoint中EEG相关的键名
        eeg_keys_in_checkpoint = [k for k in state_dict.keys() if 'eeg' in k.lower()]
        print(f"Checkpoint中EEG相关的键名数量: {len(eeg_keys_in_checkpoint)}")
        if eeg_keys_in_checkpoint:
            print(f"EEG相关键名示例: {eeg_keys_in_checkpoint[:5]}")
        
        # 首先尝试直接加载单独的EEG模型权重（适用于eeg_only设置）
        print("尝试加载单独的EEG模型权重...")
        for k, v in state_dict.items():
            # 直接匹配EEG编码器权重
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                print(f"Loaded: {k}")
        
        # 尝试键名映射（无论直接匹配是否成功）
        print("尝试键名映射...")
        for k, v in state_dict.items():
            # 跳过已经直接匹配的权重
            if k in pretrained_dict:
                continue
                
            # 处理不同的键名映射情况
            mapped_key = k
            
            # 情况1: student. -> eeg_encoder.model.
            if k.startswith('student.'):
                mapped_key = k.replace('student.', 'eeg_encoder.model.')
            
            # 情况2: 处理MindMix中的EEG编码器权重 (优先级更高)
            elif k.startswith('eeg_model.model.'):
                mapped_key = k.replace('eeg_model.model.', 'eeg_encoder.model.')
                print(f"Mapping: {k} -> {mapped_key}")
            
            # 情况3: 从融合模型中加载EEG编码器权重
            elif k.startswith('eeg_model.'):
                mapped_key = k.replace('eeg_model.', 'eeg_encoder.')
            
            # 情况4: 直接匹配eeg_encoder和eeg_proj
            elif k.startswith('eeg_encoder.') or k.startswith('eeg_proj.'):
                mapped_key = k
            
            # 检查映射后的键名是否存在且形状匹配
            if mapped_key in model_dict and model_dict[mapped_key].shape == v.shape:
                pretrained_dict[mapped_key] = v
                print(f"Loaded: {k} -> {mapped_key}")
            else:
                # 调试信息：显示为什么没有加载
                if mapped_key not in model_dict:
                    print(f"Debug: {k} -> {mapped_key} (键名不存在)")
                elif model_dict[mapped_key].shape != v.shape:
                    print(f"Debug: {k} -> {mapped_key} (形状不匹配: {v.shape} vs {model_dict[mapped_key].shape})")
        
        # 尝试更宽松的匹配
        print("尝试更宽松的匹配...")
        for k, v in state_dict.items():
            # 跳过已经加载的权重
            if k in pretrained_dict or any(k.startswith(loaded_key) for loaded_key in pretrained_dict.keys()):
                continue
                
            # 查找包含EEG相关关键词的权重
            if any(keyword in k.lower() for keyword in ['eeg', 'student', 'model']):
                # 尝试不同的映射策略
                possible_mappings = []
                
                # 策略1: 直接替换前缀
                if k.startswith('eeg_model.'):
                    possible_mappings.append(k.replace('eeg_model.', 'eeg_encoder.'))
                
                # 策略2: 处理student前缀
                if k.startswith('student.'):
                    possible_mappings.append(k.replace('student.', 'eeg_encoder.model.'))
                
                # 策略3: 处理嵌套的model结构
                if 'model.' in k:
                    possible_mappings.append(k.replace('model.', 'eeg_encoder.model.'))
                
                # 策略4: 处理eeg_model.model. -> eeg_encoder.model.
                if k.startswith('eeg_model.model.'):
                    possible_mappings.append(k.replace('eeg_model.model.', 'eeg_encoder.model.'))
                
                # 尝试所有可能的映射
                for mapped_key in possible_mappings:
                    if mapped_key in model_dict and model_dict[mapped_key].shape == v.shape:
                        pretrained_dict[mapped_key] = v
                        print(f"Loaded: {k} -> {mapped_key}")
                        break
        
        if pretrained_dict:
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} layers")
            
            # 打印加载的权重类型统计
            eeg_encoder_count = sum(1 for k in pretrained_dict.keys() if k.startswith('eeg_encoder.'))
            eeg_proj_count = sum(1 for k in pretrained_dict.keys() if k.startswith('eeg_proj.'))
            classifier_count = sum(1 for k in pretrained_dict.keys() if k.startswith('classifier.'))
            
            print(f"  - EEG Encoder layers: {eeg_encoder_count}")
            print(f"  - EEG Projection layers: {eeg_proj_count}")
            print(f"  - Classifier layers: {classifier_count}")
            
            if eeg_encoder_count == 0:
                print("Warning: No EEG encoder weights were loaded!")
                print("This might indicate a key mapping issue.")
        else:
            print("Warning: No compatible weights found in the checkpoint")
            print(f"Available keys in checkpoint: {list(state_dict.keys())[:10]}...")
            print(f"Expected keys in model: {list(model_dict.keys())[:10]}...")


class MultimodalRealModel(nn.Module):
    """策略2: 使用真实EEG-Audio数据的多模态模型"""
    def __init__(self, eeg_encoder, audio_encoder, fusion_method='clara', num_classes=2, use_auditory_type=False):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.audio_encoder = audio_encoder
        self.fusion_method = fusion_method
        self.use_auditory_type = use_auditory_type
        
        # 特征投影层
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
        
        # 融合模块
        if fusion_method == 'clara':
            self.fusion_module = CLARA(embed_dim=256, use_auditory_type=use_auditory_type)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 对比学习损失
        self.clip_loss = ClipLoss()
    
    def forward(self, eeg, audio, labels=None, mode='train'):
        # 特征提取
        eeg_emb = self.eeg_encoder(eeg)
        eeg_features = self.eeg_proj(eeg_emb)
        
        # 处理音频数据
        if len(audio.shape) == 2:  # [B, T]
            # 原始音频信号，需要通过audio encoder
            audio_emb = self.audio_encoder(audio).last_hidden_state.mean(1)
        else:  # [B, D]
            # 已经是特征向量
            audio_emb = audio
        
        audio_features = self.audio_proj(audio_emb)
        
        # 跨模态融合
        if self.fusion_method == 'clara':
            fused_eeg, aligned_audio = self.fusion_module(eeg_features, audio_features)
        else:
            fused_eeg = eeg_features
            aligned_audio = audio_features
        
        # 对于CLIP训练，直接返回特征
        if mode == 'clip_train':
            return fused_eeg, aligned_audio
        
        # 分类
        emotion_logits = self.classifier(fused_eeg)
        
        if mode == 'train':
            # 训练时返回对比学习损失
            clip_loss = self.clip_loss(fused_eeg, aligned_audio)
            return emotion_logits, clip_loss
        else:
            return emotion_logits
    
    def forward_contrastive(self, eeg, target_audio, negative_audio):
        """对比学习前向传播（与MindMix_clip_finetune.py的CLIPModel.forward完全一致）"""
        # EEG特征提取
        eeg_emb = self.eeg_encoder(eeg)
        eeg_emb = self.eeg_proj(eeg_emb)  # [B,256]
        
        # 音频特征提取
        target_audio_emb = self.audio_encoder(target_audio).last_hidden_state.mean(1)    
        target_audio_emb = self.audio_proj(target_audio_emb)  # [B,256]
        
        negative_audio_emb = self.audio_encoder(negative_audio).last_hidden_state.mean(1)
        negative_audio_emb = self.audio_proj(negative_audio_emb)
        
        # 根据融合方法进行跨模态交互
        if self.fusion_method == 'clara':
            # CLARA (真正的Shared Low-Rank Alignment)
            fused_eeg, aligned_audio = self.fusion_module(eeg_emb, target_audio_emb)
            _, aligned_negative_audio = self.fusion_module(eeg_emb, negative_audio_emb)
            final_audio_emb = aligned_audio  # 使用CLARA对齐后的音频特征
            negative_audio_emb = aligned_negative_audio
        else:
            # 其他融合方法
            fused_eeg, aligned_audio = self.fusion_module(eeg_emb, target_audio_emb)
            final_audio_emb = aligned_audio
            # 负样本保持原始特征以保持对比学习稳定性
        
        return fused_eeg, final_audio_emb, negative_audio_emb
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重（与MindMix_clip_finetune.py的CLIPModel.load_pretrained_weights完全一致，但添加键名映射）"""
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重，允许部分匹配，并处理键名映射
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 处理键名映射：预训练模型 -> 通用框架
            mapped_key = k
            if k.startswith('eeg_model.'):
                mapped_key = k.replace('eeg_model.', 'eeg_encoder.')
            elif k.startswith('audio_model.'):
                mapped_key = k.replace('audio_model.', 'audio_encoder.')
            
            # 检查映射后的键名是否存在且形状匹配
            if mapped_key in model_dict and model_dict[mapped_key].shape == v.shape:
                pretrained_dict[mapped_key] = v
                print(f"Loaded: {k} -> {mapped_key}")
            elif k in model_dict and model_dict[k].shape == v.shape:
                # 直接匹配（用于fusion_module, eeg_proj, audio_proj等）
                pretrained_dict[k] = v
                print(f"Loaded: {k}")
            else:
                print(f"Skipped: {k} (shape mismatch or not found)")
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Successfully loaded {len(pretrained_dict)} layers from pretrained model")


class MultimodalPrototypeModel(nn.Module):
    """策略3: 使用伪音频原型的多模态模型"""
    def __init__(self, eeg_encoder, audio_encoder, fusion_method='clara', num_classes=2, use_auditory_type=False):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.audio_encoder = audio_encoder
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        self.use_auditory_type = use_auditory_type
        
        # 特征投影层
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
        
        # 可学习的情绪音频原型
        self.emotion_audio_prototypes = nn.Parameter(
            torch.randn(num_classes, 768) * 0.1
        )
        
        # 融合模块
        if fusion_method == 'clara':
            self.fusion_module = CLARA(embed_dim=256, use_auditory_type=use_auditory_type)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 辅助任务：音频-标签对齐
        self.audio_label_projector = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
    
    def get_emotion_audio(self, labels=None, batch_size=None):
        """获取情绪相关的音频原型"""
        if labels is not None:
            return self.emotion_audio_prototypes[labels]
        else:
            device = self.emotion_audio_prototypes.device
            random_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
            return self.emotion_audio_prototypes[random_labels]
    
    def forward(self, eeg, labels=None, mode='train'):
        batch_size = eeg.size(0)
        
        # EEG特征提取
        eeg_emb = self.eeg_encoder(eeg)
        eeg_features = self.eeg_proj(eeg_emb)
        
        # 获取伪音频特征
        if mode == 'train' and labels is not None:
            pseudo_audio_raw = self.get_emotion_audio(labels=labels)
        else:
            pseudo_audio_raw = self.get_emotion_audio(batch_size=batch_size)
        
        pseudo_audio_features = self.audio_proj(pseudo_audio_raw)
        
        # 跨模态融合
        if self.fusion_method == 'clara':
            fused_eeg, aligned_audio = self.fusion_module(eeg_features, pseudo_audio_features)
        
        # 主任务：情绪分类
        emotion_logits = self.classifier(fused_eeg)
        
        # 辅助任务：音频-标签对齐
        audio_label_logits = self.audio_label_projector(aligned_audio)
        
        return emotion_logits, audio_label_logits
    
    def compute_loss(self, emotion_logits, audio_label_logits, true_labels, alpha=0.7, beta=0.3):
        """计算多任务损失"""
        emotion_loss = F.cross_entropy(emotion_logits, true_labels)
        audio_loss = F.cross_entropy(audio_label_logits, true_labels)
        total_loss = alpha * emotion_loss + beta * audio_loss
        return total_loss, emotion_loss, audio_loss
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练CLIP权重"""
        print(f"Loading pretrained CLIP weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 只加载EEG和audio相关的权重，不加载分类头
            if any(prefix in k for prefix in ['eeg_encoder.', 'audio_encoder.', 'eeg_proj.', 'audio_proj.', 'fusion_module.']):
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    print(f"Loaded: {k}")
        
        if pretrained_dict:
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} layers")


def create_model(strategy, eeg_encoder, audio_encoder=None, fusion_method='clara', num_classes=2, use_auditory_type=False):
    """工厂函数：根据策略创建相应的模型"""
    if strategy == 'eeg_only':
        return EEGOnlyModel(eeg_encoder, num_classes)
    elif strategy == 'multimodal_real':
        if audio_encoder is None:
            raise ValueError("Audio encoder is required for multimodal_real strategy")
        return MultimodalRealModel(eeg_encoder, audio_encoder, fusion_method, num_classes, use_auditory_type)
    elif strategy == 'multimodal_prototype':
        if audio_encoder is None:
            raise ValueError("Audio encoder is required for multimodal_prototype strategy")
        return MultimodalPrototypeModel(eeg_encoder, audio_encoder, fusion_method, num_classes, use_auditory_type)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}") 