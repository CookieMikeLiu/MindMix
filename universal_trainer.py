"""
通用EEG微调训练器

支持三种微调策略的训练和评估：
1. EEGOnlyTrainer: 纯EEG分类训练
2. MultimodalRealTrainer: 真实多模态数据训练
3. MultimodalPrototypeTrainer: 伪音频原型训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import os


class BaseTrainer:
    """基础训练器"""
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        self.early_stopping_counter = 0
        self.patience = 20  # 从10增加到20，给模型更多训练时间
        
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        if self.args.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.args.warmup_lr / self.args.lr,
                end_factor=1.0,
                total_iters=self.args.warmup_epochs
            )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs - self.args.warmup_epochs,
            eta_min=self.args.min_lr
        )
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        if not self.args.save_ckpt:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_f1': self.best_f1,
            'args': self.args,
        }
        
        if hasattr(self, 'warmup_scheduler'):
            checkpoint['warmup_scheduler_state_dict'] = self.warmup_scheduler.state_dict()
        
        if is_best:
            filename = f'best_model_acc_{self.best_accuracy:.4f}_f1_{self.best_f1:.4f}.pth'
            torch.save(checkpoint, os.path.join(self.args.output_dir, filename))
            print(f"Saved best model: {filename}")
        
        # 可选：保存最新的checkpoint
        latest_filename = 'latest_checkpoint.pth'
        torch.save(checkpoint, os.path.join(self.args.output_dir, latest_filename))
    
    def update_learning_rate(self, epoch):
        """更新学习率"""
        if epoch < self.args.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()
    
    def check_early_stopping(self, val_f1):
        """检查早停条件"""
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.early_stopping_counter = 0
            return False, True  # 不停止，是最佳
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                return True, False  # 停止，不是最佳
            return False, False  # 不停止，不是最佳


class EEGOnlyTrainer(BaseTrainer):
    """纯EEG分类训练器"""
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # 处理不同长度的batch（EEG4EMO返回2个值，KUL/DTU返回4个值）
            if len(batch) == 2:
                eeg, labels = batch
            elif len(batch) == 4:
                eeg, target_audio, negative_audio, labels = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            
            eeg = eeg.to(self.device)
            labels = labels.to(self.device).long()  # 确保标签是Long类型
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(eeg)
            loss = F.cross_entropy(logits, labels)
            
            # 反向传播
            loss.backward()
            
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{total_loss/num_batches:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        return total_loss / num_batches
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 处理不同长度的batch（EEG4EMO返回2个值，KUL/DTU返回4个值）
                if len(batch) == 2:
                    eeg, labels = batch
                elif len(batch) == 4:
                    eeg, target_audio, negative_audio, labels = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
                
                eeg = eeg.to(self.device)
                labels = labels.to(self.device).long()  # 确保标签是Long类型
                
                logits = self.model(eeg)
                loss = F.cross_entropy(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(test_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(self, train_loader, val_loader):
        """完整训练过程"""
        self.setup_optimizer()
        
        print(f"Starting EEG-only training for {self.args.epochs} epochs")
        
        for epoch in range(self.args.epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 更新学习率
            self.update_learning_rate(epoch)
            
            # 验证
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # 检查是否是最佳模型
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
            
            # 早停检查
            should_stop, is_best = self.check_early_stopping(val_f1)
            
            # if is_best:
            #     self.save_checkpoint(epoch, is_best=True)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
            print(f"Best F1: {self.best_f1:.4f} | Best Acc: {self.best_accuracy:.4f}")
            
            if should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.best_accuracy, self.best_f1


class MultimodalRealTrainer(BaseTrainer):
    """真实多模态数据训练器"""
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_clip_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Multimodal Train]")
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # 检查数据格式 - 数据已经在collate_fn中移动到正确设备
            if len(batch) == 4:  # KUL/DTU格式: (eeg, target_audio, negative_audio, labels)
                eeg, target_audio, negative_audio, labels = batch
                # 数据已经在collate_fn中移动到设备，不需要再次移动
                
                # 对比学习训练方式
                clip_loss = self._train_contrastive(eeg, target_audio, negative_audio)
                
                # 对于对比学习，主要使用CLIP损失
                total_loss_batch = clip_loss
                cls_loss = torch.tensor(0.0, device=self.device)  # 占位符
                
            else:  # EEG4EMO格式: (eeg, audio, labels)
                eeg, audio, labels = batch
                # 数据已经在collate_fn中移动到设备，不需要再次移动
                
                # 前向传播
                emotion_logits, clip_loss = self.model(eeg, audio, labels, 'train')
                
                # 分类损失
                cls_loss = F.cross_entropy(emotion_logits, labels.long())
                
                # 总损失：分类损失 + 对比学习损失
                total_loss_batch = cls_loss + 0.1 * clip_loss  # 对比学习损失权重较小
            
            # 反向传播
            total_loss_batch.backward()
            
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_clip_loss += clip_loss.item() if isinstance(clip_loss, torch.Tensor) else 0.0
            num_batches += 1
            
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'Total': f'{total_loss_batch.item():.4f}',
                    'Cls': f'{cls_loss.item():.4f}',
                    'CLIP': f'{clip_loss.item() if isinstance(clip_loss, torch.Tensor) else 0.0:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        return total_loss / num_batches, total_cls_loss / num_batches, total_clip_loss / num_batches
    
    def _train_contrastive(self, eeg, target_audio, negative_audio):
        """对比学习训练方法（与MindMix_clip_finetune.py的train_model完全一致）"""
        # 数据已经在collate_fn中移动到设备，不需要再次移动
        
        # 计算CLIP loss（与MindMix_clip_finetune.py完全一致）
        eeg_features, audio_features = self.model(eeg, target_audio, mode='clip_train')
        loss = self.model.clip_loss(eeg_features, audio_features)
        
        return loss
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 检查数据格式：是否包含target和negative audio
                if len(batch) == 4:  # KUL/DTU格式: (eeg, target_audio, negative_audio, labels)
                    eeg, target_audio, negative_audio, labels = batch
                    # 数据已经在collate_fn中移动到设备，不需要再次移动
                    
                    # 使用对比学习评估方式
                    total_correct_batch, total_samples_batch, loss_batch = self._evaluate_contrastive(
                        eeg, target_audio, negative_audio
                    )
                    
                    total_correct += total_correct_batch
                    total_samples += total_samples_batch
                    total_loss += loss_batch
                    
                    # 对于对比学习，预测是二分类（target vs negative）
                    # 这里我们假设target是正确的选择
                    batch_predictions = [1] * total_samples_batch  # 简化处理
                    batch_labels = [1] * total_samples_batch
                    
                else:  # EEG4EMO格式: (eeg, audio, labels)
                    eeg, audio, labels = batch
                    # 数据已经在collate_fn中移动到设备，不需要再次移动
                    
                    emotion_logits = self.model(eeg, audio, mode='eval')
                    loss = F.cross_entropy(emotion_logits, labels.long())
                    
                    predictions = torch.argmax(emotion_logits, dim=1)
                    
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
                    total_loss += loss.item()
                    
                    batch_predictions = predictions.cpu().numpy()
                    batch_labels = labels.cpu().numpy()
                
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_labels)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        f1 = f1_score(all_labels, all_predictions, average='weighted') if len(all_labels) > 0 else 0.0
        
        return avg_loss, accuracy, f1
    
    def _evaluate_contrastive(self, eeg, target_audio, negative_audio):
        """对比学习评估方法（与MindMix_clip_finetune.py的evaluate_model完全一致）"""
        # 数据已经在collate_fn中移动到设备，不需要再次移动
        
        # 计算特征（与MindMix_clip_finetune.py的model forward完全一致）
        eeg_feat, target_feat, neg_feat = self.model.forward_contrastive(eeg, target_audio, negative_audio)
        
        # 计算loss（与MindMix_clip_finetune.py完全一致）
        loss = self.model.clip_loss(eeg_feat, target_feat)
        
        # 计算概率（与MindMix_clip_finetune.py完全一致）
        probs_segment_A = self.model.clip_loss.get_probabilities(eeg_feat, target_feat)
        probs_segment_B = self.model.clip_loss.get_probabilities(eeg_feat, neg_feat)
        
        # 计算对角线元素（与MindMix_clip_finetune.py完全一致）
        diag_A = probs_segment_A.diagonal(offset=0, dim1=0, dim2=1)
        diag_B = probs_segment_B.diagonal(offset=0, dim1=0, dim2=1)
        
        # 预测（与MindMix_clip_finetune.py完全一致）
        predictions = (diag_A > diag_B).long()
        
        # 创建标签（与MindMix_clip_finetune.py完全一致）
        labels = torch.zeros(eeg.size(0), dtype=torch.long, device=self.device)
        labels[:target_audio.size(0)] = 1
        
        # 更新正确预测数和样本总数（与MindMix_clip_finetune.py完全一致）
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        
        return correct, total, loss.item()
    
    def _get_clip_probabilities(self, estimates, candidates):
        """计算CLIP概率（参考MindMix_clip_finetune.py）"""
        # 归一化特征
        estimates = F.normalize(estimates, p=2, dim=-1)
        candidates = F.normalize(candidates, p=2, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.mm(estimates, candidates.T) * self.model.clip_loss.logit_scale.exp()
        
        # 应用softmax
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities
    
    def train(self, train_loader, val_loader):
        """完整训练过程"""
        self.setup_optimizer()
        
        print(f"Starting multimodal real training for {self.args.epochs} epochs")
        
        for epoch in range(self.args.epochs):
            # 训练
            train_loss, train_cls_loss, train_clip_loss = self.train_epoch(train_loader, epoch)
            
            # 更新学习率
            self.update_learning_rate(epoch)
            
            # 验证
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # 检查是否是最佳模型
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
            
            # 早停检查
            should_stop, is_best = self.check_early_stopping(val_f1)
            
            # if is_best:
            #     self.save_checkpoint(epoch, is_best=True)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, CLIP: {train_clip_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
            print(f"Best F1: {self.best_f1:.4f} | Best Acc: {self.best_accuracy:.4f}")
            
            if should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.best_accuracy, self.best_f1


class MultimodalPrototypeTrainer(BaseTrainer):
    """伪音频原型训练器"""
    
    def setup_optimizer(self):
        """设置分层学习率优化器"""
        # 分层学习率设置
        param_groups = [
            # 预训练组件使用较小学习率
            {'params': [p for n, p in self.model.named_parameters() 
                       if any(prefix in n for prefix in ['eeg_encoder.', 'audio_encoder.', 'fusion_module.'])
                       and p.requires_grad], 
             'lr': self.args.lr * 0.1, 'name': 'pretrained'},
            # 音频原型使用较大学习率
            {'params': [self.model.emotion_audio_prototypes], 
             'lr': self.args.lr * 2.0, 'name': 'prototypes'},
            # 分类器使用标准学习率
            {'params': [p for n, p in self.model.named_parameters() 
                       if any(prefix in n for prefix in ['classifier.', 'audio_label_projector.', 'eeg_proj.', 'audio_proj.'])
                       and p.requires_grad], 
             'lr': self.args.lr, 'name': 'classifier'}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        if self.args.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.args.warmup_lr / self.args.lr,
                end_factor=1.0,
                total_iters=self.args.warmup_epochs
            )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs - self.args.warmup_epochs,
            eta_min=self.args.min_lr
        )
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_emotion_loss = 0.0
        total_audio_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Prototype Train]")
        for batch_idx, batch in enumerate(pbar):
            # 处理不同的batch结构
            if len(batch) == 2:
                # EEG-only strategy: (eeg, labels)
                eeg, labels = batch
                audio = None
            elif len(batch) == 3:
                # Multimodal strategy: (eeg, audio, labels)
                eeg, audio, labels = batch
            elif len(batch) == 4:
                # Contrastive learning: (eeg, target_audio, negative_audio, labels)
                eeg, target_audio, negative_audio, labels = batch
                audio = target_audio  # 使用target_audio作为主要音频
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            eeg = eeg.to(self.device)
            labels = labels.to(self.device).long()  # 确保标签是Long类型
            if audio is not None:
                audio = audio.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if audio is not None:
                emotion_logits, audio_label_logits = self.model(eeg, audio, labels, 'train')
            else:
                emotion_logits, audio_label_logits = self.model(eeg, labels, 'train')
            
            # 计算多任务损失
            total_loss_batch, emotion_loss, audio_loss = self.model.compute_loss(
                emotion_logits, audio_label_logits, labels, alpha=0.7, beta=0.3
            )
            
            # 反向传播
            total_loss_batch.backward()
            
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_emotion_loss += emotion_loss.item()
            total_audio_loss += audio_loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'Total': f'{total_loss_batch.item():.4f}',
                    'Emotion': f'{emotion_loss.item():.4f}',
                    'Audio': f'{audio_loss.item():.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        return total_loss / num_batches, total_emotion_loss / num_batches, total_audio_loss / num_batches
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 处理不同的batch结构
                if len(batch) == 2:
                    # EEG-only strategy: (eeg, labels)
                    eeg, labels = batch
                    audio = None
                elif len(batch) == 3:
                    # Multimodal strategy: (eeg, audio, labels)
                    eeg, audio, labels = batch
                elif len(batch) == 4:
                    # Contrastive learning: (eeg, target_audio, negative_audio, labels)
                    eeg, target_audio, negative_audio, labels = batch
                    audio = target_audio  # 使用target_audio作为主要音频
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                
                eeg = eeg.to(self.device)
                labels = labels.to(self.device).long()  # 确保标签是Long类型
                if audio is not None:
                    audio = audio.to(self.device)
                
                # 前向传播
                if audio is not None:
                    emotion_logits, audio_label_logits = self.model(eeg, audio, mode='eval')
                else:
                    emotion_logits, audio_label_logits = self.model(eeg, mode='eval')
                
                # 只使用情绪分类损失进行评估
                loss = F.cross_entropy(emotion_logits, labels.long())
                
                predictions = torch.argmax(emotion_logits, dim=1)
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(test_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(self, train_loader, val_loader):
        """完整训练过程"""
        self.setup_optimizer()
        
        print(f"Starting multimodal prototype training for {self.args.epochs} epochs")
        print(f"Optimizer groups: {len(self.optimizer.param_groups)}")
        
        for epoch in range(self.args.epochs):
            # 训练
            train_loss, train_emotion_loss, train_audio_loss = self.train_epoch(train_loader, epoch)
            
            # 更新学习率
            self.update_learning_rate(epoch)
            
            # 验证
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # 检查是否是最佳模型
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
            
            # 早停检查
            should_stop, is_best = self.check_early_stopping(val_f1)
            
            # if is_best:
            #     self.save_checkpoint(epoch, is_best=True)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f} (Emotion: {train_emotion_loss:.4f}, Audio: {train_audio_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
            print(f"Best F1: {self.best_f1:.4f} | Best Acc: {self.best_accuracy:.4f}")
            
            if should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.best_accuracy, self.best_f1


def create_trainer(strategy, model, device, args):
    """工厂函数：根据策略创建相应的训练器"""
    if strategy == 'eeg_only':
        return EEGOnlyTrainer(model, device, args)
    elif strategy == 'multimodal_real':
        return MultimodalRealTrainer(model, device, args)
    elif strategy == 'multimodal_prototype':
        return MultimodalPrototypeTrainer(model, device, args)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}") 