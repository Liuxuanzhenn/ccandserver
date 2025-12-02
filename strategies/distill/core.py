"""统一蒸馏引擎（支持Logits/Feature/MSE及AMP加速）"""
import os
import torch
from typing import Any, Dict

try:
    from ..common import write_report, evaluate_accuracy
except ImportError:
    from strategies.common import write_report, evaluate_accuracy

try:
    from ...utils.hooks import FeatureHook
except ImportError:
    from utils.hooks import FeatureHook

from .losses import DistillLoss


def run_distillation(student: Any, teacher: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """执行统一蒸馏流程（支持AMP混合精度加速）"""
    artifacts_dir = cfg.get("artifacts_dir") 
    s_hook, t_hook = None, None
    
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        from torch.cuda.amp import autocast, GradScaler
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda"
        
        student.to(device)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        
        # 钩子管理（Feature蒸馏时启用）
        if cfg.get("use_feature"):
            s_hook = FeatureHook(student)
            t_hook = FeatureHook(teacher)
        
        # 数据准备
        train_dir = cfg.get("train_data_dir")
        if not train_dir or not os.path.exists(train_dir):
            return {"status": "skipped", "reason": "no train_data_dir"}
        
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        loader = DataLoader(
            datasets.ImageFolder(train_dir, transform),
            batch_size=cfg.get("batch_size", 32),
            shuffle=True,
            num_workers=4 if use_amp else 0,
            pin_memory=use_amp
        )
        
        # 训练组件
        optimizer = torch.optim.Adam(student.parameters(), lr=cfg.get("lr", 1e-3))
        scaler = GradScaler() if use_amp else None
        loss_fn = DistillLoss()
        
        # 训练循环
        epochs = cfg.get("epochs", 10)
        losses = []
        
        for _ in range(epochs):
            student.train()
            epoch_loss = 0.0
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.no_grad():
                    t_out = teacher(images)
                    t_feats = t_hook.get() if t_hook else {}
                
                with autocast(enabled=use_amp):
                    s_out = student(images)
                    s_feats = s_hook.get() if s_hook else {}
                    loss = loss_fn(s_out, t_out, s_feats, t_feats, labels, cfg)
                
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                if s_hook: s_hook.clear()
                if t_hook: t_hook.clear()
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / max(1, len(loader)))
        
        # 清理
        if s_hook: s_hook.remove()
        if t_hook: t_hook.remove()
        
        rep = {"status": "ok", "mode": "hybrid", "epochs": epochs, "final_loss": losses[-1] if losses else 0}
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep
        
    except Exception as e:
        if s_hook: s_hook.remove()
        if t_hook: t_hook.remove()
        rep = {"status": "error", "reason": str(e)}
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep

