"""蒸馏损失函数库"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class DistillLoss:
    """统一蒸馏损失计算器"""
    
    def __init__(self):
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def __call__(self, s_out, t_out, s_feats: Dict, t_feats: Dict, 
                 labels: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=labels.device)
        
        # Logits蒸馏 (KL散度)
        if cfg.get("use_logits"):
            T = cfg["temperature"]
            s_logits = s_out[0] if isinstance(s_out, (tuple, list)) else s_out
            t_logits = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
            
            kd_loss = self.kl(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1)
            ) * (T ** 2)
            hard_loss = self.ce(s_logits, labels)
            
            alpha = cfg.get("alpha_logits", 0.5)
            loss += alpha * kd_loss + cfg.get("alpha_hard", 0.3) * hard_loss
        
        # Feature蒸馏 (MSE)
        if cfg.get("use_feature") and s_feats and t_feats:
            feat_loss = 0.0
            count = 0
            for name, s_f in s_feats.items():
                if name in t_feats and s_f.shape == t_feats[name].shape:
                    feat_loss += self.mse(s_f, t_feats[name])
                    count += 1
            if count > 0:
                loss += cfg.get("alpha_feature", 0.0) * (feat_loss / count)
        
        # MSE蒸馏 (生成模型)
        if cfg.get("use_mse"):
            s_rec = s_out[0] if isinstance(s_out, tuple) else s_out
            t_rec = t_out[0] if isinstance(t_out, tuple) else t_out
            loss += cfg.get("alpha_mse", 1.0) * self.mse(s_rec, t_rec)
        
        return loss

