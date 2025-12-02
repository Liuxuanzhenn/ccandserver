"""蒸馏策略选择器 - 根据模型类型自动选择最优蒸馏配置"""
from typing import Any, Dict, Optional

try:
    from .core import run_distillation
except ImportError:
    from strategies.distill.core import run_distillation


def decide_and_apply_distill(
    student: Any, teacher: Any, cfg: Dict[str, Any], family: Optional[str] = None
) -> Dict[str, Any]:
    """蒸馏策略选择和应用"""
    if not isinstance(cfg, dict):
        return {"status": "skipped", "reason": "invalid config"}
    
    family = str(family or "generic").lower()
    alpha = cfg.get("alpha", 0.7)
    
    # 基础配置
    config = {
        "epochs": cfg.get("epochs", 10),
        "batch_size": cfg.get("batch_size", 32),
        "lr": cfg.get("lr", 1e-3),
        "temperature": cfg.get("temperature", 4.0),
        "train_data_dir": cfg.get("train_data_dir"),
        "val_data_dir": cfg.get("val_data_dir"),
        "artifacts_dir": cfg.get("artifacts_dir"),
        "use_logits": False,
        "use_feature": False,
        "use_mse": False,
        "alpha_logits": 0.0,
        "alpha_feature": 0.0,
        "alpha_mse": 0.0,
        "alpha_hard": 1.0 - alpha
    }
    
    # 策略分发
    if family in ["resnet", "vgg", "vit", "cnn", "van", "inceptionv4", "yolo"]:
        # 分类/检测模型: Logits + Feature 混合蒸馏
        config["use_logits"] = True
        config["use_feature"] = True
        config["alpha_logits"] = alpha * 0.6
        config["alpha_feature"] = alpha * 0.4
        if family == "yolo":
            config["batch_size"] = 16
    
    elif family == "vae":
        # 生成模型: MSE蒸馏
        config["use_mse"] = True
        config["alpha_mse"] = alpha
        config["alpha_hard"] = 0.0
    
    elif family in ["lstm", "rnn", "gcn", "transformer"]:
        # 序列/图模型: Feature蒸馏
        config["use_feature"] = True
        config["alpha_feature"] = alpha
        config["alpha_hard"] = 0.0
    
    else:
        # 通用: Logits蒸馏
        config["use_logits"] = True
        config["alpha_logits"] = alpha
    
    return run_distillation(student, teacher, config)
