"""特征提取钩子工具类"""
import torch.nn as nn
from typing import Dict, List, Optional

class FeatureHook:
    """用于自动查找模型关键层并注册Forward Hook提取特征"""
    
    LAYER_KEYWORDS = ["layer4", "layer3", "features", "backbone", "block"]
    
    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        self.features: Dict[str, any] = {}
        self.hooks = []
        layers = layer_names or self._auto_find_layers(model)
        self._register(model, layers)
    
    def _auto_find_layers(self, model: nn.Module) -> List[str]:
        found = {}
        for name, _ in model.named_modules():
            for kw in self.LAYER_KEYWORDS:
                if kw in name.lower():
                    found[kw] = name
        return list(found.values())[-1:] if found else []
    
    def _register(self, model: nn.Module, layer_names: List[str]):
        name_to_module = dict(model.named_modules())
        for name in layer_names:
            if name in name_to_module:
                hook = name_to_module[name].register_forward_hook(
                    lambda m, i, o, n=name: self.features.update({n: o})
                )
                self.hooks.append(hook)
    
    def get(self) -> Dict[str, any]:
        return self.features
    
    def clear(self):
        self.features.clear()
    
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

