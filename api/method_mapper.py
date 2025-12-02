"""方法映射器 - 扁平化命名转内部strategy"""
import logging
from typing import Dict, Any, Optional, List, Union

from services.files import ExtraFilesManager

logger = logging.getLogger(__name__)

# 操作类型前缀
_OP_PREFIXES = ("quantize_", "prune_", "distill_")


class MethodMapper:
    """扁平化method转内部strategy"""
    
    def convert_to_strategy(
        self,
        method: Union[str, List[str]],
        extra_manager: ExtraFilesManager,
        method_params: Optional[Dict[str, Dict[str, Any]]] = None,
        export_formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        转换method为strategy
        
        Args:
            method: "quantize_fp16" 或 ["quantize_auto", "prune_structured"]
            extra_manager: 额外文件管理器
            method_params: 参数覆盖，如 {"prune_structured": {"target_sparsity": 0.5}}
            export_formats: 导出格式列表
        """
        methods = [method] if isinstance(method, str) else method
        params = method_params or {}
        
        strategy = {}
        for m in methods:
            op, sub = self._parse_method(m)
            if op == "quantize":
                strategy["quantize"] = self._build_quantize(sub, extra_manager, params.get(m, {}))
            elif op == "prune":
                strategy["prune"] = self._build_prune(sub, extra_manager, params.get(m, {}))
            elif op == "distill":
                strategy["distill"] = self._build_distill(sub, extra_manager, params.get(m, {}))
        
        # 过滤未启用项
        strategy = {k: v for k, v in strategy.items() if v.get("enable")}
        
        if export_formats:
            strategy["export"] = {"formats": export_formats}
        
        return strategy
    
    def _parse_method(self, method: str) -> tuple:
        """解析方法名，返回(操作类型, 子方法)"""
        for prefix in _OP_PREFIXES:
            if method.startswith(prefix):
                return prefix[:-1], method[len(prefix):]
        raise ValueError(f"Invalid method format: {method}")
    
    def _build_quantize(self, sub: str, extra: ExtraFilesManager, overrides: Dict) -> Dict[str, Any]:
        """构建量化配置"""
        cfg = {"enable": True}
        
        if sub == "auto":
            cfg["auto"] = True
        elif sub in ("fp16", "int8_dynamic", "qat"):
            cfg["precision"] = sub
            if sub == "qat":
                train_dir = extra.get_train_data_dir()
                if not train_dir:
                    raise ValueError("qat requires train_data")
                cfg["train_data_dir"] = train_dir
                cfg["epochs"] = overrides.get("epochs", 10)
        elif sub in ("int8", "int8_static"):
            calib = extra.get_calib_dir()
            if calib:
                cfg["precision"] = "int8_static"
                cfg["calib_dir"] = calib
            else:
                if sub == "int8_static":
                    logger.warning("int8_static fallback to int8_dynamic (no calibration_data)")
                cfg["precision"] = "int8_dynamic"
        else:
            raise ValueError(f"Unknown quantize method: {sub}")
        
        return cfg
    
    def _build_prune(self, sub: str, extra: ExtraFilesManager, overrides: Dict) -> Dict[str, Any]:
        """构建剪枝配置"""
        cfg = {"enable": True}
        
        if sub == "auto":
            cfg["type"] = "auto"
            cfg["auto"] = True
        elif sub in ("structured", "unstructured"):
            cfg["type"] = sub
            cfg["target_sparsity"] = overrides.get("target_sparsity", 0.3)
        else:
            raise ValueError(f"Unknown prune method: {sub}")
        
        val_dir = extra.get_val_data_dir()
        if val_dir:
            cfg["val_data_dir"] = val_dir
        
        return cfg
    
    def _build_distill(self, sub: str, extra: ExtraFilesManager, overrides: Dict) -> Dict[str, Any]:
        """构建蒸馏配置"""
        teacher = extra.get_teacher_model_dir()
        train = extra.get_train_data_dir()
        
        if not teacher:
            raise ValueError("distill requires teacher_model")
        if not train:
            raise ValueError("distill requires train_data")
        
        return {
            "enable": True,
            "teacher_dir": teacher,
            "train_data_dir": train,
            "temperature": overrides.get("temperature", 4.0),
            "alpha": overrides.get("alpha", 0.7),
            "epochs": overrides.get("epochs", 20)
        }
