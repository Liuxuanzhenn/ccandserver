"""模型能力配置

加载和查询 model_capabilities.json 配置
"""
import json
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    from ..config.settings import Config
except ImportError:
    try:
        from config.settings import Config
    except ImportError:
        Config = None


class CapabilityRegistryV2:
    """模型能力注册表 V2"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            if Config:
                config_path = str(Config.MODEL_CAPABILITIES)
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                config_path = os.path.join(base_dir, "configs", "model_capabilities.json")
        
        self.config_path = config_path
        self._capabilities = self._load_capabilities()
    
    def _load_capabilities(self) -> Dict[str, Any]:
        """加载能力配置文件（新格式：顶层直接是模型键）"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 新格式：顶层直接是模型键（如 "pytorch.resnet"）
            # 过滤掉非模型键（如 "code", "message", "data" 等）
            capabilities = {}
            for key, value in data.items():
                if isinstance(value, dict) and "framework" in value and "family" in value:
                    capabilities[key] = value
            
            return capabilities
        except Exception as e:
            logger.error(f"Failed to load capabilities from {self.config_path}: {e}")
            return {}
    
    def get(self, framework: str, family: str) -> Optional[Dict[str, Any]]:
        """获取指定模型的能力信息"""
        key = f"{framework.lower()}.{family.lower()}"
        return self._capabilities.get(key)
    
    def get_file_types_mapping(self) -> Dict[str, str]:
        """获取文件类型映射"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("file_types", {})
        except Exception as e:
            logger.error(f"Failed to load file_types from {self.config_path}: {e}")
            return {}
    
    def get_supported_operations(self, framework: str, family: str) -> Dict[str, Any]:
        """获取模型支持的操作"""
        cap = self.get(framework, family)
        if not cap:
            return {}
        
        methods = cap.get("methods", {})
        operations = {}
        
        for op_type in ["quantize", "prune", "distill"]:
            op_config = methods.get(op_type)
            if isinstance(op_config, dict) and "available" in op_config:
                operations[op_type] = {
                    "enabled": True,
                    "methods": op_config.get("available", []),
                    "recommended": op_config.get("recommended")
                }
        
        return operations
    
    def get_all_operation_requirements(self, framework: str, family: str) -> Dict[str, Any]:
        """获取所有操作的需求（需要的额外文件等）"""
        cap = self.get(framework, family)
        if not cap:
            return {}

        methods = cap.get("methods", {})
        requirements = {}

        for op_type in ["quantize", "prune", "distill"]:
            op_config = methods.get(op_type)
            if isinstance(op_config, dict) and "requirements" in op_config:
                req_dict = op_config.get("requirements", {})
                op_requirements = {}

                for method_name, method_req in req_dict.items():
                    op_requirements[method_name] = {
                        "required_extra_files": method_req.get("required_files", []),
                        "optional_extra_files": method_req.get("optional_files", [])
                    }

                if op_requirements:
                    requirements[op_type] = op_requirements

        return requirements

    def get_simplified_methods(self, framework: str, family: str) -> Dict[str, Any]:
        """获取简化版的方法信息（用于前端展示）

        返回格式：
        {
            "available": ["quantize_auto", "quantize_fp16", "quantize_int8", "quantize_qat", 
                         "prune_auto", "prune_structured_pruning", "prune_unstructured_pruning",
                         "distill_auto"]
        }
        """
        cap = self.get(framework, family)
        if not cap:
            return {"available": []}

        methods = cap.get("methods", {})
        available_methods = []

        for op_type in ["quantize", "prune", "distill"]:
            op_config = methods.get(op_type)
            if isinstance(op_config, dict) and "available" in op_config:
                method_list = op_config.get("available", [])
                # 将方法名格式化为 "操作类型_方法名"
                for method_name in method_list:
                    alias = method_name
                    if op_type == "prune" and alias.endswith("_pruning"):
                        alias = alias[:-len("_pruning")]
                    available_methods.append(f"{op_type}_{alias}")

        return {"available": available_methods}


# 全局单例
_registry_instance: Optional[CapabilityRegistryV2] = None


def get_registry_v2() -> CapabilityRegistryV2:
    """获取能力注册表单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = CapabilityRegistryV2()
    return _registry_instance

