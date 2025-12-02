"""API Schema定义

使用Pydantic定义请求和响应格式
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class DetectCapabilitiesRequest(BaseModel):
    """接口1请求格式"""
    model_dir: str = Field(..., description="模型目录路径（系统提供的路径）")
    
    class Config:
        schema_extra = {
            "example": {
                "model_dir": "/nfs/dubhe-prod/train-manage/1/job-1099-hjs6q/model"
            }
        }


class OperationRequirement(BaseModel):
    """操作需求"""
    required_files: List[str] = Field(default_factory=list, description="必需的文件类型列表")
    optional_files: List[str] = Field(default_factory=list, description="可选的文件类型列表")
    configurable: Dict[str, Any] = Field(default_factory=dict, description="可配置参数")


class DetectCapabilitiesResponse(BaseModel):
    """接口1响应格式"""
    code: int = Field(..., description="状态码")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")
    
    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "framework": "pytorch",
                    "family": "yolo",
                    "supported_operations": [
                        "quantize_fp16",
                        "quantize_int8_dynamic",
                        "prune_structured"
                    ],
                    "methods": {
                        "available": [
                            "quantize_fp16",
                            "quantize_int8_dynamic",
                            "prune_structured"
                        ]
                    }
                }
            }
        }


class ExecuteCompressionRequest(BaseModel):
    """接口2请求格式"""
    model_dir: str = Field(..., description="模型目录路径")
    result_dir: str = Field(..., description="结果目录路径")
    extra_dir: Optional[str] = Field(None, description="额外文件目录路径（可选）")
    method: Union[str, List[str]] = Field(..., description="压缩方法（quantize_fp16 / [quantize_auto, prune_structured]）")
    method_params: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, 
        description="参数覆盖，如 {\"prune_structured\": {\"target_sparsity\": 0.5}}"
    )
    export_formats: Optional[List[str]] = Field(None, description="导出格式列表（可选）")
    
    @validator("method")
    def validate_method(cls, v):
        """验证method参数（扁平字符串模式）"""
        valid_methods = {
            "quantize_fp16",
            "quantize_int8_dynamic",
            "quantize_int8_static",
            "quantize_int8",
            "quantize_qat",
            "quantize_auto",
            "prune_structured",
            "prune_unstructured",
            "prune_auto",
            "distill_auto"
        }
        to_check = [v] if isinstance(v, str) else list(v)
        if not to_check:
            raise ValueError("method cannot be empty")
        for item in to_check:
            if item not in valid_methods:
                raise ValueError(f"Invalid method: {item}. Must be one of {sorted(valid_methods)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_dir": "/nfs/.../model",
                "result_dir": "/nfs/.../result",
                "extra_dir": "/nfs/.../extra",
                "method": ["quantize_auto", "prune_structured", "distill_auto"],
                "method_params": {
                    "prune_structured": {"target_sparsity": 0.5},
                    "distill_auto": {"temperature": 5.0, "alpha": 0.8, "epochs": 25}
                },
                "export_formats": ["pt", "onnx"]
            }
        }


class ExecuteCompressionResponse(BaseModel):
    """接口2响应格式"""
    code: int = Field(..., description="状态码")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")
    
    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "job_id": "j_xxx",
                    "result_dir": "/nfs/.../result",
                    "operations": [
                        {"operation": "quantize", "status": "success"}
                    ],
                    "outputs": [
                        {"type": "quantized_model", "path": "model_fp16.pt"},
                        {"type": "onnx_model", "path": "model_fp16.onnx"}
                    ],
                    "metrics": {
                        "size_before_mb": 6.2,
                        "size_after_mb": 3.1,
                        "compression_ratio": 0.5
                    }
                }
            }
        }


class ErrorResponse(BaseModel):
    """错误响应格式"""
    code: int = Field(..., description="错误码")
    message: str = Field(..., description="错误消息")
    data: Optional[Any] = Field(None, description="错误详情")
    
    class Config:
        schema_extra = {
            "example": {
                "code": 400,
                "message": "Invalid request",
                "data": {
                    "field": "model_dir",
                    "error": "model_dir is required"
                }
            }
        }

