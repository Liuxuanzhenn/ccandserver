"""格式转换API接口"""
import logging
import os
import time
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify

from services.model import ModelDetector
from utils.path import PathManager
from utils.error import create_error_response, create_success_response, APIError, ErrorCode
from adapters.registry import get_adapter

logger = logging.getLogger(__name__)

convert_api_bp = Blueprint('convert_api', __name__)

# 格式名称映射
_FORMAT_MAP = {
    "onnx": "onnx", "torchscript": "torchscript", "pt": "pt",
    "pth": "pt", "pb": "pb", "savedmodel": "savedmodel"
}


@convert_api_bp.post("/convert-format")
def convert_format():
    """格式转换接口
    ---
    tags:
      - 格式转换
    summary: 将模型从一种格式转换为另一种格式
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - model_dir
            - result_dir
            - target_formats
          properties:
            model_dir:
              type: string
              description: 模型目录路径
            result_dir:
              type: string
              description: 结果输出目录路径
            target_formats:
              type: array
              items:
                type: string
              description: 目标格式列表（如 ["onnx", "torchscript"]）
              example: ["onnx"]
            model_file:
              type: string
              description: 可选，指定要转换的模型文件（相对于model_dir）
    responses:
      200:
        description: 格式转换成功
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            message:
              type: string
              example: success
            data:
              type: object
              properties:
                job_id:
                  type: string
                result_dir:
                  type: string
                artifacts:
                  type: array
                  items:
                    type: string
                converted_formats:
                  type: object
      400:
        description: 请求参数错误
      500:
        description: 服务器内部错误
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        model_dir = data.get("model_dir")
        result_dir = data.get("result_dir")
        target_formats = data.get("target_formats", [])
        model_file = data.get("model_file")  # 可选，指定要转换的模型文件
        
        # 参数验证
        if not model_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "model_dir is required"
            )), 400
        
        if not result_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "result_dir is required"
            )), 400
        
        if not target_formats or not isinstance(target_formats, list):
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "target_formats is required and must be a list"
            )), 400
        
        # 验证路径
        try:
            model_dir = PathManager.validate_model_dir(model_dir)
            result_dir = PathManager.validate_result_dir(result_dir, create_if_not_exists=True)
        except ValueError as e:
            return jsonify(create_error_response(
                ErrorCode.PATH_INVALID,
                str(e)
            )), 400
        
        # 检测模型信息
        detector = ModelDetector()
        detection = detector.detect_from_dir(model_dir)
        framework = detection["framework"]
        family = detection["family"]
        original_format = detection.get("original_format")

        # generic时尝试从同级raw目录重新检测
        if model_file and family == "generic":
            raw_dir = os.path.join(os.path.dirname(model_dir.rstrip("/\\")), "raw")
            if os.path.isdir(raw_dir):
                try:
                    raw_det = detector.detect_from_dir(raw_dir)
                    if raw_det.get("family", "generic") != "generic":
                        framework, family = raw_det.get("framework", framework), raw_det["family"]
                        original_format = raw_det.get("original_format", original_format)
                except Exception:
                    pass
        logger.info(f"[convert-format] framework={framework}, family={family}")
        
        if not framework or not family:
            return jsonify(create_error_response(
                ErrorCode.MODEL_NOT_FOUND,
                f"Cannot detect model framework or family from {model_dir}"
            )), 400
        
        # 获取adapter
        adapter_class = get_adapter(framework, family)
        if not adapter_class:
            return jsonify(create_error_response(
                ErrorCode.ADAPTER_NOT_FOUND,
                f"No adapter found for framework={framework}, family={family}"
            )), 400
        
        # 如果指定了model_file，验证文件存在
        if model_file:
            model_file_path = os.path.join(model_dir, model_file)
            if not os.path.exists(model_file_path):
                return jsonify(create_error_response(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Model file not found: {model_file_path}"
                )), 400
        
        # 创建adapter实例
        adapter = adapter_class(model_dir, result_dir, model_file=model_file)
        
        # 加载模型
        try:
            adapter.load()
            if adapter.model is None:
                return jsonify(create_error_response(ErrorCode.MODEL_LOAD_FAILED, "Failed to load model")), 400
        except Exception as e:
            logger.error(f"Model load failed: {e}", exc_info=True)
            return jsonify(create_error_response(ErrorCode.MODEL_LOAD_FAILED, f"Model load failed: {str(e)}")), 400
        
        # 执行格式转换
        try:
            # 检查是否尝试将INT8模型转换为ONNX（不支持）
            # 1. 检查指定的 model_file
            check_files = [model_file] if model_file else []
            # 2. 如果没指定，检查adapter找到的权重文件
            if not check_files:
                weight = adapter._find_weight()
                if weight:
                    check_files.append(os.path.basename(weight))
            
            for fname in check_files:
                if fname and "int8" in fname.lower() and "onnx" in [str(f).lower() for f in target_formats]:
                    return jsonify(create_error_response(
                        ErrorCode.EXPORT_FAILED,
                        f"Cannot convert INT8 quantized model '{fname}' to ONNX. "
                        "Please convert from the original FP32/FP16 model."
                    )), 400

            # 标准化格式名称
            normalized_formats = list(dict.fromkeys(
                _FORMAT_MAP.get(str(f).lower(), str(f).lower()) for f in target_formats
            ))
            
            # 执行导出
            try:
                artifacts = adapter.export(normalized_formats, normalized_formats)
            except ValueError as e:
                if "INT8" in str(e):
                    return jsonify(create_error_response(ErrorCode.EXPORT_FAILED, str(e))), 400
                raise
            if not artifacts:
                return jsonify(create_error_response(
                    ErrorCode.EXPORT_FAILED,
                    f"Format conversion failed: no files generated for formats {target_formats}"
                )), 400
            
            # 构建统一格式的响应
            job_id = f"convert_{int(time.time())}"
            
            # 计算文件大小
            size_before = size_after = None
            try:
                src_file = adapter._find_weight()
                if src_file and os.path.exists(src_file):
                    size_before = round(os.path.getsize(src_file) / (1024 * 1024), 4)
                if artifacts:
                    size_after = round(os.path.getsize(artifacts[0]) / (1024 * 1024), 4)
            except Exception:
                pass
            
            # 生成outputs列表
            outputs = []
            for path in artifacts:
                rel_path = os.path.relpath(path, result_dir).replace("\\", "/")
                ext = os.path.splitext(path)[1].lower()
                out_type = "onnx_model" if ext == ".onnx" else "torchscript_model" if "torchscript" in path.lower() else "converted_model"
                outputs.append({"type": out_type, "path": rel_path})
            
            return jsonify(create_success_response({
                "job_id": job_id,
                "result_dir": result_dir,
                "operations": [{"operation": "convert", "from": original_format, "to": normalized_formats, "status": "success"}],
                "outputs": outputs,
                "metrics": {"size_before_mb": size_before, "size_after_mb": size_after}
            }))
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}", exc_info=True)
            return jsonify(create_error_response(
                ErrorCode.EXPORT_FAILED,
                f"Format conversion failed: {str(e)}"
            )), 500
        
    except APIError as e:
        return jsonify(e.to_dict()), e.code
    except Exception as e:
        logger.error(f"Error in convert_format: {e}", exc_info=True)
        return jsonify(create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(e)}"
        )), 500


@convert_api_bp.post("/list-supported-formats")
def list_supported_formats():
    """列出支持的格式转换
    ---
    tags:
      - 格式转换
    summary: 列出模型支持的格式转换选项
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - model_dir
          properties:
            model_dir:
              type: string
              description: 模型目录路径
    responses:
      200:
        description: 成功返回支持的格式列表
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            message:
              type: string
              example: success
            data:
              type: object
              properties:
                framework:
                  type: string
                  example: pytorch
                family:
                  type: string
                  example: resnet
                original_format:
                  type: string
                  example: pt
                supported_target_formats:
                  type: array
                  items:
                    type: string
                  example: ["onnx", "torchscript", "pt"]
      400:
        description: 请求参数错误
      500:
        description: 服务器内部错误
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        model_dir = data.get("model_dir")
        
        if not model_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "model_dir is required"
            )), 400
        
        try:
            model_dir = PathManager.validate_model_dir(model_dir)
        except ValueError as e:
            return jsonify(create_error_response(
                ErrorCode.PATH_INVALID,
                str(e)
            )), 400
        
        # 检测模型信息
        detector = ModelDetector()
        detection = detector.detect_from_dir(model_dir)
        framework = detection["framework"]
        family = detection["family"]
        original_format = detection.get("original_format")
        
        if not framework or not family:
            return jsonify(create_error_response(
                ErrorCode.MODEL_NOT_FOUND,
                f"Cannot detect model framework or family from {model_dir}"
            )), 400
        
        # 根据framework和family确定支持的格式
        supported_formats = []
        
        if framework == "pytorch":
            supported_formats = ["onnx", "torchscript", "pt"]
        elif framework == "tensorflow":
            supported_formats = ["onnx", "pb", "savedmodel"]
        elif framework == "paddlepaddle":
            supported_formats = ["onnx", "paddle_infer"]
        elif framework == "onnx":
            supported_formats = ["onnx", "pt"]  # ONNX可以转回PyTorch（部分支持）
        
        return jsonify(create_success_response({
            "framework": framework,
            "family": family,
            "original_format": original_format,
            "supported_target_formats": supported_formats
        }))
        
    except APIError as e:
        return jsonify(e.to_dict()), e.code
    except Exception as e:
        logger.error(f"Error in list_supported_formats: {e}", exc_info=True)
        return jsonify(create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(e)}"
        )), 500

