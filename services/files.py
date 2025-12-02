"""文件管理服务"""
import os
import logging
from zipfile import ZipFile, is_zipfile
from typing import Optional, Dict, List
from utils.path import PathManager

logger = logging.getLogger(__name__)


class ExtraFilesManager:
    """Extra目录文件管理器"""
    
    SUBDIRS = {
        "calibration_data": "calibration_data",
        "train_data": "train_data",
        "val_data": "val_data",
        "teacher_model": "teacher_model",
        "metadata": "metadata"
    }
    
    # 目录名识别映射（支持多种命名）
    RECOGNITION_MAP = {
        "calibration_data": ["calibration_data", "calib", "calibration"],
        "train_data": ["train_data", "train", "training"],
        "val_data": ["val_data", "val", "validation", "valid"],
        "teacher_model": ["teacher_model", "teacher"]
    }
    
    def __init__(self, extra_dir: Optional[str] = None):
        self.extra_dir = extra_dir
    
    def get_calib_dir(self) -> Optional[str]:
        return self._get_subdir("calibration_data")
    
    def get_train_data_dir(self) -> Optional[str]:
        return self._get_subdir("train_data")
    
    def get_val_data_dir(self) -> Optional[str]:
        return self._get_subdir("val_data")
    
    def get_teacher_model_dir(self) -> Optional[str]:
        return self._get_subdir("teacher_model")
    
    def get_metadata_dir(self) -> Optional[str]:
        return self._get_subdir("metadata")
    
    def _get_subdir(self, name: str) -> Optional[str]:
        if not self.extra_dir or not os.path.exists(self.extra_dir):
            return None
        path = os.path.join(self.extra_dir, self.SUBDIRS.get(name, name))
        return path if os.path.isdir(path) else None
    
    def list_available_files(self) -> Dict[str, List[str]]:
        result = {}
        for key in self.SUBDIRS:
            subdir = self._get_subdir(key)
            if subdir:
                files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]
                if files:
                    result[key] = files
        return result
    
    def check_requirements(self, required: List[str]) -> Dict[str, bool]:
        return {r: self._get_subdir(r) is not None for r in required}
    
    def extract_and_distribute(self, zip_path: str) -> Dict[str, List[str]]:
        if not self.extra_dir:
            raise ValueError("extra_dir is not set")
        if not os.path.exists(zip_path) or not is_zipfile(zip_path):
            raise ValueError(f"Invalid zip file: {zip_path}")
        
        PathManager.ensure_dir(self.extra_dir)
        result = {}
        
        with ZipFile(zip_path, 'r') as zf:
            # 获取所有顶层目录
            top_dirs = self._get_top_level_dirs(zf)
            
            for dir_name in top_dirs:
                file_type = self._identify_type(dir_name)
                if file_type:
                    target = os.path.join(self.extra_dir, self.SUBDIRS[file_type])
                    PathManager.ensure_dir(target)
                    files = self._extract_dir(zf, dir_name, target)
                    if files:
                        result[file_type] = files
        
        return result
    
    def _get_top_level_dirs(self, zf: ZipFile) -> List[str]:
        dirs = set()
        for name in zf.namelist():
            if ".." in name:
                continue
            parts = name.replace('\\', '/').strip('/').split('/')
            if len(parts) > 1:
                dirs.add(parts[0])
        return sorted(dirs)
    
    def _identify_type(self, dir_name: str) -> Optional[str]:
        name = dir_name.lower().strip()
        # 精确匹配
        for ftype, patterns in self.RECOGNITION_MAP.items():
            if name in patterns:
                return ftype
        # 关键词匹配
        if "calib" in name:
            return "calibration_data"
        if "teacher" in name:
            return "teacher_model"
        if "train" in name:
            return "train_data"
        if "val" in name:
            return "val_data"
        return None
    
    def _extract_dir(self, zf: ZipFile, dir_name: str, target: str) -> List[str]:
        prefix = dir_name + "/"
        extracted = []
        for name in zf.namelist():
            normalized = name.replace('\\', '/')
            if not normalized.startswith(prefix) or ".." in name:
                continue
            rel = normalized[len(prefix):]
            if not rel or normalized.endswith('/'):
                continue
            # 确保父目录存在
            target_path = os.path.join(target, rel)
            PathManager.ensure_dir(os.path.dirname(target_path))
            # 提取文件
            with zf.open(name) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
            extracted.append(rel)
        return extracted
