"""
文件工具模块
提供文件操作、数据读写、压缩加密、格式转换等功能
"""
import os
import sys
import json
import yaml
import pickle
import csv
import zipfile
import tarfile
import gzip
import bz2
import lzma
import zlib
import hashlib
import base64
import struct
import mimetypes
import tempfile
import shutil
import fnmatch
import re
import stat
import pathlib
import itertools
import contextlib
import warnings
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO, TextIO, Callable, Iterator
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import math
import hashlib


class FileFormat(str, Enum):
    """文件格式枚举"""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"
    XML = "xml"
    CSV = "csv"
    TSV = "tsv"
    TXT = "txt"
    BINARY = "binary"
    PICKLE = "pickle"
    NPY = "npy"
    NPZ = "npz"
    HDF5 = "hdf5"
    PARQUET = "parquet"
    FEATHER = "feather"
    EXCEL = "excel"
    SQLITE = "sqlite"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class CompressionMethod(str, Enum):
    """压缩方法枚举"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZLIB = "zlib"
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    TAR_XZ = "tar.xz"


class EncryptionMethod(str, Enum):
    """加密方法枚举"""
    NONE = "none"
    AES = "aes"
    XOR = "xor"
    BASE64 = "base64"


class HashAlgorithm(str, Enum):
    """哈希算法枚举"""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class FileType(str, Enum):
    """文件类型枚举"""
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    DEVICE = "device"
    FIFO = "fifo"
    SOCKET = "socket"
    UNKNOWN = "unknown"


class FileOperationError(Exception):
    """文件操作异常"""
    pass


class FileValidationError(Exception):
    """文件验证异常"""
    pass


class FileEncryptionError(Exception):
    """文件加密异常"""
    pass


class FileCompressionError(Exception):
    """文件压缩异常"""
    pass


@contextlib.contextmanager
def safe_open(filepath: Union[str, Path], mode: str = 'r', 
             encoding: str = 'utf-8', errors: str = None, 
             newline: str = None, **kwargs):
    """
    安全打开文件上下文管理器
    
    Args:
        filepath: 文件路径
        mode: 打开模式
        encoding: 编码
        errors: 错误处理
        newline: 换行符
        **kwargs: 其他参数
        
    Yields:
        文件对象
    """
    filepath = Path(filepath)
    file_obj = None
    
    try:
        if 'b' in mode:
            # 二进制模式
            file_obj = open(filepath, mode, **kwargs)
        else:
            # 文本模式
            file_obj = open(filepath, mode, encoding=encoding, errors=errors, 
                          newline=newline, **kwargs)
        
        yield file_obj
        
    except Exception as e:
        raise FileOperationError(f"无法打开文件 {filepath}: {e}")
    
    finally:
        if file_obj:
            file_obj.close()


def ensure_directory(dirpath: Union[str, Path], parents: bool = True, 
                    exist_ok: bool = True) -> Path:
    """
    确保目录存在
    
    Args:
        dirpath: 目录路径
        parents: 是否创建父目录
        exist_ok: 存在时是否抛出异常
        
    Returns:
        Path: 目录路径对象
    """
    dirpath = Path(dirpath)
    
    try:
        dirpath.mkdir(parents=parents, exist_ok=exist_ok)
        return dirpath
    except Exception as e:
        raise FileOperationError(f"无法创建目录 {dirpath}: {e}")


def ensure_parent_directory(filepath: Union[str, Path]) -> Path:
    """
    确保文件的父目录存在
    
    Args:
        filepath: 文件路径
        
    Returns:
        Path: 父目录路径
    """
    filepath = Path(filepath)
    return ensure_directory(filepath.parent)


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        filepath: 文件路径
        
    Returns:
        Dict[str, Any]: 文件信息
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        stat_info = filepath.stat()
        
        # 文件类型
        if filepath.is_dir():
            file_type = FileType.DIRECTORY
        elif filepath.is_file():
            file_type = FileType.FILE
        elif filepath.is_symlink():
            file_type = FileType.SYMLINK
        else:
            file_type = FileType.UNKNOWN
        
        # 文件权限
        permissions = {
            'readable': os.access(filepath, os.R_OK),
            'writable': os.access(filepath, os.W_OK),
            'executable': os.access(filepath, os.X_OK)
        }
        
        # 文件格式
        file_format = detect_file_format(filepath)
        
        info = {
            'path': str(filepath.absolute()),
            'name': filepath.name,
            'stem': filepath.stem,
            'suffix': filepath.suffix,
            'suffixes': filepath.suffixes,
            'type': file_type.value,
            'size': stat_info.st_size,
            'created': datetime.fromtimestamp(stat_info.st_ctime),
            'modified': datetime.fromtimestamp(stat_info.st_mtime),
            'accessed': datetime.fromtimestamp(stat_info.st_atime),
            'permissions': permissions,
            'owner': stat_info.st_uid,
            'group': stat_info.st_gid,
            'inode': stat_info.st_ino,
            'device': stat_info.st_dev,
            'format': file_format.value,
            'mime_type': mimetypes.guess_type(str(filepath))[0],
            'is_hidden': filepath.name.startswith('.')
        }
        
        return info
    
    except Exception as e:
        raise FileOperationError(f"无法获取文件信息 {filepath}: {e}")


def detect_file_format(filepath: Union[str, Path]) -> FileFormat:
    """
    检测文件格式
    
    Args:
        filepath: 文件路径
        
    Returns:
        FileFormat: 文件格式
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return FileFormat.UNKNOWN
    
    # 获取扩展名
    suffix = filepath.suffix.lower()
    
    # 常见格式映射
    format_map = {
        '.json': FileFormat.JSON,
        '.yaml': FileFormat.YAML,
        '.yml': FileFormat.YAML,
        '.xml': FileFormat.XML,
        '.csv': FileFormat.CSV,
        '.tsv': FileFormat.TSV,
        '.txt': FileFormat.TXT,
        '.pickle': FileFormat.PICKLE,
        '.pkl': FileFormat.PICKLE,
        '.npy': FileFormat.NPY,
        '.npz': FileFormat.NPZ,
        '.h5': FileFormat.HDF5,
        '.hdf5': FileFormat.HDF5,
        '.parquet': FileFormat.PARQUET,
        '.feather': FileFormat.FEATHER,
        '.xlsx': FileFormat.EXCEL,
        '.xls': FileFormat.EXCEL,
        '.db': FileFormat.SQLITE,
        '.sqlite': FileFormat.SQLITE,
        '.sqlite3': FileFormat.SQLITE,
    }
    
    # 图片格式
    image_suffixes = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    if suffix in image_suffixes:
        return FileFormat.IMAGE
    
    # 音频格式
    audio_suffixes = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
    if suffix in audio_suffixes:
        return FileFormat.AUDIO
    
    # 视频格式
    video_suffixes = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'}
    if suffix in video_suffixes:
        return FileFormat.VIDEO
    
    # 压缩格式
    compression_suffixes = {'.gz', '.bz2', '.xz', '.zip', '.tar', '.7z', '.rar'}
    if suffix in compression_suffixes:
        return FileFormat.BINARY
    
    # 检查实际内容
    if suffix not in format_map:
        try:
            with open(filepath, 'rb') as f:
                # 读取文件头部
                header = f.read(1024)
                
                # 检查JSON
                try:
                    json.loads(header.decode('utf-8', errors='ignore'))
                    return FileFormat.JSON
                except:
                    pass
                
                # 检查YAML
                try:
                    yaml.safe_load(header.decode('utf-8', errors='ignore'))
                    return FileFormat.YAML
                except:
                    pass
                
                # 检查CSV/TSV
                try:
                    text = header.decode('utf-8', errors='ignore')
                    if ',' in text or '\t' in text:
                        lines = text.split('\n')
                        if len(lines) > 1:
                            return FileFormat.CSV
                    return FileFormat.TXT
                except:
                    pass
                
                # 检查二进制
                if header.startswith(b'\x89PNG') or header.startswith(b'\xff\xd8'):
                    return FileFormat.IMAGE
                elif header.startswith(b'ID3') or header.startswith(b'RIFF'):
                    return FileFormat.AUDIO
                
        except:
            pass
        
        return FileFormat.UNKNOWN
    
    return format_map.get(suffix, FileFormat.UNKNOWN)


def calculate_file_hash(filepath: Union[str, Path], 
                       algorithm: HashAlgorithm = HashAlgorithm.SHA256,
                       chunk_size: int = 8192) -> str:
    """
    计算文件哈希值
    
    Args:
        filepath: 文件路径
        algorithm: 哈希算法
        chunk_size: 块大小
        
    Returns:
        str: 哈希值
    """
    filepath = Path(filepath)
    
    if not filepath.exists() or not filepath.is_file():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 选择哈希算法
    if algorithm == HashAlgorithm.MD5:
        hasher = hashlib.md5()
    elif algorithm == HashAlgorithm.SHA1:
        hasher = hashlib.sha1()
    elif algorithm == HashAlgorithm.SHA256:
        hasher = hashlib.sha256()
    elif algorithm == HashAlgorithm.SHA512:
        hasher = hashlib.sha512()
    elif algorithm == HashAlgorithm.BLAKE2B:
        hasher = hashlib.blake2b()
    elif algorithm == HashAlgorithm.BLAKE2S:
        hasher = hashlib.blake2s()
    else:
        raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    try:
        with open(filepath, 'rb') as f:
            # 分块读取文件
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    except Exception as e:
        raise FileOperationError(f"无法计算文件哈希 {filepath}: {e}")


def calculate_directory_hash(dirpath: Union[str, Path], 
                           algorithm: HashAlgorithm = HashAlgorithm.SHA256,
                           include_hidden: bool = False) -> str:
    """
    计算目录哈希值（基于所有文件）
    
    Args:
        dirpath: 目录路径
        algorithm: 哈希算法
        include_hidden: 是否包含隐藏文件
        
    Returns:
        str: 目录哈希值
    """
    dirpath = Path(dirpath)
    
    if not dirpath.exists() or not dirpath.is_dir():
        raise FileNotFoundError(f"目录不存在: {dirpath}")
    
    hasher = hashlib.sha256() if algorithm == HashAlgorithm.SHA256 else hashlib.md5()
    
    try:
        # 获取所有文件（按路径排序以确保一致性）
        all_files = []
        for root, dirs, files in os.walk(dirpath):
            # 过滤隐藏文件/目录
            if not include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                files = [f for f in files if not f.startswith('.')]
            
            for file in sorted(files):
                filepath = Path(root) / file
                all_files.append(filepath)
        
        # 对每个文件计算哈希并组合
        for filepath in sorted(all_files):
            # 添加文件路径到哈希
            hasher.update(str(filepath.relative_to(dirpath)).encode('utf-8'))
            
            # 添加文件内容哈希
            file_hash = calculate_file_hash(filepath, algorithm)
            hasher.update(file_hash.encode('utf-8'))
        
        return hasher.hexdigest()
    
    except Exception as e:
        raise FileOperationError(f"无法计算目录哈希 {dirpath}: {e}")


def copy_file(source: Union[str, Path], destination: Union[str, Path], 
             overwrite: bool = False, preserve_metadata: bool = True) -> Path:
    """
    复制文件
    
    Args:
        source: 源文件路径
        destination: 目标文件路径
        overwrite: 是否覆盖已存在文件
        preserve_metadata: 是否保留元数据
        
    Returns:
        Path: 目标文件路径
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    if destination.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在: {destination}")
    
    # 确保目标目录存在
    ensure_parent_directory(destination)
    
    try:
        if preserve_metadata:
            shutil.copy2(source, destination)
        else:
            shutil.copy(source, destination)
        
        return destination
    
    except Exception as e:
        raise FileOperationError(f"无法复制文件 {source} -> {destination}: {e}")


def move_file(source: Union[str, Path], destination: Union[str, Path], 
             overwrite: bool = False) -> Path:
    """
    移动文件
    
    Args:
        source: 源文件路径
        destination: 目标文件路径
        overwrite: 是否覆盖已存在文件
        
    Returns:
        Path: 目标文件路径
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    if destination.exists():
        if overwrite:
            delete_file(destination)
        else:
            raise FileExistsError(f"目标文件已存在: {destination}")
    
    # 确保目标目录存在
    ensure_parent_directory(destination)
    
    try:
        shutil.move(str(source), str(destination))
        return destination
    
    except Exception as e:
        raise FileOperationError(f"无法移动文件 {source} -> {destination}: {e}")


def delete_file(filepath: Union[str, Path], force: bool = False) -> bool:
    """
    删除文件
    
    Args:
        filepath: 文件路径
        force: 强制删除（忽略错误）
        
    Returns:
        bool: 是否成功
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return True
    
    try:
        if filepath.is_file() or filepath.is_symlink():
            filepath.unlink()
        elif filepath.is_dir():
            shutil.rmtree(filepath)
        else:
            # 特殊文件
            os.remove(filepath)
        
        return True
    
    except Exception as e:
        if force:
            try:
                # 尝试强制删除
                if sys.platform == 'win32':
                    subprocess.run(['del', '/f', '/q', str(filepath)], shell=True, 
                                 capture_output=True)
                else:
                    subprocess.run(['rm', '-rf', str(filepath)], capture_output=True)
                return True
            except:
                pass
        
        raise FileOperationError(f"无法删除文件 {filepath}: {e}")


def rename_file(filepath: Union[str, Path], new_name: str, 
               overwrite: bool = False) -> Path:
    """
    重命名文件
    
    Args:
        filepath: 文件路径
        new_name: 新文件名
        overwrite: 是否覆盖已存在文件
        
    Returns:
        Path: 新文件路径
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    new_path = filepath.parent / new_name
    
    if new_path.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在: {new_path}")
    
    try:
        if new_path.exists() and overwrite:
            delete_file(new_path)
        
        filepath.rename(new_path)
        return new_path
    
    except Exception as e:
        raise FileOperationError(f"无法重命名文件 {filepath} -> {new_path}: {e}")


def find_files(directory: Union[str, Path], 
              pattern: str = "*", 
              recursive: bool = True,
              case_sensitive: bool = False,
              include_hidden: bool = False) -> List[Path]:
    """
    查找文件
    
    Args:
        directory: 目录路径
        pattern: 文件模式（支持通配符）
        recursive: 是否递归查找
        case_sensitive: 是否区分大小写
        include_hidden: 是否包含隐藏文件
        
    Returns:
        List[Path]: 找到的文件列表
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    try:
        if not recursive:
            # 非递归查找
            files = list(directory.glob(pattern))
        else:
            # 递归查找
            files = []
            for filepath in directory.rglob(pattern):
                if not include_hidden and filepath.name.startswith('.'):
                    continue
                files.append(filepath)
        
        # 大小写敏感过滤
        if not case_sensitive:
            pattern_lower = pattern.lower()
            files = [f for f in files if fnmatch.fnmatch(f.name.lower(), pattern_lower)]
        
        return sorted(files)
    
    except Exception as e:
        raise FileOperationError(f"无法查找文件 {directory}: {e}")


def read_text_file(filepath: Union[str, Path], encoding: str = 'utf-8', 
                  errors: str = 'strict') -> str:
    """
    读取文本文件
    
    Args:
        filepath: 文件路径
        encoding: 编码
        errors: 错误处理
        
    Returns:
        str: 文件内容
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'r', encoding=encoding, errors=errors) as f:
            return f.read()
    
    except Exception as e:
        raise FileOperationError(f"无法读取文件 {filepath}: {e}")


def write_text_file(filepath: Union[str, Path], content: str, 
                   encoding: str = 'utf-8', mode: str = 'w', 
                   errors: str = 'strict') -> Path:
    """
    写入文本文件
    
    Args:
        filepath: 文件路径
        content: 内容
        encoding: 编码
        mode: 写入模式
        errors: 错误处理
        
    Returns:
        Path: 文件路径
    """
    filepath = Path(filepath)
    
    # 确保目录存在
    ensure_parent_directory(filepath)
    
    try:
        with open(filepath, mode, encoding=encoding, errors=errors) as f:
            f.write(content)
        
        return filepath
    
    except Exception as e:
        raise FileOperationError(f"无法写入文件 {filepath}: {e}")


def read_binary_file(filepath: Union[str, Path]) -> bytes:
    """
    读取二进制文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        bytes: 文件内容
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            return f.read()
    
    except Exception as e:
        raise FileOperationError(f"无法读取二进制文件 {filepath}: {e}")


def write_binary_file(filepath: Union[str, Path], data: bytes, 
                     mode: str = 'wb') -> Path:
    """
    写入二进制文件
    
    Args:
        filepath: 文件路径
        data: 二进制数据
        mode: 写入模式
        
    Returns:
        Path: 文件路径
    """
    filepath = Path(filepath)
    
    # 确保目录存在
    ensure_parent_directory(filepath)
    
    try:
        with open(filepath, mode) as f:
            f.write(data)
        
        return filepath
    
    except Exception as e:
        raise FileOperationError(f"无法写入二进制文件 {filepath}: {e}")


def read_json_file(filepath: Union[str, Path], encoding: str = 'utf-8') -> Any:
    """
    读取JSON文件
    
    Args:
        filepath: 文件路径
        encoding: 编码
        
    Returns:
        Any: JSON数据
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)
    
    except json.JSONDecodeError as e:
        raise FileValidationError(f"无效的JSON格式 {filepath}: {e}")
    except Exception as e:
        raise FileOperationError(f"无法读取JSON文件 {filepath}: {e}")


def write_json_file(filepath: Union[str, Path], data: Any, 
                   indent: int = 2, encoding: str = 'utf-8', 
                   ensure_ascii: bool = False) -> Path:
    """
    写入JSON文件
    
    Args:
        filepath: 文件路径
        data: 数据
        indent: 缩进
        encoding: 编码
        ensure_ascii: 是否确保ASCII
        
    Returns:
        Path: 文件路径
    """
    filepath = Path(filepath)
    
    # 确保目录存在
    ensure_parent_directory(filepath)
    
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        return filepath
    
    except Exception as e:
        raise FileOperationError(f"无法写入JSON文件 {filepath}: {e}")


def read_yaml_file(filepath: Union[str, Path], encoding: str = 'utf-8') -> Any:
    """
    读取YAML文件
    
    Args:
        filepath: 文件路径
        encoding: 编码
        
    Returns:
        Any: YAML数据
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return yaml.safe_load(f)
    
    except yaml.YAMLError as e:
        raise FileValidationError(f"无效的YAML格式 {filepath}: {e}")
    except Exception as e:
        raise FileOperationError(f"无法读取YAML文件 {filepath}: {e}")


def write_yaml_file(filepath: Union[str, Path], data: Any, 
                   encoding: str = 'utf-8', default_flow_style: bool = False) -> Path:
    """
    写入YAML文件
    
    Args:
        filepath: 文件路径
        data: 数据
        encoding: 编码
        default_flow_style: 默认流样式
        
    Returns:
        Path: 文件路径
    """
    filepath = Path(filepath)
    
    # 确保目录存在
    ensure_parent_directory(filepath)
    
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            yaml.dump(data, f, default_flow_style=default_flow_style, 
                     allow_unicode=True)
        
        return filepath
    
    except Exception as e:
        raise FileOperationError(f"无法写入YAML文件 {filepath}: {e}")


def read_csv_file(filepath: Union[str, Path], encoding: str = 'utf-8', 
                 delimiter: str = ',', **kwargs) -> List[Dict[str, Any]]:
    """
    读取CSV文件
    
    Args:
        filepath: 文件路径
        encoding: 编码
        delimiter: 分隔符
        **kwargs: pandas.read_csv参数
        
    Returns:
        List[Dict[str, Any]]: CSV数据
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        # 使用pandas读取
        df = pd.read_csv(filepath, encoding=encoding, delimiter=delimiter, **kwargs)
        
        # 转换为字典列表
        return df.to_dict('records')
    
    except Exception as e:
        raise FileOperationError(f"无法读取CSV文件 {filepath}: {e}")


def write_csv_file(filepath: Union[str, Path], data: List[Dict[str, Any]], 
                  encoding: str = 'utf-8', delimiter: str = ',', 
                  write_header: bool = True, **kwargs) -> Path:
    """
    写入CSV文件
    
    Args:
        filepath: 文件路径
        data: 数据
        encoding: 编码
        delimiter: 分隔符
        write_header: 是否写入表头
        **kwargs: pandas.to_csv参数
        
    Returns:
        Path: 文件路径
    """
    filepath = Path(filepath)
    
    # 确保目录存在
    ensure_parent_directory(filepath)
    
    try:
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 写入CSV
        df.to_csv(filepath, encoding=encoding, sep=delimiter, 
                 index=False, header=write_header, **kwargs)
        
        return filepath
    
    except Exception as e:
        raise FileOperationError(f"无法写入CSV文件 {filepath}: {e}")


def read_pickle_file(filepath: Union[str, Path]) -> Any:
    """
    读取Pickle文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        Any: Pickle数据
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    except pickle.PickleError as e:
        raise FileValidationError(f"无效的Pickle格式 {filepath}: {e}")
    except Exception as e:
        raise FileOperationError(f"无法读取Pickle文件 {filepath}: {e}")


def write_pickle_file(filepath: Union[str, Path], data: Any, 
                     protocol: int = pickle.HIGHEST_PROTOCOL) -> Path:
    """
    写入Pickle文件
    
    Args:
        filepath: 文件路径
        data: 数据
        protocol: Pickle协议版本
        
    Returns:
        Path: 文件路径
    """
    filepath = Path(filepath)
    
    # 确保目录存在
    ensure_parent_directory(filepath)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        
        return filepath
    
    except Exception as e:
        raise FileOperationError(f"无法写入Pickle文件 {filepath}: {e}")


def read_numpy_file(filepath: Union[str, Path]) -> np.ndarray:
    """
    读取NumPy文件（.npy, .npz）
    
    Args:
        filepath: 文件路径
        
    Returns:
        np.ndarray: NumPy数组
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        if filepath.suffix == '.npy':
            return np.load(filepath)
        elif filepath.suffix == '.npz':
            # 返回字典
            data = np.load(filepath)
            result = {key: data[key] for key in data.files}
            data.close()
            return result
        else:
            raise ValueError(f"不支持的NumPy文件格式: {filepath.suffix}")
    
    except Exception as e:
        raise FileOperationError(f"无法读取NumPy文件 {filepath}: {e}")


def write_numpy_file(filepath: Union[str, Path], data: np.ndarray, 
                    compressed: bool = False) -> Path:
    """
    写入NumPy文件
    
    Args:
        filepath: 文件路径
        data: NumPy数组
        compressed: 是否压缩
        
    Returns:
        Path: 文件路径
    """
    filepath = Path(filepath)
    
    # 确保目录存在
    ensure_parent_directory(filepath)
    
    try:
        if compressed or filepath.suffix == '.npz':
            np.savez_compressed(filepath, data=data)
        else:
            np.save(filepath, data)
        
        return filepath
    
    except Exception as e:
        raise FileOperationError(f"无法写入NumPy文件 {filepath}: {e}")


def compress_file(source: Union[str, Path], 
                 destination: Union[str, Path] = None,
                 method: CompressionMethod = CompressionMethod.GZIP,
                 compression_level: int = 6,
                 remove_source: bool = False) -> Path:
    """
    压缩文件
    
    Args:
        source: 源文件路径
        destination: 目标文件路径
        method: 压缩方法
        compression_level: 压缩级别
        remove_source: 压缩后是否删除源文件
        
    Returns:
        Path: 目标文件路径
    """
    source = Path(source)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    # 确定目标文件路径
    if destination is None:
        if method == CompressionMethod.GZIP:
            dest = source.with_suffix(source.suffix + '.gz')
        elif method == CompressionMethod.BZIP2:
            dest = source.with_suffix(source.suffix + '.bz2')
        elif method == CompressionMethod.LZMA:
            dest = source.with_suffix(source.suffix + '.xz')
        elif method == CompressionMethod.ZIP:
            dest = source.with_suffix('.zip')
        elif method == CompressionMethod.TAR:
            dest = source.with_suffix('.tar')
        elif method == CompressionMethod.TAR_GZ:
            dest = source.with_suffix('.tar.gz')
        elif method == CompressionMethod.TAR_BZ2:
            dest = source.with_suffix('.tar.bz2')
        elif method == CompressionMethod.TAR_XZ:
            dest = source.with_suffix('.tar.xz')
        else:
            dest = source.with_suffix(source.suffix + '.compressed')
    else:
        dest = Path(destination)
    
    # 确保目标目录存在
    ensure_parent_directory(dest)
    
    try:
        if method == CompressionMethod.GZIP:
            with open(source, 'rb') as f_in:
                with gzip.open(dest, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif method == CompressionMethod.BZIP2:
            with open(source, 'rb') as f_in:
                with bz2.open(dest, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif method == CompressionMethod.LZMA:
            with open(source, 'rb') as f_in:
                with lzma.open(dest, 'wb', preset=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif method == CompressionMethod.ZLIB:
            with open(source, 'rb') as f_in:
                compressed = zlib.compress(f_in.read(), level=compression_level)
                with open(dest, 'wb') as f_out:
                    f_out.write(compressed)
        
        elif method == CompressionMethod.ZIP:
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(source, arcname=source.name)
        
        elif method in [CompressionMethod.TAR, CompressionMethod.TAR_GZ, 
                       CompressionMethod.TAR_BZ2, CompressionMethod.TAR_XZ]:
            mode = 'w'
            if method == CompressionMethod.TAR_GZ:
                mode = 'w:gz'
            elif method == CompressionMethod.TAR_BZ2:
                mode = 'w:bz2'
            elif method == CompressionMethod.TAR_XZ:
                mode = 'w:xz'
            
            with tarfile.open(dest, mode) as tar:
                tar.add(source, arcname=source.name)
        
        else:
            raise ValueError(f"不支持的压缩方法: {method}")
        
        # 验证压缩文件
        if dest.exists() and dest.stat().st_size > 0:
            if remove_source:
                delete_file(source)
            return dest
        else:
            raise FileCompressionError(f"压缩文件创建失败: {dest}")
    
    except Exception as e:
        # 清理不完整的压缩文件
        if dest.exists():
            delete_file(dest, force=True)
        raise FileCompressionError(f"无法压缩文件 {source} -> {dest}: {e}")


def decompress_file(source: Union[str, Path], 
                   destination: Union[str, Path] = None,
                   method: CompressionMethod = None,
                   remove_source: bool = False) -> Path:
    """
    解压文件
    
    Args:
        source: 源文件路径
        destination: 目标文件路径
        method: 压缩方法（自动检测）
        remove_source: 解压后是否删除源文件
        
    Returns:
        Path: 目标文件路径
    """
    source = Path(source)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    # 自动检测压缩方法
    if method is None:
        suffix = source.suffix.lower()
        if suffix == '.gz' or source.suffixes[-2:] == ['.tar', '.gz']:
            method = CompressionMethod.TAR_GZ if 'tar' in source.suffixes else CompressionMethod.GZIP
        elif suffix == '.bz2' or source.suffixes[-2:] == ['.tar', '.bz2']:
            method = CompressionMethod.TAR_BZ2 if 'tar' in source.suffixes else CompressionMethod.BZIP2
        elif suffix == '.xz' or source.suffixes[-2:] == ['.tar', '.xz']:
            method = CompressionMethod.TAR_XZ if 'tar' in source.suffixes else CompressionMethod.LZMA
        elif suffix == '.zip':
            method = CompressionMethod.ZIP
        elif suffix == '.tar':
            method = CompressionMethod.TAR
        else:
            # 尝试检测
            with open(source, 'rb') as f:
                header = f.read(2)
                if header == b'\x1f\x8b':
                    method = CompressionMethod.GZIP
                elif header == b'BZ':
                    method = CompressionMethod.BZIP2
                elif header == b'\xfd7':
                    method = CompressionMethod.LZMA
                elif header == b'PK':
                    method = CompressionMethod.ZIP
                else:
                    # 尝试作为zlib
                    try:
                        f.seek(0)
                        zlib.decompress(f.read())
                        method = CompressionMethod.ZLIB
                    except:
                        raise ValueError(f"无法识别的压缩格式: {source}")
    
    # 确定目标文件路径
    if destination is None:
        if method in [CompressionMethod.GZIP, CompressionMethod.BZIP2, CompressionMethod.LZMA]:
            # 移除压缩后缀
            suffixes = source.suffixes
            if suffixes and suffixes[-1] in ['.gz', '.bz2', '.xz']:
                dest = source.with_suffix(''.join(suffixes[:-1]))
            else:
                dest = source.with_suffix('')
        elif method == CompressionMethod.ZIP:
            dest = source.with_suffix('')
        elif method in [CompressionMethod.TAR, CompressionMethod.TAR_GZ, 
                       CompressionMethod.TAR_BZ2, CompressionMethod.TAR_XZ]:
            dest = source.parent / source.stem.replace('.tar', '')
        else:
            dest = source.with_suffix('')
    else:
        dest = Path(destination)
    
    # 确保目标目录存在
    ensure_parent_directory(dest)
    
    try:
        if method == CompressionMethod.GZIP:
            with gzip.open(source, 'rb') as f_in:
                with open(dest, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif method == CompressionMethod.BZIP2:
            with bz2.open(source, 'rb') as f_in:
                with open(dest, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif method == CompressionMethod.LZMA:
            with lzma.open(source, 'rb') as f_in:
                with open(dest, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif method == CompressionMethod.ZLIB:
            with open(source, 'rb') as f_in:
                decompressed = zlib.decompress(f_in.read())
                with open(dest, 'wb') as f_out:
                    f_out.write(decompressed)
        
        elif method == CompressionMethod.ZIP:
            with zipfile.ZipFile(source, 'r') as zipf:
                # 解压所有文件
                zipf.extractall(dest.parent)
                # 如果是单个文件，重命名为目标文件名
                if len(zipf.namelist()) == 1:
                    extracted = dest.parent / zipf.namelist()[0]
                    if extracted != dest:
                        move_file(extracted, dest, overwrite=True)
        
        elif method in [CompressionMethod.TAR, CompressionMethod.TAR_GZ, 
                       CompressionMethod.TAR_BZ2, CompressionMethod.TAR_XZ]:
            mode = 'r'
            if method == CompressionMethod.TAR_GZ:
                mode = 'r:gz'
            elif method == CompressionMethod.TAR_BZ2:
                mode = 'r:bz2'
            elif method == CompressionMethod.TAR_XZ:
                mode = 'r:xz'
            
            with tarfile.open(source, mode) as tar:
                # 如果是单个文件，提取到目标文件
                members = tar.getmembers()
                if len(members) == 1 and members[0].isfile():
                    member = members[0]
                    with tar.extractfile(member) as f_in:
                        with open(dest, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    # 提取到目录
                    dest.mkdir(parents=True, exist_ok=True)
                    tar.extractall(dest)
        
        else:
            raise ValueError(f"不支持的压缩方法: {method}")
        
        # 验证解压文件
        if dest.exists() and dest.stat().st_size > 0:
            if remove_source:
                delete_file(source)
            return dest
        else:
            raise FileCompressionError(f"解压文件创建失败: {dest}")
    
    except Exception as e:
        # 清理不完整的解压文件
        if dest.exists():
            delete_file(dest, force=True)
        raise FileCompressionError(f"无法解压文件 {source} -> {dest}: {e}")


def encrypt_file(source: Union[str, Path], 
                destination: Union[str, Path] = None,
                key: str = None,
                method: EncryptionMethod = EncryptionMethod.AES,
                remove_source: bool = False) -> Path:
    """
    加密文件
    
    Args:
        source: 源文件路径
        destination: 目标文件路径
        key: 加密密钥
        method: 加密方法
        remove_source: 加密后是否删除源文件
        
    Returns:
        Path: 目标文件路径
    """
    source = Path(source)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    # 确定目标文件路径
    if destination is None:
        dest = source.with_suffix(source.suffix + '.enc')
    else:
        dest = Path(destination)
    
    # 确保目标目录存在
    ensure_parent_directory(dest)
    
    try:
        if method == EncryptionMethod.NONE:
            # 不加密，直接复制
            copy_file(source, dest, overwrite=True)
        
        elif method == EncryptionMethod.BASE64:
            # Base64编码
            with open(source, 'rb') as f_in:
                data = f_in.read()
                encoded = base64.b64encode(data)
                with open(dest, 'wb') as f_out:
                    f_out.write(encoded)
        
        elif method == EncryptionMethod.XOR:
            # XOR加密
            if not key:
                raise ValueError("XOR加密需要密钥")
            
            key_bytes = key.encode('utf-8')
            with open(source, 'rb') as f_in:
                data = f_in.read()
                # 简单XOR加密
                encrypted = bytearray()
                for i, byte in enumerate(data):
                    key_byte = key_bytes[i % len(key_bytes)]
                    encrypted.append(byte ^ key_byte)
                
                with open(dest, 'wb') as f_out:
                    f_out.write(encrypted)
        
        elif method == EncryptionMethod.AES:
            # AES加密
            try:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import pad
                from Crypto.Random import get_random_bytes
                
                if not key:
                    # 生成随机密钥
                    key = get_random_bytes(32)  # 256位
                elif isinstance(key, str):
                    # 确保密钥长度为16, 24或32字节
                    key_bytes = key.encode('utf-8')
                    if len(key_bytes) not in [16, 24, 32]:
                        # 使用SHA256哈希
                        key = hashlib.sha256(key_bytes).digest()
                    else:
                        key = key_bytes
                
                # 生成随机IV
                iv = get_random_bytes(16)
                
                with open(source, 'rb') as f_in:
                    data = f_in.read()
                
                # 加密
                cipher = AES.new(key, AES.MODE_CBC, iv)
                encrypted = cipher.encrypt(pad(data, AES.block_size))
                
                # 写入文件：IV + 加密数据
                with open(dest, 'wb') as f_out:
                    f_out.write(iv)
                    f_out.write(encrypted)
                
                # 保存密钥（如果提供了密钥，则不保存）
                if not key:
                    key_file = dest.with_suffix('.key')
                    with open(key_file, 'wb') as f_key:
                        f_key.write(key)
            
            except ImportError:
                warnings.warn("PyCryptodome未安装，使用简单的XOR加密")
                return encrypt_file(source, dest, key or 'default_key', 
                                  EncryptionMethod.XOR, remove_source)
        
        else:
            raise ValueError(f"不支持的加密方法: {method}")
        
        if remove_source:
            delete_file(source)
        
        return dest
    
    except Exception as e:
        raise FileEncryptionError(f"无法加密文件 {source} -> {dest}: {e}")


def decrypt_file(source: Union[str, Path], 
                destination: Union[str, Path] = None,
                key: Union[str, bytes] = None,
                method: EncryptionMethod = EncryptionMethod.AES,
                remove_source: bool = False) -> Path:
    """
    解密文件
    
    Args:
        source: 源文件路径
        destination: 目标文件路径
        key: 解密密钥
        method: 加密方法
        remove_source: 解密后是否删除源文件
        
    Returns:
        Path: 目标文件路径
    """
    source = Path(source)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    # 确定目标文件路径
    if destination is None:
        if source.suffix == '.enc':
            dest = source.with_suffix('')
        else:
            dest = source.with_suffix('.decrypted')
    else:
        dest = Path(destination)
    
    # 确保目标目录存在
    ensure_parent_directory(dest)
    
    try:
        if method == EncryptionMethod.NONE:
            # 不解密，直接复制
            copy_file(source, dest, overwrite=True)
        
        elif method == EncryptionMethod.BASE64:
            # Base64解码
            with open(source, 'rb') as f_in:
                encoded = f_in.read()
                decoded = base64.b64decode(encoded)
                with open(dest, 'wb') as f_out:
                    f_out.write(decoded)
        
        elif method == EncryptionMethod.XOR:
            # XOR解密
            if not key:
                raise ValueError("XOR解密需要密钥")
            
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            
            with open(source, 'rb') as f_in:
                encrypted = f_in.read()
                # XOR解密（与加密相同）
                decrypted = bytearray()
                for i, byte in enumerate(encrypted):
                    key_byte = key_bytes[i % len(key_bytes)]
                    decrypted.append(byte ^ key_byte)
                
                with open(dest, 'wb') as f_out:
                    f_out.write(decrypted)
        
        elif method == EncryptionMethod.AES:
            # AES解密
            try:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import unpad
                
                if not key:
                    # 尝试从.key文件读取密钥
                    key_file = source.with_suffix('.key')
                    if key_file.exists():
                        with open(key_file, 'rb') as f_key:
                            key = f_key.read()
                    else:
                        raise ValueError("AES解密需要密钥")
                elif isinstance(key, str):
                    # 确保密钥长度为16, 24或32字节
                    key_bytes = key.encode('utf-8')
                    if len(key_bytes) not in [16, 24, 32]:
                        # 使用SHA256哈希
                        key = hashlib.sha256(key_bytes).digest()
                    else:
                        key = key_bytes
                
                with open(source, 'rb') as f_in:
                    # 读取IV和加密数据
                    iv = f_in.read(16)
                    encrypted = f_in.read()
                
                # 解密
                cipher = AES.new(key, AES.MODE_CBC, iv)
                decrypted = cipher.decrypt(encrypted)
                
                # 去除填充
                try:
                    decrypted = unpad(decrypted, AES.block_size)
                except ValueError:
                    # 可能是无填充的
                    pass
                
                with open(dest, 'wb') as f_out:
                    f_out.write(decrypted)
            
            except ImportError:
                warnings.warn("PyCryptodome未安装，使用简单的XOR解密")
                return decrypt_file(source, dest, key or 'default_key', 
                                  EncryptionMethod.XOR, remove_source)
        
        else:
            raise ValueError(f"不支持的加密方法: {method}")
        
        if remove_source:
            delete_file(source)
        
        return dest
    
    except Exception as e:
        raise FileEncryptionError(f"无法解密文件 {source} -> {dest}: {e}")


def split_file(source: Union[str, Path], 
              chunk_size: int = 100 * 1024 * 1024,  # 100MB
              output_dir: Union[str, Path] = None,
              pattern: str = "{name}.part{index:03d}") -> List[Path]:
    """
    分割文件
    
    Args:
        source: 源文件路径
        chunk_size: 块大小（字节）
        output_dir: 输出目录
        pattern: 输出文件模式
        
    Returns:
        List[Path]: 分割后的文件列表
    """
    source = Path(source)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    file_size = source.stat().st_size
    if file_size <= chunk_size:
        # 文件小于块大小，不分割
        return [source]
    
    # 计算块数
    num_chunks = math.ceil(file_size / chunk_size)
    
    # 确定输出目录
    if output_dir is None:
        output_dir = source.parent
    else:
        output_dir = Path(output_dir)
        ensure_directory(output_dir)
    
    chunks = []
    
    try:
        with open(source, 'rb') as f_in:
            for i in range(num_chunks):
                # 生成输出文件名
                output_name = pattern.format(
                    name=source.stem,
                    index=i + 1,
                    total=num_chunks,
                    suffix=source.suffix
                )
                output_path = output_dir / output_name
                
                # 读取块数据
                chunk_data = f_in.read(chunk_size)
                if not chunk_data:
                    break
                
                # 写入块文件
                with open(output_path, 'wb') as f_out:
                    f_out.write(chunk_data)
                
                chunks.append(output_path)
        
        return chunks
    
    except Exception as e:
        # 清理已创建的分块文件
        for chunk in chunks:
            if chunk.exists():
                delete_file(chunk, force=True)
        raise FileOperationError(f"无法分割文件 {source}: {e}")


def merge_files(sources: List[Union[str, Path]], 
               destination: Union[str, Path],
               delete_sources: bool = False) -> Path:
    """
    合并文件
    
    Args:
        sources: 源文件列表
        destination: 目标文件路径
        delete_sources: 合并后是否删除源文件
        
    Returns:
        Path: 目标文件路径
    """
    if not sources:
        raise ValueError("源文件列表为空")
    
    dest = Path(destination)
    
    # 确保目标目录存在
    ensure_parent_directory(dest)
    
    try:
        with open(dest, 'wb') as f_out:
            for source in sources:
                source_path = Path(source)
                if not source_path.exists():
                    raise FileNotFoundError(f"源文件不存在: {source_path}")
                
                with open(source_path, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
        
        # 验证合并文件大小
        total_size = sum(Path(s).stat().st_size for s in sources)
        if dest.stat().st_size != total_size:
            raise FileOperationError(f"合并文件大小不匹配: 期望 {total_size}, 实际 {dest.stat().st_size}")
        
        if delete_sources:
            for source in sources:
                delete_file(source)
        
        return dest
    
    except Exception as e:
        # 清理不完整的合并文件
        if dest.exists():
            delete_file(dest, force=True)
        raise FileOperationError(f"无法合并文件: {e}")


def compare_files(file1: Union[str, Path], 
                 file2: Union[str, Path],
                 method: str = 'hash') -> Dict[str, Any]:
    """
    比较两个文件
    
    Args:
        file1: 文件1路径
        file2: 文件2路径
        method: 比较方法（hash, binary, text）
        
    Returns:
        Dict[str, Any]: 比较结果
    """
    file1 = Path(file1)
    file2 = Path(file2)
    
    if not file1.exists():
        raise FileNotFoundError(f"文件不存在: {file1}")
    if not file2.exists():
        raise FileNotFoundError(f"文件不存在: {file2}")
    
    result = {
        'identical': False,
        'method': method,
        'size1': file1.stat().st_size,
        'size2': file2.stat().st_size,
        'size_equal': file1.stat().st_size == file2.stat().st_size
    }
    
    if not result['size_equal'] and method != 'hash':
        # 大小不同，肯定不同
        return result
    
    try:
        if method == 'hash':
            # 使用哈希比较
            hash1 = calculate_file_hash(file1)
            hash2 = calculate_file_hash(file2)
            result['identical'] = hash1 == hash2
            result['hash1'] = hash1
            result['hash2'] = hash2
        
        elif method == 'binary':
            # 二进制比较
            with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                chunk_size = 8192
                identical = True
                differences = []
                
                chunk_num = 0
                while True:
                    chunk1 = f1.read(chunk_size)
                    chunk2 = f2.read(chunk_size)
                    
                    if not chunk1 and not chunk2:
                        break
                    
                    if chunk1 != chunk2:
                        identical = False
                        # 记录差异位置
                        for i, (b1, b2) in enumerate(zip(chunk1, chunk2)):
                            if b1 != b2:
                                differences.append({
                                    'position': chunk_num * chunk_size + i,
                                    'byte1': b1,
                                    'byte2': b2
                                })
                                if len(differences) >= 10:  # 最多记录10个差异
                                    break
                        
                        if len(differences) >= 10:
                            break
                    
                    chunk_num += 1
                
                result['identical'] = identical
                result['differences'] = differences
        
        elif method == 'text':
            # 文本比较
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                    lines1 = f1.readlines()
                    lines2 = f2.readlines()
                    
                    result['line_count1'] = len(lines1)
                    result['line_count2'] = len(lines2)
                    
                    if len(lines1) != len(lines2):
                        result['identical'] = False
                    else:
                        identical = True
                        differences = []
                        
                        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
                            if line1 != line2:
                                identical = False
                                differences.append({
                                    'line': i + 1,
                                    'line1': line1.rstrip('\n'),
                                    'line2': line2.rstrip('\n')
                                })
                        
                        result['identical'] = identical
                        result['differences'] = differences
        
        else:
            raise ValueError(f"不支持的比较方法: {method}")
        
        return result
    
    except Exception as e:
        raise FileOperationError(f"无法比较文件 {file1} 和 {file2}: {e}")


def create_backup(source: Union[str, Path], 
                 backup_dir: Union[str, Path] = None,
                 max_backups: int = 5,
                 timestamp_format: str = "%Y%m%d_%H%M%S") -> Path:
    """
    创建文件备份
    
    Args:
        source: 源文件路径
        backup_dir: 备份目录
        max_backups: 最大备份数量
        timestamp_format: 时间戳格式
        
    Returns:
        Path: 备份文件路径
    """
    source = Path(source)
    
    if not source.exists():
        raise FileNotFoundError(f"源文件不存在: {source}")
    
    # 确定备份目录
    if backup_dir is None:
        backup_dir = source.parent / 'backups'
    else:
        backup_dir = Path(backup_dir)
    
    ensure_directory(backup_dir)
    
    # 生成备份文件名
    timestamp = datetime.now().strftime(timestamp_format)
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_name
    
    # 复制文件
    backup_path = copy_file(source, backup_path, overwrite=True)
    
    # 清理旧备份
    if max_backups > 0:
        # 获取该文件的所有备份
        backup_pattern = f"{source.stem}_*{source.suffix}"
        backups = sorted(backup_dir.glob(backup_pattern), 
                        key=lambda x: x.stat().st_mtime, 
                        reverse=True)
        
        # 删除超出数量的旧备份
        for old_backup in backups[max_backups:]:
            try:
                old_backup.unlink()
            except:
                pass
    
    return backup_path


def get_file_size_human_readable(size_bytes: int) -> str:
    """
    获取人类可读的文件大小
    
    Args:
        size_bytes: 字节大小
        
    Returns:
        str: 人类可读的大小
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    i = 0
    
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {units[i]}"


def get_directory_size(directory: Union[str, Path], 
                      include_hidden: bool = False) -> int:
    """
    获取目录大小
    
    Args:
        directory: 目录路径
        include_hidden: 是否包含隐藏文件
        
    Returns:
        int: 目录总大小（字节）
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return 0
    
    total_size = 0
    
    try:
        for entry in directory.rglob('*'):
            if not include_hidden and entry.name.startswith('.'):
                continue
            
            if entry.is_file():
                try:
                    total_size += entry.stat().st_size
                except:
                    pass
    
    except Exception as e:
        warnings.warn(f"无法计算目录大小 {directory}: {e}")
    
    return total_size


# 使用示例
if __name__ == "__main__":
    # 创建测试目录
    test_dir = Path("test_file_utils")
    ensure_directory(test_dir)
    
    print(f"测试目录: {test_dir}")
    
    try:
        # 1. 测试文件读写
        print("\n=== 测试文件读写 ===")
        
        # 写入文本文件
        text_file = test_dir / "test.txt"
        write_text_file(text_file, "Hello, World!\nThis is a test file.")
        print(f"写入文本文件: {text_file}")
        
        # 读取文本文件
        text_content = read_text_file(text_file)
        print(f"读取文本文件: {len(text_content)} 字符")
        
        # 写入JSON文件
        json_file = test_dir / "test.json"
        json_data = {"name": "test", "value": 123, "list": [1, 2, 3]}
        write_json_file(json_file, json_data)
        print(f"写入JSON文件: {json_file}")
        
        # 读取JSON文件
        json_content = read_json_file(json_file)
        print(f"读取JSON文件: {json_content}")
        
        # 2. 测试文件信息
        print("\n=== 测试文件信息 ===")
        
        file_info = get_file_info(text_file)
        print(f"文件信息:")
        print(f"  大小: {file_info['size']} 字节")
        print(f"  修改时间: {file_info['modified']}")
        print(f"  格式: {file_info['format']}")
        
        # 3. 测试文件哈希
        print("\n=== 测试文件哈希 ===")
        
        file_hash = calculate_file_hash(text_file, HashAlgorithm.SHA256)
        print(f"文件哈希 (SHA256): {file_hash}")
        
        # 4. 测试文件压缩
        print("\n=== 测试文件压缩 ===")
        
        compressed_file = compress_file(json_file, method=CompressionMethod.GZIP)
        print(f"压缩文件: {compressed_file}")
        print(f"压缩后大小: {compressed_file.stat().st_size} 字节")
        
        # 5. 测试文件解压
        decompressed_file = decompress_file(compressed_file)
        print(f"解压文件: {decompressed_file}")
        
        # 6. 测试文件加密
        print("\n=== 测试文件加密 ===")
        
        encrypted_file = encrypt_file(text_file, method=EncryptionMethod.XOR, key="secret")
        print(f"加密文件: {encrypted_file}")
        
        # 7. 测试文件解密
        decrypted_file = decrypt_file(encrypted_file, method=EncryptionMethod.XOR, key="secret")
        print(f"解密文件: {decrypted_file}")
        
        # 8. 测试文件比较
        print("\n=== 测试文件比较 ===")
        
        comparison = compare_files(text_file, decrypted_file, method='hash')
        print(f"文件比较结果: {comparison['identical']}")
        
        # 9. 测试文件备份
        print("\n=== 测试文件备份 ===")
        
        backup_file = create_backup(json_file, max_backups=3)
        print(f"创建备份: {backup_file}")
        
        # 10. 测试文件查找
        print("\n=== 测试文件查找 ===")
        
        found_files = find_files(test_dir, pattern="*.json")
        print(f"找到JSON文件: {len(found_files)} 个")
        
        # 11. 测试目录大小
        print("\n=== 测试目录大小 ===")
        
        dir_size = get_directory_size(test_dir)
        print(f"目录大小: {get_file_size_human_readable(dir_size)}")
        
        # 12. 测试文件分割和合并
        print("\n=== 测试文件分割和合并 ===")
        
        # 创建一个大文件
        large_file = test_dir / "large.bin"
        with open(large_file, 'wb') as f:
            f.write(os.urandom(1024 * 1024))  # 1MB随机数据
        
        # 分割文件
        chunks = split_file(large_file, chunk_size=200 * 1024)  # 200KB
        print(f"分割为 {len(chunks)} 个块")
        
        # 合并文件
        merged_file = test_dir / "merged.bin"
        merge_files(chunks, merged_file)
        print(f"合并文件: {merged_file}")
        
        # 验证合并
        comparison = compare_files(large_file, merged_file, method='hash')
        print(f"合并验证: {comparison['identical']}")
        
    finally:
        # 清理测试目录
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
            print(f"\n清理测试目录: {test_dir}")