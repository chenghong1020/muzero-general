import copy
import os
import threading
import time
from typing import Dict, List, Any, Union, Optional
import types

import torch

from config import MuZeroConfig


class SharedStorage:
    """
    SharedStorage 是 MuZero 算法中的关键组件，负责存储和同步最新的网络权重及训练统计信息。
    它作为训练器和自博弈 Actor 之间的桥梁，确保所有组件能够访问到最新的模型和训练数据。
    """

    def __init__(self, checkpoint: Dict, config: MuZeroConfig):
        """
        初始化共享存储。
        
        参数:
            checkpoint: 包含初始网络权重和统计信息的字典
            config: MuZero配置对象
        """
        self.config = config
        self._checkpoint = copy.deepcopy(checkpoint)
        self._model_version = 0
        self._lock = threading.RLock()  # 使用可重入锁保证线程安全
        
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        保存当前检查点到指定路径。
        
        参数:
            path: 保存路径，默认为配置中的结果路径
        
        返回:
            保存的文件路径
        """
        if path is None:
            path = os.path.join(self.config.results_path, "model.checkpoint")
        
        with self._lock:
            torch.save(self._checkpoint, path)
        
        return path
    
    def get_checkpoint(self) -> Dict:
        """
        获取完整的检查点数据（只读视图）。
        
        返回:
            包含所有权重和统计信息的检查点字典的只读视图
        
        注意:
            返回的是只读视图，尝试修改会引发TypeError异常
        """
        with self._lock:
            return types.MappingProxyType(self._checkpoint)
    
    def get_weights(self) -> Dict:
        """
        获取最新的网络权重（只读视图）。
        
        返回: 
            最新的网络权重的只读视图
            
        注意: 
            返回的是只读视图，尝试修改会引发TypeError异常
        """
        with self._lock:
            return types.MappingProxyType(self._checkpoint["weights"])
    
    def validate_weights_structure(self, weights: Dict) -> bool:
        """
        验证权重结构是否符合规范。
        
        参数:
            weights: 要验证的权重字典
            
        返回:
            如果权重结构有效则返回 True，否则返回 False
        """
        required_keys = ["representation", "dynamics", "prediction"]
        return all(key in weights for key in required_keys)
    
    def set_weights(self, weights: Dict) -> None:
        """
        更新网络权重。
        
        参数:
            weights: 新的网络权重
            
        异常:
            ValueError: 当权重结构不符合规范时抛出
        """
        if not self.validate_weights_structure(weights):
            raise ValueError("权重结构无效，必须包含 'representation', 'dynamics', 'prediction' 键")
            
        with self._lock:
            self._checkpoint["weights"] = copy.deepcopy(weights)
            self._model_version += 1
    
    def get_weights_version(self) -> int:
        """
        获取当前权重的版本号。
        
        返回:
            权重版本号
        """
        with self._lock:
            return self._model_version
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取所有统计信息（只读视图）。
        
        返回:
            包含所有统计信息的字典的只读视图
            
        注意:
            返回的是只读视图，尝试修改会引发TypeError异常
        """
        with self._lock:
            if "info" not in self._checkpoint:
                self._checkpoint["info"] = {}
            return types.MappingProxyType(self._checkpoint["info"])
    
    def set_info(self, info: Dict[str, Any]) -> None:
        """
        更新所有统计信息。
        
        参数:
            info: 包含所有统计信息的字典
        """
        with self._lock:
            self._checkpoint["info"] = copy.deepcopy(info)

    def atomic_update_info(self, update_fn: Callable[[Dict], Dict]) -> None:
        """
        原子化更新统计信息
        
        参数:
            update_fn: 接收当前info字典，返回更新后的字典
        """
        with self._lock:
            current_info = copy.deepcopy(self._checkpoint.get("info", {}))
            updated_info = update_fn(current_info)
            self._checkpoint["info"] = updated_info
    
    def wait_for_training_step(self, target_step: int, timeout: Optional[float] = None) -> int:
        """
        等待直到训练步数达到目标值。
        
        用于自博弈Actor等待新模型时的同步。
        
        参数:
            target_step: 目标训练步数
            timeout: 超时时间（秒），None表示无限等待
            
        返回:
            达到目标步数时的训练步数
        """
        start_time = time.time()
        while True:
            with self._lock:
                current_step = self._checkpoint["info"].get("training_step", 0)
                if current_step >= target_step:
                    return current_step
            
            # 检查是否超时
            if timeout is not None and time.time() - start_time > timeout:
                with self._lock:
                    return self._checkpoint["info"].get("training_step", 0)
            
            # 短暂休眠，避免忙等待
            time.sleep(0.1)