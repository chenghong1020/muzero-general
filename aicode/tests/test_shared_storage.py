import os
import time
import threading
import pytest
import torch
import types
from typing import Dict

import sys
import pathlib

# 假设 tests 目录与 aicode 目录同级
project_root = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MuZeroConfig
from shared_storage import SharedStorage


class TestSharedStorage:
    """SharedStorage 类的单元测试"""

    @pytest.fixture
    def config(self):
        """创建测试用的配置对象"""
        config = MuZeroConfig(
            action_space_size=9,
            observation_shape=(3, 3, 1),
            results_path="/tmp/muzero_test"
        )
        return config

    @pytest.fixture
    def checkpoint(self):
        """创建测试用的检查点数据"""
        # 创建一个模拟的网络权重
        representation_weights = {"conv1.weight": torch.randn(16, 3, 3, 3)}
        dynamics_weights = {"fc1.weight": torch.randn(64, 128)}
        prediction_weights = {"policy_head.weight": torch.randn(9, 64)}
        
        weights = {
            "representation": representation_weights,
            "dynamics": dynamics_weights,
            "prediction": prediction_weights
        }
        
        # 创建一个模拟的训练信息
        info = {
            "training_step": 0,
            "total_reward": 0.0,
            "mean_value": 0.0,
            "lr": 0.01
        }
        
        return {"weights": weights, "info": info}

    @pytest.fixture
    def storage(self, config, checkpoint):
        """创建 SharedStorage 实例"""
        return SharedStorage(checkpoint, config)

    def test_init(self, storage, config, checkpoint):
        """测试初始化"""
        assert storage.config == config
        assert storage._model_version == 0
        assert isinstance(storage._lock, threading.RLock)
        
        # 检查检查点是否是深拷贝
        assert storage._checkpoint is not checkpoint
        assert storage._checkpoint != {}

    def test_save_checkpoint(self, storage, tmpdir):
        """测试保存检查点"""
        # 使用临时目录
        path = os.path.join(tmpdir, "test_checkpoint.pt")
        saved_path = storage.save_checkpoint(path)
        
        assert saved_path == path
        assert os.path.exists(path)
        
        # 验证保存的内容
        loaded = torch.load(path)
        assert "weights" in loaded
        assert "info" in loaded

    def test_get_checkpoint(self, storage):
        """测试获取检查点"""
        checkpoint = storage.get_checkpoint()
        
        # 验证返回的是只读视图
        assert isinstance(checkpoint, types.MappingProxyType)
        
        # 验证内容
        assert "weights" in checkpoint
        assert "info" in checkpoint
        
        # 验证只读性质
        with pytest.raises(TypeError):
            checkpoint["new_key"] = "value"

    def test_get_weights(self, storage):
        """测试获取权重"""
        weights = storage.get_weights()
        
        # 验证返回的是只读视图
        assert isinstance(weights, types.MappingProxyType)
        
        # 验证内容
        assert "representation" in weights
        assert "dynamics" in weights
        assert "prediction" in weights
        
        # 验证只读性质
        with pytest.raises(TypeError):
            weights["new_key"] = "value"

    def test_validate_weights_structure(self, storage):
        """测试权重结构验证"""
        # 有效的权重结构
        valid_weights = {
            "representation": {},
            "dynamics": {},
            "prediction": {}
        }
        assert storage.validate_weights_structure(valid_weights) is True
        
        # 无效的权重结构 - 缺少键
        invalid_weights1 = {
            "representation": {},
            "dynamics": {}
        }
        assert storage.validate_weights_structure(invalid_weights1) is False
        
        # 无效的权重结构 - 额外的键不影响验证
        valid_weights2 = {
            "representation": {},
            "dynamics": {},
            "prediction": {},
            "extra": {}
        }
        assert storage.validate_weights_structure(valid_weights2) is True

    def test_set_weights(self, storage):
        """测试设置权重"""
        # 准备新的权重
        new_weights = {
            "representation": {"new_layer.weight": torch.randn(10, 10)},
            "dynamics": {"new_layer.weight": torch.randn(10, 10)},
            "prediction": {"new_layer.weight": torch.randn(10, 10)}
        }
        
        # 设置权重
        storage.set_weights(new_weights)
        
        # 验证权重已更新
        assert storage._checkpoint["weights"] is not new_weights  # 应该是深拷贝
        assert "new_layer.weight" in storage._checkpoint["weights"]["representation"]
        
        # 验证版本号已增加
        assert storage._model_version == 1
        
        # 测试无效权重结构
        invalid_weights = {
            "representation": {},
            "invalid_key": {}
        }
        
        with pytest.raises(ValueError):
            storage.set_weights(invalid_weights)

    def test_get_weights_version(self, storage):
        """测试获取权重版本号"""
        # 初始版本号
        assert storage.get_weights_version() == 0
        
        # 更新权重后的版本号
        new_weights = {
            "representation": {},
            "dynamics": {},
            "prediction": {}
        }
        storage.set_weights(new_weights)
        assert storage.get_weights_version() == 1

    def test_get_info(self, storage):
        """测试获取统计信息"""
        info = storage.get_info()
        
        # 验证返回的是只读视图
        assert isinstance(info, types.MappingProxyType)
        
        # 验证内容
        assert "training_step" in info
        assert "total_reward" in info
        
        # 验证只读性质
        with pytest.raises(TypeError):
            info["new_key"] = "value"
            
        # 测试 info 不存在的情况
        storage._checkpoint.pop("info")
        info = storage.get_info()
        assert isinstance(info, types.MappingProxyType)
        assert len(info) == 0

    def test_set_info(self, storage):
        """测试设置统计信息"""
        # 准备新的统计信息
        new_info = {
            "training_step": 100,
            "total_reward": 10.5,
            "new_metric": "value"
        }
        
        # 设置统计信息
        storage.set_info(new_info)
        
        # 验证统计信息已更新
        assert storage._checkpoint["info"] is not new_info  # 应该是深拷贝
        assert storage._checkpoint["info"]["training_step"] == 100
        assert storage._checkpoint["info"]["new_metric"] == "value"

    def test_wait_for_training_step(self, storage):
        """测试等待训练步数功能"""
        # 设置初始训练步数
        storage._checkpoint["info"]["training_step"] = 5
        
        # 测试已达到目标步数的情况
        result = storage.wait_for_training_step(5)
        assert result == 5
        
        # 测试超时的情况
        start_time = time.time()
        result = storage.wait_for_training_step(10, timeout=0.5)
        elapsed_time = time.time() - start_time
        assert result == 5  # 未达到目标步数，返回当前步数
        assert elapsed_time >= 0.5  # 确保等待了超时时间
        
        # 测试在等待过程中达到目标步数的情况
        def update_step():
            time.sleep(0.2)
            storage._checkpoint["info"]["training_step"] = 20
            
        thread = threading.Thread(target=update_step)
        thread.start()
        
        start_time = time.time()
        result = storage.wait_for_training_step(15, timeout=1.0)
        elapsed_time = time.time() - start_time
        
        assert result == 20
        assert 0.2 <= elapsed_time < 1.0  # 确保在超时前返回
        
        thread.join()