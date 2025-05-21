import os
import time
import threading
import pytest
import torch
import types
from typing import Dict
import numpy as np

import sys
import pathlib

# 将 aicode/src 添加到 Python 路径，以便导入模块
# 注意：根据你的项目结构和运行测试的方式，可能需要调整路径
script_dir = os.path.dirname(__file__)
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
        
    def test_weights_update_simulation(self, storage):
        """模拟训练过程中的权重更新场景"""
        # 初始版本
        initial_version = storage.get_weights_version()
        
        # 模拟训练循环中的多次权重更新
        for i in range(5):
            # 创建新的权重 - 模拟训练后的权重变化
            new_weights = {
                "representation": {
                    "conv1.weight": torch.randn(16, 3, 3, 3),
                    "conv1.bias": torch.randn(16)
                },
                "dynamics": {
                    "fc1.weight": torch.randn(64, 128),
                    "fc1.bias": torch.randn(64)
                },
                "prediction": {
                    "policy_head.weight": torch.randn(9, 64),
                    "policy_head.bias": torch.randn(9),
                    "value_head.weight": torch.randn(1, 64),
                    "value_head.bias": torch.randn(1)
                }
            }
            
            # 更新权重
            storage.set_weights(new_weights)
            
            # 验证版本号递增
            assert storage.get_weights_version() == initial_version + i + 1
            
            # 验证权重已更新且是深拷贝
            current_weights = storage.get_weights()
            assert "conv1.bias" in current_weights["representation"]
            assert "value_head.weight" in current_weights["prediction"]
    
    def test_info_update_simulation(self, storage):
        """模拟训练过程中的统计信息更新场景"""
        # 初始训练步数
        initial_step = storage._checkpoint["info"]["training_step"]
        
        # 模拟训练循环中的多次统计信息更新
        for i in range(1, 6):
            # 创建新的统计信息 - 模拟训练过程中的指标变化
            new_info = {
                "training_step": initial_step + i * 100,
                "total_reward": 10.0 + i * 5.0,
                "mean_value": 0.5 + i * 0.1,
                "lr": 0.01 * (0.9 ** i),  # 模拟学习率衰减
                "policy_loss": 1.5 - i * 0.2,
                "value_loss": 2.0 - i * 0.3,
                "reward_loss": 1.0 - i * 0.1,
                "total_loss": 4.5 - i * 0.6,
                "episode_length": 50 + i * 10,
                "game_win_rate": min(0.1 * i, 1.0)  # 模拟胜率提升
            }
            
            # 更新统计信息
            storage.set_info(new_info)
            
            # 验证统计信息已更新
            current_info = storage.get_info()
            assert current_info["training_step"] == initial_step + i * 100
            assert current_info["total_reward"] == 10.0 + i * 5.0
            assert current_info["game_win_rate"] == min(0.1 * i, 1.0)
            
            # 验证学习率衰减
            assert abs(current_info["lr"] - 0.01 * (0.9 ** i)) < 1e-6
    
    def test_concurrent_weight_access(self, storage):
        """测试并发环境下的权重访问"""
        # 模拟训练线程
        def training_thread():
            for i in range(3):
                # 模拟训练计算
                time.sleep(0.05)
                
                # 更新权重
                new_weights = {
                    "representation": {"layer.weight": torch.randn(10, 10)},
                    "dynamics": {"layer.weight": torch.randn(10, 10)},
                    "prediction": {"layer.weight": torch.randn(10, 10)}
                }
                storage.set_weights(new_weights)
        
        # 模拟自博弈线程
        results = []
        def selfplay_thread():
            for i in range(10):
                # 获取当前权重版本
                version = storage.get_weights_version()
                # 获取权重
                weights = storage.get_weights()
                # 记录结果
                results.append((version, "representation" in weights))
                time.sleep(0.02)
        
        # 启动线程
        train_thread = threading.Thread(target=training_thread)
        play_thread = threading.Thread(target=selfplay_thread)
        
        train_thread.start()
        play_thread.start()
        
        train_thread.join()
        play_thread.join()
        
        # 验证所有访问都成功获取到有效权重
        for version, has_representation in results:
            assert has_representation
        
        # 验证最终版本号正确
        assert storage.get_weights_version() == 3
    
    def test_concurrent_info_access(self, storage):
        """测试并发环境下的统计信息访问"""
        # 初始化训练步数
        storage._checkpoint["info"]["training_step"] = 0
        
        # 模拟训练线程
        def training_thread():
            for i in range(1, 6):
                # 模拟训练计算
                time.sleep(0.05)
                
                # 更新统计信息
                new_info = {
                    "training_step": i * 10,
                    "total_reward": i * 2.5,
                    "mean_value": i * 0.1
                }
                storage.set_info(new_info)
        
        # 模拟监控线程
        steps = []
        def monitor_thread():
            for i in range(15):
                # 获取当前训练步数
                info = storage.get_info()
                steps.append(info.get("training_step", 0))
                time.sleep(0.02)
        
        # 启动线程
        train_thread = threading.Thread(target=training_thread)
        monitor_thread = threading.Thread(target=monitor_thread)
        
        train_thread.start()
        monitor_thread.start()
        
        train_thread.join()
        monitor_thread.join()
        
        # 验证最终训练步数正确
        assert storage.get_info()["training_step"] == 50
        
        # 验证监控线程观察到训练步数的增长
        assert steps[-1] >= steps[0]
        
    def test_realistic_training_scenario(self, storage):
        """测试真实训练场景下的 SharedStorage 行为"""
        # 初始化
        initial_weights = storage.get_weights()
        initial_version = storage.get_weights_version()
        
        # 模拟完整训练循环
        num_iterations = 5
        for i in range(num_iterations):
            # 1. 模拟网络训练和权重更新
            # 创建略微变化的权重 (模拟梯度下降)
            new_representation = {
                k: v + torch.randn_like(v) * 0.01  # 添加小的随机变化
                for k, v in initial_weights["representation"].items()
            }
            new_dynamics = {
                k: v + torch.randn_like(v) * 0.01
                for k, v in initial_weights["dynamics"].items()
            }
            new_prediction = {
                k: v + torch.randn_like(v) * 0.01
                for k, v in initial_weights["prediction"].items()
            }
            
            new_weights = {
                "representation": new_representation,
                "dynamics": new_dynamics,
                "prediction": new_prediction
            }
            
            # 更新权重
            storage.set_weights(new_weights)
            
            # 2. 模拟训练统计信息更新
            # 计算模拟的损失和指标
            policy_loss = 1.5 * (0.9 ** i)  # 模拟损失下降
            value_loss = 2.0 * (0.9 ** i)
            reward_loss = 1.0 * (0.9 ** i)
            total_loss = policy_loss + value_loss + reward_loss
            
            # 模拟奖励增加
            reward_base = 10.0
            reward_improvement = i * 5.0
            total_reward = reward_base + reward_improvement
            
            # 模拟胜率提升
            win_rate = min(0.2 + i * 0.15, 1.0)
            
            # 更新统计信息
            new_info = {
                "training_step": (i + 1) * 1000,
                "total_reward": total_reward,
                "mean_value": 0.5 + i * 0.1,
                "lr": 0.001 * (0.95 ** i),  # 学习率衰减
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "reward_loss": reward_loss,
                "total_loss": total_loss,
                "game_win_rate": win_rate,
                "games_played": (i + 1) * 200,
                "training_time": (i + 1) * 300,  # 秒
                "steps_per_second": 3.3 + i * 0.1  # 模拟训练速度提升
            }
            
            storage.set_info(new_info)
            
        # 验证最终状态
        final_version = storage.get_weights_version()
        final_info = storage.get_info()
        
        # 验证版本号增加了预期的次数
        assert final_version == initial_version + num_iterations
        
        # 验证训练步数正确
        assert final_info["training_step"] == num_iterations * 1000
        
        # 验证损失下降
        assert final_info["total_loss"] < 4.5  # 初始总损失
        
        # 验证奖励增加
        assert final_info["total_reward"] > 10.0  # 初始奖励