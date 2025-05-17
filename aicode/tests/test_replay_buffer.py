import pytest
import numpy as np
import torch
import sys
import os
import multiprocessing as mp
from unittest.mock import MagicMock, patch
import time

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config import MuZeroConfig
from structures import GameHistory, TrainingBatch
from replay_buffer import ReplayBuffer


@pytest.fixture
def mock_config():
    """创建用于测试的 MuZeroConfig 实例"""
    config = MuZeroConfig(
        seed=42,
        action_space_size=9,  # 假设是井字棋
        observation_shape=(3, 3, 2),  # 棋盘状态 + 玩家信息
        stacked_observations=2,
        replay_buffer_size=10,
        batch_size=2,
        td_steps=5,
        num_unroll_steps=3,
        discount=0.99,
        use_priority=True,  # 启用优先经验回放（原来是PER=True）
        priority_alpha=0.5  # 原来是PER_alpha=0.5
    )
    # 添加 action_space 属性，用于 sample_batch 方法
    config.action_space = list(range(config.action_space_size))
    return config


@pytest.fixture
def sample_game_history(mock_config):
    """创建一个样本游戏历史"""
    game = GameHistory(config=mock_config)
    
    # 添加观察、动作、奖励等历史数据
    # 假设是一个简单的 3x3x2 的观察空间
    for i in range(5):  # 创建一个有 5 步的游戏
        # 观察：随机生成的 3x3x2 数组
        obs = np.random.rand(3, 3, 2).astype(np.float32)
        game.observation_history.append(obs)
        
        # 动作：0-8 之间的随机值
        if i < 4:  # 最后一步没有动作
            action = np.random.randint(0, 9)
            game.action_history.append(action)
        
        # 奖励：-1 到 1 之间的随机值
        if i > 0:  # 第一步没有奖励
            reward = np.random.uniform(-1, 1)
            game.reward_history.append(reward)
        
        # 轮到的玩家：1 或 -1
        player = 1 if i % 2 == 0 else -1
        game.to_play_history.append(player)
        
        # MCTS 根节点价值：-1 到 1 之间的随机值
        root_value = np.random.uniform(-1, 1)
        
        # MCTS 子节点访问次数：随机生成
        child_visits = {}
        total_visits = np.random.randint(10, 100)
        for a in range(9):
            visits = np.random.randint(0, total_visits // 2)
            if visits > 0:
                child_visits[a] = visits
                total_visits -= visits
        
        # 确保至少有一个动作有访问次数
        if not child_visits:
            child_visits[np.random.randint(0, 9)] = 1
            
        # 存储搜索统计信息
        game.store_search_statistics(root_value, child_visits)
    
    return game


@pytest.fixture
def initialized_replay_buffer(mock_config):
    """初始化一个 ReplayBuffer 实例"""
    # 创建 ReplayBuffer 实例
    buffer = ReplayBuffer(mock_config)
    buffer.start()  # 启动进程

    yield buffer

    # 清理
    buffer.close()
    return buffer


class TestReplayBuffer:
    """ReplayBuffer 类的测试"""
    
    def test_initialization(self, mock_config):
        """测试 ReplayBuffer 初始化"""
        buffer = ReplayBuffer(mock_config)
        
        assert buffer.config == mock_config
        assert buffer.buffer == {}
        assert buffer.num_played_games == 0
        assert buffer.num_played_steps == 0
        assert buffer.total_samples == 0
    
    def test_save_game(self, mock_config, sample_game_history):
        """测试保存游戏历史"""
        buffer = ReplayBuffer(mock_config)
        
        # 模拟 SharedStorage
        mock_shared_storage = MagicMock()
        mock_shared_storage.set_info = MagicMock()
        
        # 保存游戏
        buffer._save_game(sample_game_history)
        
        # 验证游戏已保存
        assert len(buffer.buffer) == 1
        assert buffer.num_played_games == 1
        assert buffer.num_played_steps == len(sample_game_history.action_history)
        assert buffer.total_samples == len(sample_game_history.root_values)
        
        # 验证优先级计算
        assert sample_game_history.priorities is not None
        assert sample_game_history.game_priority is not None
        assert len(sample_game_history.priorities) == len(sample_game_history.root_values)
    
    def test_save_multiple_games_and_buffer_size_limit(self, mock_config, sample_game_history):
        """测试保存多个游戏并验证缓冲区大小限制"""
        buffer = ReplayBuffer(mock_config)
        
        # 保存超过缓冲区大小限制的游戏
        for _ in range(mock_config.replay_buffer_size + 5):
            # 创建游戏的深拷贝，避免引用同一个对象
            game_copy = GameHistory(config=mock_config)
            game_copy.observation_history = sample_game_history.observation_history.copy()
            game_copy.action_history = sample_game_history.action_history.copy()
            game_copy.reward_history = sample_game_history.reward_history.copy()
            game_copy.to_play_history = sample_game_history.to_play_history.copy()
            game_copy.root_values = sample_game_history.root_values.copy()
            game_copy.child_visits = sample_game_history.child_visits.copy()
            
            buffer._save_game(game_copy)
        
        # 验证缓冲区大小不超过限制
        assert len(buffer.buffer) == mock_config.replay_buffer_size
        assert buffer.num_played_games == mock_config.replay_buffer_size + 5
    
    def test_compute_target_value(self, mock_config, sample_game_history):
        """测试目标值计算"""
        buffer = ReplayBuffer(mock_config)
        
        # 计算目标值
        index = 1  # 选择一个中间索引
        target_value = buffer._compute_target_value(sample_game_history, index)
         
        # 验证计算结果
        assert target_value != 0.0 
    
    def test_sample_position(self, mock_config, sample_game_history):
        """测试位置采样"""
        buffer = ReplayBuffer(mock_config)
        
        # 计算优先级
        buffer._save_game(sample_game_history)
        
        # 采样位置
        position, prob = buffer._sample_position(sample_game_history)
        
        # 验证结果
        assert 0 <= position < len(sample_game_history.root_values)
        assert 0 < prob <= 1
        
        # 测试强制均匀采样
        position_uniform, prob_uniform = buffer._sample_position(sample_game_history, force_uniform=True)
        assert 0 <= position_uniform < len(sample_game_history.root_values)
        assert prob_uniform == 1.0 / len(sample_game_history.root_values)
    
    def test_sample_n_games(self, mock_config):
        """测试游戏采样"""
        buffer = ReplayBuffer(mock_config)
        
        # 添加多个游戏
        for i in range(5):
            game = GameHistory(config=mock_config)
            # 添加一些基本数据
            game.root_values = [0.5] * 3
            game.action_history = [0, 1, 2]
            game.priorities = np.array([1.0, 2.0, 3.0])
            game.game_priority = 3.0
            
            buffer._save_game(game)
        
        # 采样游戏
        n_games = 3
        sampled_games = buffer._sample_n_games(n_games)
        
        # 验证结果
        assert len(sampled_games) == n_games
        for game_id, game_history, game_prob in sampled_games:
            assert game_id in buffer.buffer
            assert game_history is buffer.buffer[game_id]
            assert 0 < game_prob <= 1
        
        # 测试强制均匀采样
        sampled_games_uniform = buffer._sample_n_games(n_games, force_uniform=True)
        assert len(sampled_games_uniform) == n_games
    
    def test_sample_batch(self, mock_config):
        """测试批次采样"""
        buffer = ReplayBuffer(mock_config)
        
        # 添加多个游戏
        for i in range(5):
            game = GameHistory(config=mock_config)
            
            # 添加观察历史
            for j in range(10):
                game.observation_history.append(np.random.rand(3, 3, 2).astype(np.float32))
            
            # 添加动作历史
            game.action_history = [np.random.randint(0, 9) for _ in range(9)]
            
            # 添加奖励历史
            game.reward_history = [np.random.uniform(-1, 1) for _ in range(9)]
            
            # 添加根节点价值
            game.root_values = [np.random.uniform(-1, 1) for _ in range(9)]
            
            # 添加子节点访问次数
            for j in range(9):
                visits = {}
                for a in range(9):
                    if np.random.random() > 0.5:
                        visits[a] = np.random.randint(1, 10)
                if not visits:  # 确保至少有一个动作
                    visits[0] = 1
                game.child_visits.append(visits)
            
            buffer._save_game(game)
        
        # 采样批次
        batch_size = 2
        num_unroll_steps = 3
        batch = buffer._sample_batch(num_unroll_steps, batch_size)
        
        # 验证批次结构
        assert isinstance(batch, TrainingBatch)
        assert batch.observation_batch.shape[0] == batch_size
        assert batch.action_batch.shape == (batch_size, num_unroll_steps)
        assert batch.target_value.shape == (batch_size, num_unroll_steps + 1)
        assert batch.target_reward.shape == (batch_size, num_unroll_steps + 1)
        assert batch.target_policy.shape == (batch_size, num_unroll_steps + 1, mock_config.action_space_size)
        
        # 验证数据类型
        assert batch.observation_batch.dtype == torch.float32
        assert batch.action_batch.dtype == torch.int64
        assert batch.target_value.dtype == torch.float32
        assert batch.target_reward.dtype == torch.float32
        assert batch.target_policy.dtype == torch.float32
        
        # 如果启用了优先经验回放，验证权重批次
        if mock_config.use_priority:  # 原来是mock_config.PER
            assert batch.weight_batch is not None
            assert batch.weight_batch.shape[0] == batch_size
            assert batch.weight_batch.dtype == torch.float32
    
    def test_multithreading_safety(self, mock_config, sample_game_history):
        """测试多线程环境下 ReplayBuffer 的线程安全性"""
        import threading
        
        buffer = ReplayBuffer(mock_config)
        
        # 创建多个游戏历史对象
        game_histories = []
        for i in range(10):
            # 创建游戏的深拷贝，避免引用同一个对象
            game_copy = GameHistory(config=mock_config)
            game_copy.observation_history = sample_game_history.observation_history.copy()
            game_copy.action_history = sample_game_history.action_history.copy()
            game_copy.reward_history = sample_game_history.reward_history.copy()
            game_copy.to_play_history = sample_game_history.to_play_history.copy()
            game_copy.root_values = sample_game_history.root_values.copy()
            game_copy.child_visits = sample_game_history.child_visits.copy()
            game_histories.append(game_copy)
        
        # 定义线程函数：保存游戏
        def save_games_thread():
            for game in game_histories[:5]:
                buffer.save_game(game)
        
        # 定义线程函数：采样批次
        def sample_batch_thread():
            # 等待一些游戏被保存
            time.sleep(0.2)
            
            batch = buffer.sample_batch(mock_config.num_unroll_steps, 2)
            assert isinstance(batch, TrainingBatch)
            assert batch.observation_batch.shape[0] == 2
        
        # 创建并启动线程
        threads = []
        for _ in range(2):  # 创建多个保存游戏的线程
            t = threading.Thread(target=save_games_thread)
            threads.append(t)
            t.start()
        
        # 创建采样批次的线程
        t = threading.Thread(target=sample_batch_thread)
        threads.append(t)
        t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证结果
        assert buffer.num_played_games > 0
        assert len(buffer.buffer) > 0
        
        # 关闭线程池
        buffer.close()
    
    
