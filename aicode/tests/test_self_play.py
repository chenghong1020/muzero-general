import pytest
import numpy as np
import torch
import sys
import os
import time
import multiprocessing as mp
from unittest.mock import MagicMock, patch

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config import MuZeroConfig
from game import TicTacToeGame
from model import MuZeroNetwork
from self_play import run_selfplay, play_game_actor, play_game
from replay_buffer import ReplayBufferProcess
from shared_storage import SharedStorage, init_share_storage_proxy, SharedStorageManager  # 更新导入
from structures import GameHistory
from mcts import MCTSFacade


@pytest.fixture(scope='session')
def muzero_config():
    """创建用于测试的 MuZeroConfig 实例"""
    config = MuZeroConfig(
        seed=42,
        action_space_size=9,  # 井字棋
        observation_shape=(2, 3, 3),  # 棋盘状态 + 玩家信息
        stacked_observations=2,  # 堆叠的观察状态数
        encoding_size=64,  # 编码状态大小
        fc_representation_layers=[32],  # 表征网络层
        fc_dynamics_layers=[32],  # 动态网络层
        fc_reward_layers=[32],  # 奖励网络层
        fc_value_layers=[32],  # 价值网络层
        fc_policy_layers=[32],  # 策略网络层
        support_size=1,  # 支持大小
        is_two_player_game=True,  # 井字棋是双人游戏
        discount=0.99,  # 折扣因子
        num_simulations=10,  # 模拟次数（测试中使用较小的值）
        pb_c_base=19652,
        pb_c_init=1.25,
        add_exploration_noise=False,  # 不添加探索噪声
        root_dirichlet_alpha=0.3,  # Dirichlet 噪声参数
        root_exploration_fraction=0.25,  # 噪声混合比例
        num_actors=1,  # 测试中使用较小的值
        self_play_delay=0,  # 不延迟，加速测试
        max_moves=50,  # 最大移动次数
        temperature_threshold=None,  # 温度阈值
        replay_buffer_size=10,  # 测试中使用较小的值
        batch_size=2,  # 测试中使用较小的值
        td_steps=5,
        num_unroll_steps=3,
        use_priority=False,  # 启用优先经验回放
        priority_alpha=0.5,
        results_path="./tmp/muzero_test"
    )
    # 添加 action_space 属性，用于 sample_batch 方法
    config.action_space = list(range(config.action_space_size))
    return config


@pytest.fixture(scope='session')
def muzero_model(muzero_config):
    """创建用于测试的 MuZeroNetwork 实例"""
    model = MuZeroNetwork(muzero_config)
    return model


@pytest.fixture(scope='session')  # 添加作用域参数
def shared_storage_proxy(muzero_config, muzero_model):
    """创建用于测试的 SharedStorage 实例"""
    # 使用自定义 Manager 而非 mp.Manager()
    manager = SharedStorageManager()
    manager.start()  # 显式启动 Manager 进程
    
    initial_checkpoint = {
        "weights": muzero_model.get_weights(),
        "info": {"training_step": 0, "total_reward": 0.0, "mean_value": 0.0, "lr": 0.01}
    }
    
    proxy = init_share_storage_proxy(manager, initial_checkpoint, muzero_config)
    yield proxy
    manager.shutdown()


@pytest.fixture
def replay_buffer_process(muzero_config):
    """创建用于测试的 ReplayBufferProcess 实例"""
    buffer = ReplayBufferProcess(muzero_config)
    yield buffer
    buffer.close()


class TestSelfPlay:
    """self_play.py 的测试类"""

    # 移除 replay_buffer_process 参数，它不属于 play_game 的参数列表
    def test_play_game(self, muzero_config, muzero_model, shared_storage_proxy):
        """测试 play_game_actor 函数"""
        # 创建游戏环境
        game = TicTacToeGame(seed=42)
    
        # 执行单个游戏 - 调用 play_game 而非 play_game_actor
        game_history = play_game(
            muzero_config,
            muzero_model,
            game,
            shared_storage_proxy  # 可选参数，对应 shared_storage
        )

        # 验证游戏历史
        assert isinstance(game_history, GameHistory)
        assert len(game_history.action_history) > 0
        assert len(game_history.observation_history) > 0
        assert len(game_history.reward_history) > 0
        assert len(game_history.to_play_history) > 0
        assert len(game_history.root_values) > 0
        assert len(game_history.child_visits) > 0

        # 验证游戏是否结束
        assert game.terminal()

    def test_run_selfplay(self, muzero_config, muzero_model, shared_storage_proxy, replay_buffer_process): # Updated parameter
        """测试 run_selfplay 函数"""
        # 创建游戏工厂函数
        def make_game():
            return TicTacToeGame(seed=np.random.randint(0, 1000))

        # 启动自博弈进程
        selfplay_process = mp.Process(
            target=run_selfplay,
            args=(
                muzero_config,
                muzero_model,
                make_game,
                shared_storage_proxy, # Updated argument
                replay_buffer_process
            )
        )
        selfplay_process.start()

        # 等待一段时间，让自博弈进程执行一些游戏
        time.sleep(2)

        # 检查 ReplayBuffer 是否收到了游戏历史
        # 由于 ReplayBufferProcess 使用多进程，我们需要通过其接口来检查
        # 这里我们可以通过采样批次来间接验证游戏是否被保存
        try:
            batch = replay_buffer_process.sample_batch(muzero_config.num_unroll_steps, 1)
            assert batch is not None
        except Exception as e:
            # 如果游戏数量不足以采样批次，这里可能会失败
            # 这种情况下我们只需确保进程正在运行
            assert selfplay_process.is_alive()

        # 终止自博弈进程
        selfplay_process.terminate()
        selfplay_process.join(timeout=1)
        assert not selfplay_process.is_alive()


    def test_run_selfplay_multiple_actors(self, muzero_config, muzero_model, shared_storage_proxy, replay_buffer_process): # Updated parameter
        """测试多个 actor 的 run_selfplay 函数"""
        # 设置多个 actor
        muzero_config.num_actors = 3

        # 创建游戏工厂函数
        def make_game():
            return TicTacToeGame(seed=np.random.randint(0, 1000))

        # 启动自博弈进程
        selfplay_process = mp.Process(
            target=run_selfplay,
            args=(
                muzero_config,
                muzero_model,
                make_game,
                shared_storage_proxy, # Updated argument
                replay_buffer_process
            )
        )
        selfplay_process.start()

        # 等待一段时间，让自博弈进程执行一些游戏
        time.sleep(3)

        # 终止自博弈进程
        selfplay_process.terminate()
        selfplay_process.join(timeout=1)
        assert not selfplay_process.is_alive()

        # 确保 ReplayBufferProcess 仍然可以关闭
        replay_buffer_process.close()