import numpy as np
import torch
import pytest # 引入 pytest
import sys
import os


# 获取 src 目录的绝对路径
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# 将 src 目录添加到 sys.path 中
sys.path.append(src_path)

# 假设 structures.py 在 src 目录下，并且 tests 和 src 是同级目录
# 需要确保 Python 能够找到 src 目录，可能需要调整 PYTHONPATH 或使用相对导入
from structures import GameHistory, TrainingBatch # 修改为从 src 开始导入

# 定义一个 MockConfig 类，用于测试
class MockConfig:
    def __init__(self):
        self.action_space_size = 3
        self.discount = 0.9
        self.num_unroll_steps = 2
        self.td_steps = 1
        self.stacked_observations = 1 # 保持简单，先用 1
        self.observation_shape = (1, 3, 3) # C, H, W

@pytest.fixture
def mock_config():
    """Pytest fixture 提供 MockConfig 实例"""
    return MockConfig()

@pytest.fixture
def sample_game(mock_config):
    """Pytest fixture 提供一个模拟了几步的 GameHistory 实例"""
    game = GameHistory(config=mock_config)
    game.observation_history.append(np.zeros(mock_config.observation_shape))
    game.to_play_history.append(0)

    # 模拟游戏步骤
    for step in range(5):
        action = np.random.randint(0, mock_config.action_space_size)
        observation = np.random.rand(*mock_config.observation_shape).astype(np.float32) # 确保类型
        reward = float(np.random.choice([0, 1])) # 确保类型
        root_value = float(np.random.rand() * 2 - 1) # 确保类型
        child_visits = {a: np.random.randint(1, 10) for a in range(mock_config.action_space_size)}

        game.action_history.append(action)
        game.observation_history.append(observation)
        game.reward_history.append(reward)
        game.to_play_history.append(step % 2)
        game.store_search_statistics(root_value, child_visits)
    return game

def test_game_history_creation(sample_game, mock_config):
    """测试 GameHistory 是否能正确创建和填充"""
    assert len(sample_game) == 5
    assert len(sample_game.observation_history) == 6 # 初始帧 + 5 步
    assert len(sample_game.reward_history) == 5
    assert len(sample_game.root_values) == 5
    assert len(sample_game.child_visits) == 5
    assert sample_game.observation_history[-1].shape == mock_config.observation_shape

def test_get_stacked_observations_single(sample_game, mock_config):
    """测试单帧堆叠 (num_stacked_observations=1)"""
    mock_config.stacked_observations = 1
    index = 3
    stacked_obs = sample_game.get_stacked_observations(index=index, num_stacked_observations=mock_config.stacked_observations)
    assert stacked_obs.shape == mock_config.observation_shape
    np.testing.assert_array_equal(stacked_obs, sample_game.observation_history[index])

def test_get_stacked_observations_multiple(sample_game, mock_config):
    """测试多帧堆叠"""
    mock_config.stacked_observations = 3 # 例如堆叠 3 帧
    index = 4
    stacked_obs = sample_game.get_stacked_observations(index=index, num_stacked_observations=mock_config.stacked_observations)
    # 预期形状：(stacked * C, H, W)
    expected_shape = (mock_config.stacked_observations * mock_config.observation_shape[0],
                      mock_config.observation_shape[1],
                      mock_config.observation_shape[2])
    assert stacked_obs.shape == expected_shape

def test_get_stacked_observations_at_start(sample_game, mock_config):
    """测试在游戏开始时堆叠，检查零填充"""
    mock_config.stacked_observations = 3
    index = 1 # 需要 obs[1], obs[0], zero_pad
    stacked_obs = sample_game.get_stacked_observations(index=index, num_stacked_observations=mock_config.stacked_observations)
    expected_shape = (mock_config.stacked_observations * mock_config.observation_shape[0],
                      mock_config.observation_shape[1],
                      mock_config.observation_shape[2])
    assert stacked_obs.shape == expected_shape
    # 检查第一部分是否为零
    zero_part_shape = (mock_config.observation_shape[0], mock_config.observation_shape[1], mock_config.observation_shape[2])
    np.testing.assert_array_equal(stacked_obs[:mock_config.observation_shape[0]], np.zeros(zero_part_shape))
    # 检查后两部分是否正确
    np.testing.assert_array_equal(stacked_obs[mock_config.observation_shape[0]:2*mock_config.observation_shape[0]], sample_game.observation_history[0])
    np.testing.assert_array_equal(stacked_obs[2*mock_config.observation_shape[0]:], sample_game.observation_history[1])


def test_make_target(sample_game, mock_config):
    """测试目标生成"""
    state_index = 1
    targets = sample_game.make_target(state_index=state_index,
                                      num_unroll_steps=mock_config.num_unroll_steps,
                                      td_steps=mock_config.td_steps,
                                      discount=mock_config.discount)
    # 预期生成 num_unroll_steps + 1 个目标
    assert len(targets) == mock_config.num_unroll_steps + 1

    # 检查第一个目标的结构和类型
    value, reward, policy = targets[0]
    assert isinstance(value, float)
    assert isinstance(reward, float)
    assert isinstance(policy, dict)
    assert reward == sample_game.reward_history[state_index + 1] # 第一个奖励目标是 state_index+1 的奖励

    # 检查策略字典的值是否加起来接近 1 (允许浮点误差)
    if policy: # 如果 policy 不为空
      assert abs(sum(policy.values()) - 1.0) < 1e-6

def test_training_batch_creation(mock_config):
    """测试 TrainingBatch 数据类是否能正确创建"""
    batch_size = 4
    # 使用 mock_config 中的形状信息
    obs_shape = (batch_size, mock_config.observation_shape[0], mock_config.observation_shape[1], mock_config.observation_shape[2])
    if mock_config.stacked_observations > 1:
         obs_shape = (batch_size, mock_config.stacked_observations * mock_config.observation_shape[0], mock_config.observation_shape[1], mock_config.observation_shape[2])

    obs_batch = torch.randn(*obs_shape)
    act_batch = torch.randint(0, mock_config.action_space_size, (batch_size, mock_config.num_unroll_steps))
    val_target = torch.randn(batch_size, mock_config.num_unroll_steps + 1)
    rew_target = torch.randn(batch_size, mock_config.num_unroll_steps + 1)
    pol_target = torch.randn(batch_size, mock_config.num_unroll_steps + 1, mock_config.action_space_size)
    weight_batch = torch.rand(batch_size)
    mask_batch = torch.ones(batch_size, mock_config.num_unroll_steps + 1)

    batch = TrainingBatch(
        observation_batch=obs_batch,
        action_batch=act_batch,
        target_value=val_target,
        target_reward=rew_target,
        target_policy=pol_target,
        weight_batch=weight_batch, # 测试可选参数
        mask_batch=mask_batch      # 测试可选参数
    )

    assert batch.observation_batch.shape == obs_shape
    assert batch.action_batch.shape == (batch_size, mock_config.num_unroll_steps)
    assert batch.target_value.shape == (batch_size, mock_config.num_unroll_steps + 1)
    assert batch.target_reward.shape == (batch_size, mock_config.num_unroll_steps + 1)
    assert batch.target_policy.shape == (batch_size, mock_config.num_unroll_steps + 1, mock_config.action_space_size)
    assert batch.weight_batch.shape == (batch_size,)
    assert batch.mask_batch.shape == (batch_size, mock_config.num_unroll_steps + 1)


def test_game_history_to_training_batch_conversion(sample_game, mock_config):
    """
    测试从 GameHistory 采样并转换为 TrainingBatch 的核心逻辑 (batch_size=1)。
    模拟 ReplayBuffer.sample_batch 的部分功能。
    """
    # --- 1. 定义采样参数 ---
    state_index = 1 # 假设从游戏历史的第 1 步开始采样 (对应 observation_history[1])
    num_unroll_steps = mock_config.num_unroll_steps
    td_steps = mock_config.td_steps
    discount = mock_config.discount
    num_stacked_observations = mock_config.stacked_observations
    action_space_size = mock_config.action_space_size

    # --- 2. 从 GameHistory 提取/计算数据 ---

    # a. 获取初始堆叠观测
    stacked_obs = sample_game.get_stacked_observations(
        index=state_index,
        num_stacked_observations=num_stacked_observations
    )
    # 转换为 Tensor 并添加 batch 维度
    observation_batch = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)

    # b. 提取动作序列 (注意：动作发生在状态之后)
    # 需要 state_index+1 到 state_index + num_unroll_steps 的动作
    actions = sample_game.action_history[state_index + 1 : state_index + 1 + num_unroll_steps]
    # 填充或截断以确保长度为 num_unroll_steps (如果游戏结束较早)
    actions += [-1] * (num_unroll_steps - len(actions)) # 使用 -1 作为填充动作
    action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(0)

    # c. 计算目标 (价值、奖励、策略)
    targets = sample_game.make_target(
        state_index=state_index,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        discount=discount
    )

    # d. 将目标转换为 Tensor 并添加 batch 维度
    target_values = []
    target_rewards = []
    target_policies = []

    for value, reward, policy_dict in targets:
        target_values.append(value)
        target_rewards.append(reward)

        # 将策略字典转换为 Tensor
        policy_tensor = torch.zeros(action_space_size, dtype=torch.float32)
        if policy_dict: # 检查字典是否为空 (可能在游戏结束时发生)
            total_visits = sum(policy_dict.values())
            if total_visits > 0: # 避免除以零
                 for action, count in policy_dict.items():
                     if 0 <= action < action_space_size: # 确保动作在范围内
                         policy_tensor[action] = count / total_visits
            # else: policy_tensor 保持为零向量
        target_policies.append(policy_tensor)

    target_value_batch = torch.tensor(target_values, dtype=torch.float32).unsqueeze(0)
    target_reward_batch = torch.tensor(target_rewards, dtype=torch.float32).unsqueeze(0)
    target_policy_batch = torch.stack(target_policies).unsqueeze(0)

    # --- 3. 创建 TrainingBatch 实例 ---
    training_batch = TrainingBatch(
        observation_batch=observation_batch,
        action_batch=action_batch,
        target_value=target_value_batch,
        target_reward=target_reward_batch,
        target_policy=target_policy_batch
        # weight_batch 和 mask_batch 可以根据需要添加
    )

    # --- 4. 断言验证 ---
    assert training_batch.observation_batch.shape[0] == 1
    assert training_batch.action_batch.shape == (1, num_unroll_steps)
    assert training_batch.target_value.shape == (1, num_unroll_steps + 1)
    assert training_batch.target_reward.shape == (1, num_unroll_steps + 1)
    assert training_batch.target_policy.shape == (1, num_unroll_steps + 1, action_space_size)

    # 检查第一个目标奖励是否与 history 对应
    assert torch.isclose(training_batch.target_reward[0, 0], torch.tensor(sample_game.reward_history[state_index + 1], dtype=torch.float32))

    # 检查第一个目标策略是否与 history 对应
    first_policy_target = training_batch.target_policy[0, 0]
    first_visits_hist = sample_game.child_visits[state_index]
    total_visits_hist = sum(first_visits_hist.values())
    if total_visits_hist > 0:
        for action, prob in first_visits_hist.items():
             if 0 <= action < action_space_size:
                 assert torch.isclose(first_policy_target[action], torch.tensor(prob / total_visits_hist, dtype=torch.float32))
    else:
         assert torch.sum(first_policy_target) == 0.0 # 如果没有访问，策略应为零