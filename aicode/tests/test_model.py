import pytest
import torch
import sys
import os
import numpy as np

# 将 aicode/src 添加到 Python 路径，以便导入模块
# 注意：根据你的项目结构和运行测试的方式，可能需要调整路径
script_dir = os.path.dirname(__file__)
# 假设 tests 目录与 aicode 目录同级
project_root = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入，如果失败则从上一级目录导入 (适应不同的运行环境)
try:
    from model import MuZeroNetwork
    from config import MuZeroConfig
    from utils import support_to_scalar, scalar_to_support
except ImportError:
    # 如果直接运行 pytest tests/test_model_pytest.py，可能需要这样导入
    from model import MuZeroNetwork
    from config import MuZeroConfig
    from utils import support_to_scalar, scalar_to_support


@pytest.fixture(scope="module") # 使用 module scope，fixture 只为整个测试模块运行一次
def muzero_config():
    """Pytest fixture for MuZeroConfig."""
    return MuZeroConfig(
        action_space_size=9,         # e.g., TicTacToe
        observation_shape=(2, 3, 3), # (C, H, W)
        encoding_size=32,
        fc_representation_layers=[64],
        fc_dynamics_layers=[64],
        fc_reward_layers=[32],
        fc_value_layers=[32],
        fc_policy_layers=[32],
        support_size=5,              # 价值/奖励范围 [-5, 5]
        stacked_observations=0       # 井字棋不需要堆叠
    )

@pytest.fixture(scope="module")
def device():
    """Pytest fixture for determining the device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def muzero_model(muzero_config, device):
    """Pytest fixture for MuZeroNetwork model."""
    model = MuZeroNetwork(muzero_config)
    model.to(device)
    model.eval() # 设置为评估模式，关闭 dropout 等
    return model

@pytest.fixture
def batch_size():
    """Pytest fixture for batch size."""
    return 4

@pytest.fixture
def initial_observation(muzero_config, batch_size, device):
    """Pytest fixture for initial observation tensor."""
    obs_shape = muzero_config.observation_shape
    return torch.randn(batch_size, obs_shape[0], obs_shape[1], obs_shape[2]).to(device)

def test_initial_inference(muzero_model, muzero_config, initial_observation, batch_size, device):
    """测试 initial_inference 方法"""
    with torch.no_grad(): # 测试时不需要计算梯度
        initial_output = muzero_model.initial_inference(initial_observation)

    # 检查输出字典的键
    expected_keys = {"value", "value_logits", "reward", "reward_logits", "policy_logits", "encoded_state"}
    assert set(initial_output.keys()) == expected_keys

    # 检查输出张量的形状
    full_support_size = 2 * muzero_config.support_size + 1 if muzero_config.support_size > 0 else 1
    assert initial_output['encoded_state'].shape == (batch_size, muzero_config.encoding_size)
    assert initial_output['policy_logits'].shape == (batch_size, muzero_config.action_space_size)
    assert initial_output['value'].shape == (batch_size, 1)
    assert initial_output['value_logits'].shape == (batch_size, full_support_size)
    assert initial_output['reward'].shape == (batch_size, 1)
    assert initial_output['reward_logits'].shape == (batch_size, full_support_size)

    # 检查初始奖励是否为 0
    assert torch.allclose(initial_output['reward'], torch.zeros_like(initial_output['reward']))
    # 检查初始奖励 logits 是否对应于 0
    expected_reward_logits = scalar_to_support(torch.zeros_like(initial_output['reward']), muzero_config.support_size)
    assert torch.allclose(initial_output['reward_logits'], expected_reward_logits)

def test_recurrent_inference(muzero_model, muzero_config, initial_observation, batch_size, device):
    """测试 recurrent_inference 方法"""
    with torch.no_grad():
        initial_output = muzero_model.initial_inference(initial_observation)
        current_encoded_state = initial_output['encoded_state']

        # 随机选择一个动作
        action = torch.randint(0, muzero_config.action_space_size, (batch_size, 1)).to(device)

        recurrent_output = muzero_model.recurrent_inference(current_encoded_state, action)

    # 检查输出字典的键
    expected_keys = {"value", "value_logits", "reward", "reward_logits", "policy_logits", "encoded_state"}
    assert set(recurrent_output.keys()) == expected_keys

    # 检查输出张量的形状
    full_support_size = 2 * muzero_config.support_size + 1 if muzero_config.support_size > 0 else 1
    assert recurrent_output['encoded_state'].shape == (batch_size, muzero_config.encoding_size)
    assert recurrent_output['policy_logits'].shape == (batch_size, muzero_config.action_space_size)
    assert recurrent_output['value'].shape == (batch_size, 1)
    assert recurrent_output['value_logits'].shape == (batch_size, full_support_size)
    assert recurrent_output['reward'].shape == (batch_size, 1)
    assert recurrent_output['reward_logits'].shape == (batch_size, full_support_size)

def test_support_conversion(muzero_config, batch_size, device):
    """测试 support_to_scalar 和 scalar_to_support 的转换"""
    support_size = muzero_config.support_size
    full_support_size = 2 * support_size + 1 if support_size > 0 else 1

    # --- 1. 测试 scalar_to_support ---
    # 测试值是否符合公式
    target_scalar = torch.tensor([[1.5], [-2.3], [3], [0.0], [float(support_size)]]).to(device) # 包含边界值
    target_support_dist = scalar_to_support(target_scalar, support_size)

    # 验证 1.5 的转换
    assert torch.allclose(target_support_dist[0, support_size + 1], torch.tensor(0.5).to(device))
    assert torch.allclose(target_support_dist[0, support_size + 2], torch.tensor(0.5).to(device))
    
    # 验证 -2.3 的转换
    assert torch.allclose(target_support_dist[1, support_size - 3], torch.tensor(0.3).to(device))
    assert torch.allclose(target_support_dist[1, support_size - 2], torch.tensor(0.7).to(device))
    
    # 验证整数 3 的转换
    assert torch.allclose(target_support_dist[2, support_size + 3], torch.tensor(1.0).to(device))
    # 验证其他位置是否为 0
    mask = torch.ones(full_support_size).to(device)
    mask[support_size + 3] = 0
    assert torch.allclose(target_support_dist[2, mask.bool()], torch.zeros(int(mask.sum())).to(device))
    
    # 验证 0.0 的转换 (已有测试)
    assert torch.allclose(target_support_dist[3, support_size], torch.tensor(1.0).to(device))
    
    # 验证 support_size 的转换
    assert torch.allclose(target_support_dist[4, -1], torch.tensor(1.0).to(device))


    # 创建目标标量值
    target_scalar = torch.tensor([[1.5], [-2.3], [0.0], [float(support_size)]]).to(device) # 包含边界值
    target_support_dist = scalar_to_support(target_scalar, support_size)

    # 检查输出形状
    assert target_support_dist.shape == (batch_size, full_support_size)
    # 检查是否为有效的概率分布 (非负且总和为 1)
    assert torch.all(target_support_dist >= 0)
    assert torch.allclose(torch.sum(target_support_dist, dim=1), torch.ones(batch_size).to(device), atol=1e-6)

    
    # 检查特殊情况：输入 0.0 应该在索引 support_size 处产生概率 1.0
    zero_scalar = torch.tensor([[0.0]]).to(device)
    zero_dist = scalar_to_support(zero_scalar, support_size)
    if support_size > 0:
        expected_zero_dist = torch.zeros(1, full_support_size).to(device)
        expected_zero_dist[0, support_size] = 1.0
        assert torch.allclose(zero_dist, expected_zero_dist)

    # --- 2. 测试 support_to_scalar ---
    if support_size > 0: # 仅当使用离散支持时测试
        support = torch.linspace(-support_size, support_size, full_support_size).to(device)
        support = support.view(1, -1) # 形状 (1, full_support_size)

        # 示例 1: Logits 集中在中心 (0)
        logits1 = torch.zeros(1, full_support_size).to(device)
        logits1[0, support_size] = 10.0 # 给中心点一个很高的 logit
        probabilities1 = torch.softmax(logits1, dim=1)
        expected_scalar1 = torch.sum(support * probabilities1, dim=1, keepdim=True)
        calculated_scalar1 = support_to_scalar(logits1, support_size)
        # 期望值应该非常接近 0
        assert torch.allclose(calculated_scalar1, expected_scalar1, atol=1e-5)
        assert torch.allclose(calculated_scalar1, torch.tensor([[0.0]]).to(device), atol=1e-3)

        # 示例 2: Logits 集中在最大值 (support_size)
        logits2 = torch.zeros(1, full_support_size).to(device)
        logits2[0, -1] = 10.0 # 给最大支撑点一个很高的 logit
        probabilities2 = torch.softmax(logits2, dim=1)
        expected_scalar2 = torch.sum(support * probabilities2, dim=1, keepdim=True)
        calculated_scalar2 = support_to_scalar(logits2, support_size)
        # 期望值应该非常接近 support_size
        assert torch.allclose(calculated_scalar2, expected_scalar2, atol=1e-5)
        assert torch.allclose(calculated_scalar2, torch.tensor([[float(support_size)]]).to(device), atol=1e-2)

        # 示例 3: 随机 Logits (使用 batch_size)
        logits3 = torch.randn(batch_size, full_support_size).to(device)
        probabilities3 = torch.softmax(logits3, dim=1)
        # 手动计算期望值: support shape (1, F) * probs shape (B, F) -> sum(dim=1) -> (B, 1)
        expected_scalar3 = torch.sum(support * probabilities3, dim=1, keepdim=True)
        calculated_scalar3 = support_to_scalar(logits3, support_size)
        # 检查函数计算结果是否与手动计算一致
        assert calculated_scalar3.shape == (batch_size, 1) # 确保输出形状正确
        assert torch.allclose(calculated_scalar3, expected_scalar3, atol=1e-5)

    # --- 3. 测试 support_size = 0 的情况 ---
    # support_to_scalar 应该直接返回输入（保持形状）
    logits_input_zero = torch.randn(batch_size, 1).to(device)
    scalar_output_zero = support_to_scalar(logits_input_zero, 0)
    # 检查形状应该保持 (batch_size, 1)
    assert scalar_output_zero.shape == (batch_size, 1)
    assert torch.allclose(scalar_output_zero, logits_input_zero)