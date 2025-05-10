import pytest
import torch
import numpy as np
import os
import sys

# 将 aicode/src 添加到 Python 路径，以便导入模块
# 注意：根据你的项目结构和运行测试的方式，可能需要调整路径
script_dir = os.path.dirname(__file__)
# 假设 tests 目录与 aicode 目录同级
project_root = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcts import Node, MinMaxStats, select_child, ucb_score, backpropagate
from config import MuZeroConfig

class TestNode:
    def test_init(self):
        """测试节点初始化"""
        prior = 0.5
        to_play = 1
        node = Node(prior=prior, to_play=to_play)
        
        assert node.prior == prior
        assert node.to_play == to_play
        assert node.visit_count == 0
        assert node.value_sum == 0
        assert len(node.children) == 0
        assert node.hidden_state is None
        assert node.reward == 0

    def test_expanded(self):
        """测试节点扩展状态检查"""
        node = Node(prior=0.5, to_play=1)
        assert not node.expanded()
        
        # 添加一个子节点
        node.children[0] = Node(prior=0.3, to_play=-1)
        assert node.expanded()

    def test_value(self):
        """测试节点价值计算"""
        node = Node(prior=0.5, to_play=1)
        assert node.value() == 0  # 未访问时价值为0
        
        node.value_sum = 10
        node.visit_count = 5
        assert node.value() == 2  # 价值应该是总和除以访问次数

class TestMinMaxStats:
    def test_init(self):
        """测试MinMaxStats初始化"""
        stats = MinMaxStats()
        assert stats.maximum == -float("inf")
        assert stats.minimum == float("inf")

    def test_update(self):
        """测试更新最大最小值"""
        stats = MinMaxStats()
        stats.update(1.0)
        assert stats.maximum == 1.0
        assert stats.minimum == 1.0
        
        stats.update(2.0)
        assert stats.maximum == 2.0
        assert stats.minimum == 1.0
        
        stats.update(0.5)
        assert stats.maximum == 2.0
        assert stats.minimum == 0.5

    def test_normalize(self):
        """测试值归一化"""
        stats = MinMaxStats()
        
        # 当min=max时应该返回原值
        value = 1.0
        assert stats.normalize(value) == value
        
        # 更新范围后测试归一化
        stats.update(0.0)
        stats.update(2.0)
        assert stats.normalize(1.0) == 0.5  # (1.0 - 0.0) / (2.0 - 0.0) = 0.5

def test_ucb_score():
    """测试UCB分数计算"""
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=False,
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None  # 这会使 visit_softmax_temperature_fn 始终返回 1.0
    )
    
    parent = Node(prior=1.0, to_play=1)
    child = Node(prior=0.5, to_play=-1)
    parent.visit_count = 10
    child.visit_count = 5
    child.reward = 0.5
    child.value_sum = 5.0
    
    min_max_stats = MinMaxStats()
    min_max_stats.update(0.0)
    min_max_stats.update(1.0)
    
    score = ucb_score(config, parent, child, min_max_stats)
    assert isinstance(score, float)
    assert score > 0
    
    # 测试先验分数部分
    child_prior_test = Node(prior=0.8, to_play=-1)  # 创建新节点，更高的先验概率
    child_prior_test.visit_count = child.visit_count  # 保持相同的访问次数
    child_prior_test.reward = child.reward  # 保持相同的奖励
    child_prior_test.value_sum = child.value_sum  # 保持相同的价值总和
    
    score_high_prior = ucb_score(config, parent, child_prior_test, min_max_stats)
    assert score_high_prior > score  # 更高的先验概率应该得到更高的分数

    # 单独测试访问次数对先验项的影响
    child_unvisited = Node(prior=child.prior, to_play=-1)  # 相同的先验概率
    child_unvisited.visit_count = 0
    child_unvisited.reward = child.reward
    child_unvisited.value_sum = 0  # 未访问节点的价值总和为0
    
    score_unvisited = ucb_score(config, parent, child_unvisited, min_max_stats)
    assert score_unvisited > score  # 未访问节点应该有更高的探索倾向
    
    # 测试价值分数部分
    child.visit_count = 5
    child.reward = 1.0  # 更高的奖励
    score_high_reward = ucb_score(config, parent, child, min_max_stats)
    assert score_high_reward > score  # 更高奖励应该得到更高的分数
    
    # 测试访问次数的影响
    child.visit_count = 20  # 增加访问次数
    score_more_visits = ucb_score(config, parent, child, min_max_stats)
    assert score_more_visits < score_high_reward  # 访问次数增加应该降低探索倾向

def test_select_child():
    """测试子节点选择"""
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=True,
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None
    )
    
    node = Node(prior=1.0, to_play=1)
    min_max_stats = MinMaxStats()
    
    # 添加三个子节点，模拟不同的价值和访问情况
    node.children[0] = Node(prior=0.5, to_play=-1)  # 高先验，低访问次数
    node.children[0].visit_count = 2
    node.children[0].reward = 0.3
    node.children[0].value_sum = 1.0
    
    node.children[1] = Node(prior=0.3, to_play=-1)  # 中等先验，高价值
    node.children[1].visit_count = 5
    node.children[1].reward = 0.8
    node.children[1].value_sum = 5.0
    
    node.children[2] = Node(prior=0.2, to_play=-1)  # 低先验，高访问次数
    node.children[2].visit_count = 10
    node.children[2].reward = 0.2
    node.children[2].value_sum = 3.0
    
    # 设置父节点的访问次数
    node.visit_count = 17  # 等于所有子节点访问次数之和
    
    # 更新最大最小值统计
    for child in node.children.values():
        min_max_stats.update(child.reward + config.discount * child.value())
    
    action, child = select_child(config, node, min_max_stats)
    
    # 验证选择的节点是否具有最高的UCB分数
    selected_ucb = ucb_score(config, node, child, min_max_stats)
    for a in node.children.keys():
        if a != action:
            other_ucb = ucb_score(config, node, node.children[a], min_max_stats)
            assert selected_ucb >= other_ucb, f"选择的节点UCB分数({selected_ucb})应该大于等于其他节点({other_ucb})"

# 新增测试用例 - 测试 PUCT 常数计算
def test_pb_c_calculation():
    """测试 PUCT 常数 (pb_c) 的计算"""
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=True,
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None
    )
    
    # 创建父节点和子节点
    parent = Node(prior=1.0, to_play=1)
    child = Node(prior=0.5, to_play=-1)
    
    # 设置不同的访问次数组合
    test_cases = [
        {"parent_visits": 10, "child_visits": 0},
        {"parent_visits": 10, "child_visits": 1},
        {"parent_visits": 100, "child_visits": 10},
        {"parent_visits": 1000, "child_visits": 100}
    ]
    
    for case in test_cases:
        parent.visit_count = case["parent_visits"]
        child.visit_count = case["child_visits"]
        
        # 手动计算 pb_c
        # pb_c = (log((N(p) + C_base + 1) / C_base) + C_init) * sqrt(N(p)) / (N(c) + 1)
        expected_pb_c = (np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init) * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        # 从 ucb_score 函数中提取 pb_c 计算部分
        # 注意：这里假设 ucb_score 函数内部有一个计算 pb_c 的部分
        # 如果 ucb_score 函数没有直接暴露 pb_c 计算，可以通过比较最终的 UCB 分数来间接验证
        
        # 创建 MinMaxStats 用于 ucb_score 函数
        min_max_stats = MinMaxStats()
        min_max_stats.update(0.0)
        min_max_stats.update(1.0)
        
        # 计算 UCB 分数
        score = ucb_score(config, parent, child, min_max_stats)
        
        # 手动计算先验分数部分
        prior_score = expected_pb_c * child.prior
        
        # 手动计算价值分数部分
        child_value = child.value()
        value_score = min_max_stats.normalize(child.reward + config.discount * (-child_value if config.is_two_player_game else child_value))
        
        # 验证总分数是否等于先验分数加价值分数
        expected_score = prior_score + value_score
        assert abs(score - expected_score) < 1e-6, f"UCB 分数计算错误，期望 {expected_score}，实际 {score}"
        
        # 打印测试信息，便于调试
        print(f"父节点访问次数: {parent.visit_count}, 子节点访问次数: {child.visit_count}")
        print(f"计算的 pb_c: {expected_pb_c}")
        print(f"先验分数: {prior_score}, 价值分数: {value_score}, 总分数: {expected_score}")

# 测试先验分数计算
def test_prior_score():
    """测试先验分数 (Prior Score) 的计算"""
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=True,
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None
    )
    
    # 创建父节点
    parent = Node(prior=1.0, to_play=1)
    parent.visit_count = 50
    
    # 测试不同先验概率的子节点
    prior_values = [0.1, 0.3, 0.5, 0.8]
    
    for prior in prior_values:
        # 创建子节点
        child = Node(prior=prior, to_play=-1)
        child.visit_count = 5
        child.reward = 0.0
        child.value_sum = 0.0
        
        # 创建 MinMaxStats
        min_max_stats = MinMaxStats()
        min_max_stats.update(0.0)
        min_max_stats.update(1.0)
        
        # 计算 UCB 分数
        score = ucb_score(config, parent, child, min_max_stats)
        
        # 手动计算 pb_c
        pb_c = (np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init) * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        # 手动计算先验分数
        expected_prior_score = pb_c * prior
        
        # 手动计算价值分数
        value_score = min_max_stats.normalize(child.reward + config.discount * (-child.value() if config.is_two_player_game else child.value()))
        
        # 验证总分数是否等于先验分数加价值分数
        expected_score = expected_prior_score + value_score
        assert abs(score - expected_score) < 1e-6, f"UCB 分数计算错误，期望 {expected_score}，实际 {score}"
        
        # 验证先验概率更高的节点得分更高
        if prior > 0.1:
            child_low_prior = Node(prior=0.1, to_play=-1)
            child_low_prior.visit_count = child.visit_count
            child_low_prior.reward = child.reward
            child_low_prior.value_sum = child.value_sum
            
            score_low_prior = ucb_score(config, parent, child_low_prior, min_max_stats)
            assert score > score_low_prior, f"先验概率为 {prior} 的节点应该比先验概率为 0.1 的节点得分更高"

# 测试价值分数计算 - 单人游戏
def test_value_score_single_player():
    """测试单人游戏的价值分数 (Value Score) 计算"""
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=False,  # 单人游戏
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None
    )
    
    # 创建父节点和子节点
    parent = Node(prior=1.0, to_play=1)
    child = Node(prior=0.5, to_play=1)
    
    parent.visit_count = 10
    child.visit_count = 5
    
    # 测试不同的奖励和价值组合
    test_cases = [
        {"reward": 0.0, "value_sum": 0.0},
        {"reward": 0.5, "value_sum": 0.0},
        {"reward": 0.0, "value_sum": 5.0},
        {"reward": 0.5, "value_sum": 5.0}
    ]
    
    for case in test_cases:
        child.reward = case["reward"]
        child.value_sum = case["value_sum"]
        
        # 创建 MinMaxStats
        min_max_stats = MinMaxStats()
        min_max_stats.update(0.0)
        min_max_stats.update(1.0)
        
        # 计算 UCB 分数
        score = ucb_score(config, parent, child, min_max_stats)
        
        # 手动计算 pb_c
        pb_c = (np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init) * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        # 手动计算先验分数
        prior_score = pb_c * child.prior
        
        # 手动计算价值分数 - 单人游戏
        # ValueScore(c) = normalize(R(c) + γ * V(c))
        child_value = child.value()
        raw_value_score = child.reward + config.discount * child_value
        value_score = min_max_stats.normalize(raw_value_score)
        
        # 验证总分数是否等于先验分数加价值分数
        expected_score = prior_score + value_score
        assert abs(score - expected_score) < 1e-6, f"UCB 分数计算错误，期望 {expected_score}，实际 {score}"
        
        # 验证奖励或价值更高的节点得分更高
        if case["reward"] > 0 or case["value_sum"] > 0:
            child_zero = Node(prior=child.prior, to_play=child.to_play)
            child_zero.visit_count = child.visit_count
            child_zero.reward = 0.0
            child_zero.value_sum = 0.0
            
            score_zero = ucb_score(config, parent, child_zero, min_max_stats)
            assert score > score_zero, f"奖励为 {child.reward} 价值为 {child.value()} 的节点应该比零奖励零价值的节点得分更高"

# 测试价值分数计算 - 双人游戏
def test_value_score_two_player():
    """测试双人游戏的价值分数 (Value Score) 计算"""
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=True,  # 双人游戏
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None
    )
    
    # 创建父节点和子节点
    parent = Node(prior=1.0, to_play=1)
    child = Node(prior=0.5, to_play=-1)  # 对手节点
    
    parent.visit_count = 10
    child.visit_count = 5
    
    # 测试不同的奖励和价值组合
    test_cases = [
        {"reward": 0.0, "value_sum": 0.0},
        {"reward": 0.5, "value_sum": 0.0},
        {"reward": 0.0, "value_sum": 5.0},
        {"reward": 0.5, "value_sum": 5.0}
    ]
    
    for case in test_cases:
        child.reward = case["reward"]
        child.value_sum = case["value_sum"]
        
        # 创建 MinMaxStats
        min_max_stats = MinMaxStats()
        min_max_stats.update(0.0)
        min_max_stats.update(1.0)
        
        # 计算 UCB 分数
        score = ucb_score(config, parent, child, min_max_stats)
        
        # 手动计算 pb_c
        pb_c = (np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init) * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        # 手动计算先验分数
        prior_score = pb_c * child.prior
        
        # 手动计算价值分数 - 双人游戏
        # ValueScore(c) = normalize(R(c) + γ * (-V(c)))
        child_value = child.value()
        raw_value_score = child.reward + config.discount * (-child_value)
        value_score = min_max_stats.normalize(raw_value_score)
        
        # 验证总分数是否等于先验分数加价值分数
        expected_score = prior_score + value_score
        assert abs(score - expected_score) < 1e-6, f"UCB 分数计算错误，期望 {expected_score}，实际 {score}"

# 测试节点价值计算
def test_node_value():
    """测试节点价值 V(s) 的计算"""
    # 创建节点
    node = Node(prior=0.5, to_play=1)
    
    # 测试未访问节点
    assert node.value() == 0, "未访问节点的价值应该为0"
    
    # 测试不同的访问次数和价值总和组合
    test_cases = [
        {"visit_count": 1, "value_sum": 1.0},
        {"visit_count": 5, "value_sum": 10.0},
        {"visit_count": 10, "value_sum": -5.0},
        {"visit_count": 100, "value_sum": 200.0}
    ]
    
    for case in test_cases:
        node.visit_count = case["visit_count"]
        node.value_sum = case["value_sum"]
        
        # 手动计算期望价值
        expected_value = case["value_sum"] / case["visit_count"]
        
        # 验证节点价值计算是否正确
        assert node.value() == expected_value, f"节点价值计算错误，期望 {expected_value}，实际 {node.value()}"

# 测试价值回溯 - 单人游戏
def test_backpropagate_single_player():
    """测试单人游戏的价值回溯"""
    # 创建配置
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=False,
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None
    )
    # 创建搜索路径
    search_path = [
        Node(prior=1.0, to_play=1),    # 根节点
        Node(prior=0.5, to_play=1),    # 中间节点
        Node(prior=0.3, to_play=1)     # 叶子节点
    ]
    
    # 设置测试参数
    value = 1.0        # 叶子节点的预测价值
    to_play = 1        # 当前玩家
    discount = 0.9     # 折扣因子
    min_max_stats = MinMaxStats()
    
    # 设置节点的即时奖励
    search_path[0].reward = 0.0    # 根节点通常没有奖励
    search_path[1].reward = 0.5    # 中间节点有一些奖励
    search_path[2].reward = 1.0    # 叶子节点获得最终奖励
    
    # 执行反向传播
    backpropagate(config, search_path, value, to_play, discount, min_max_stats)
    
    # 验证反向传播的结果
    # 1. 验证访问次数更新
    for node in search_path:
        assert node.visit_count == 1, "每个节点的访问次数应该增加1"

#     根据MCTS_formula.md中的单人游戏价值更新公式：

# 1. 更新节点的累积价值: Q_sum_new(n) = Q_sum_old(n) + G_from_child
# 2. 递归计算回溯价值: G_for_parent(n) = R(n) + γ * G_from_child
# 从叶子节点开始，向上反向传播：

# ### 叶子节点 (search_path[2])
# - 访问次数增加： visit_count = 1
# - 累积价值更新： value_sum = 0 + 1.0 = 1.0
# - MinMax统计更新： value_for_minmax = 1.0 + 0.9 * 1.0 = 1.9
# - 传递给父节点的值： value = 1.0 + 0.9 * 1.0 = 1.9
# ### 中间节点 (search_path[1])
# - 访问次数增加： visit_count = 1
# - 累积价值更新： value_sum = 0 + 1.9 = 1.9
# - MinMax统计更新： value_for_minmax = 0.5 + 0.9 * 1.9 = 2.21
# - 传递给父节点的值： value = 0.5 + 0.9 * 1.9 = 2.21
# ### 根节点 (search_path[0])
# - 访问次数增加： visit_count = 1
# - 累积价值更新： value_sum = 0 + 2.21 = 2.21
# - MinMax统计更新： value_for_minmax = 0.0 + 0.9 * 2.21 = 1.989
    
    # 2. 验证价值更新（考虑折扣）
    # 叶子节点：直接使用预测价值
    assert search_path[2].value_sum == value, f"叶子节点价值错误，期望 {value}，实际 {search_path[2].value_sum}"
    
    # 中间节点：G_for_parent = R(n) + γ * G_from_child
    expected_middle_value = 1.9
    assert abs(search_path[1].value_sum - expected_middle_value) < 1e-6, f"中间节点价值错误，期望 {expected_middle_value}，实际 {search_path[1].value_sum}"
    
    # 根节点：G_for_parent = R(n) + γ * G_from_child
    expected_root_value = 2.21
    assert abs(search_path[0].value_sum - expected_root_value) < 1e-6, f"根节点价值错误，期望 {expected_root_value}，实际 {search_path[0].value_sum}"
    
    # 3. 验证最大最小值统计更新
    max_value = 2.21
    min_value = 1.9
    assert min_max_stats.maximum >= max_value, f"最大值统计错误，期望 >= {max_value}，实际 {min_max_stats.maximum}"
    assert min_max_stats.minimum <= min_value, f"最小值统计错误，期望 <= {min_value}，实际 {min_max_stats.minimum}"

# 测试价值回溯 - 双人游戏
def test_backpropagate_two_player():
    """测试双人游戏的价值回溯"""
    # 创建配置
    config = MuZeroConfig(
        action_space_size=9,
        discount=0.99,
        root_dirichlet_alpha=0.3,
        num_simulations=50,
        batch_size=512,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        is_two_player_game=True,
        pb_c_base=19652,
        pb_c_init=1.25,
        temperature_threshold=None
    )

    # 创建搜索路径
    search_path = [
        Node(prior=1.0, to_play=1),    # 根节点(红方)
        Node(prior=0.5, to_play=-1),   # 中间节点(黑方)
        Node(prior=0.3, to_play=1)     # 叶子节点(红方)
    ]
    
    # 设置测试参数
    value = 1.0        # 叶子节点的预测价值
    to_play = 1        # 当前玩家（红方）
    discount = 0.9     # 折扣因子
    min_max_stats = MinMaxStats()
    
    # 设置节点的即时奖励
    search_path[0].reward = 0.0    # 根节点通常没有奖励
    search_path[1].reward = 0.5    # 中间节点有一些奖励
    search_path[2].reward = 1.0    # 叶子节点获得最终奖励
    
    # 执行反向传播
    backpropagate(config, search_path, value, to_play, discount, min_max_stats)
    
    # 验证反向传播的结果
    # 1. 验证访问次数更新
    for node in search_path:
        assert node.visit_count == 1, "每个节点的访问次数应该增加1"

#     ### 叶子节点 (search_path[2], to_play=1)
# 1. 访问次数增加： visit_count = 1
# 2. 累积价值更新：
#    - 当前玩家与叶子节点玩家相同 (to_play=1, node.to_play=1)
#    - value_sum = 0 + 1.0 = 1.0
# 3. MinMax 统计更新：
#    - value_for_minmax = 1.0 + 0.9 * (-1.0) = 1.0 - 0.9 = 0.1
# 4. 传递给父节点的值：
#    - 当前玩家与叶子节点玩家相同，奖励取反
#    - value = -1.0 + 0.9 * 1.0 = -1.0 + 0.9 = -0.1
# ### 中间节点 (search_path[1], to_play=-1)
# 1. 访问次数增加： visit_count = 1
# 2. 累积价值更新：
#    - 当前玩家与叶子节点玩家不同 (to_play=1, node.to_play=-1)
#    - value_sum = 0 + (-(-0.1)) = 0 + 0.1 = 0.1
# 3. MinMax 统计更新：
#    - value_for_minmax = 0.5 + 0.9 * (-0.1) = 0.5 - 0.09 = 0.41
# 4. 传递给父节点的值：
#    - 当前玩家与叶子节点玩家不同，奖励保持不变
#    - value = 0.5 + 0.9 * (-0.1) = 0.5 - 0.09 = 0.41
# ### 根节点 (search_path[0], to_play=1)
# 1. 访问次数增加： visit_count = 1
# 2. 累积价值更新：
#    - 当前玩家与叶子节点玩家相同 (to_play=1, node.to_play=1)
#    - value_sum = 0 + 0.41 = 0.41
# 3. MinMax 统计更新：
#    - value_for_minmax = 0.0 + 0.9 * (-0.41) = 0 - 0.369 = -0.369
    
    # 2. 验证价值更新（考虑玩家视角和折扣）
    # 叶子节点(红方)：直接使用预测价值，因为与当前玩家相同
    assert search_path[2].value_sum == 1.0, f"叶子节点价值错误，期望 {value}，实际 {search_path[2].value_sum}"
    
    # 中间节点(黑方)：需要反转价值，并考虑自身奖励
    # G_for_parent = R(n) + γ * G_from_child，但因为是对手视角，所以 G_from_child 取负
    expected_middle_value = 0.1
    assert abs(search_path[1].value_sum - expected_middle_value) < 1e-6, f"中间节点价值错误，期望 {expected_middle_value}，实际 {search_path[1].value_sum}"
    
    # 根节点(红方)：再次反转价值，考虑折扣和奖励
    # G_for_parent = R(n) + γ * G_from_child，但因为是自己视角，所以 G_from_child 取负
    expected_root_value = 0.41
    assert abs(search_path[0].value_sum - expected_root_value) < 1e-6, f"根节点价值错误，期望 {expected_root_value}，实际 {search_path[0].value_sum}"
    
    # 3. 验证最大最小值统计更新
    # 在双人游戏中，MinMaxStats 
    max_tracked = 0.41
    min_tracked = -0.369
    assert min_max_stats.maximum >= max_tracked, f"最大值统计错误，期望 >= {max_tracked}，实际 {min_max_stats.maximum}"
    assert min_max_stats.minimum <= min_tracked, f"最小值统计错误，期望 <= {min_tracked}，实际 {min_max_stats.minimum}"

# 测试 MinMax 统计更新
def test_minmax_stats_update():
    """测试 MinMax 统计更新"""
    min_max_stats = MinMaxStats()
    
    # 初始状态
    assert min_max_stats.maximum == -float("inf"), "初始最大值应该是负无穷"
    assert min_max_stats.minimum == float("inf"), "初始最小值应该是正无穷"
    
    # 更新单个值
    min_max_stats.update(0.5)
    assert min_max_stats.maximum == 0.5, "更新后最大值应该是0.5"
    assert min_max_stats.minimum == 0.5, "更新后最小值应该是0.5"
    
    # 更新更大的值
    min_max_stats.update(1.0)
    assert min_max_stats.maximum == 1.0, "更新后最大值应该是1.0"
    assert min_max_stats.minimum == 0.5, "最小值应该保持不变"
    
    # 更新更小的值
    min_max_stats.update(0.0)
    assert min_max_stats.maximum == 1.0, "最大值应该保持不变"
    assert min_max_stats.minimum == 0.0, "更新后最小值应该是0.0"
    
    # 更新负值
    min_max_stats.update(-1.0)
    assert min_max_stats.maximum == 1.0, "最大值应该保持不变"
    assert min_max_stats.minimum == -1.0, "更新后最小值应该是-1.0"

# 测试值归一化
def test_normalize_value():
    """测试值归一化"""
    min_max_stats = MinMaxStats()
    
    # 当 min=max 时应该返回原值
    value = 1.0
    assert min_max_stats.normalize(value) == value, "当min=max时应该返回原值"
    
    # 更新范围后测试归一化
    min_max_stats.update(-1.0)
    min_max_stats.update(1.0)
    
    # 测试边界值
    assert min_max_stats.normalize(-1.0) == 0.0, "最小值应该归一化为0.0"
    assert min_max_stats.normalize(1.0) == 1.0, "最大值应该归一化为1.0"
    
    # 测试中间值
    assert min_max_stats.normalize(0.0) == 0.5, "中间值应该归一化为0.5"
    assert min_max_stats.normalize(-0.5) == 0.25, "归一化计算错误"
    assert min_max_stats.normalize(0.5) == 0.75, "归一化计算错误"