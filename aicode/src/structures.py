from typing import List, Optional, Dict, Any
import numpy as np
import torch
from dataclasses import dataclass, field

# 前向声明 MuZeroConfig 以避免循环导入，实际使用时需要导入
# from .config import MuZeroConfig

@dataclass
class GameHistory:
    """
    存储单局游戏的所有相关信息。
    参考 design_structure.md 和 design_interface.md。
    """
    config: Any # MuZeroConfig 类型提示，避免循环依赖
    action_history: List[int] = field(default_factory=list)
    observation_history: List[np.ndarray] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    to_play_history: List[int] = field(default_factory=list)
    child_visits: List[Dict[int, int]] = field(default_factory=list) # MCTS 根节点的子节点访问计数 {action: count}
    root_values: List[float] = field(default_factory=list) # MCTS 根节点的评估价值
    terminated: bool = False
    resigned: bool = False
    # 优先经验回放相关
    priorities: Optional[np.ndarray] = None
    game_priority: Optional[float] = None

    def store_search_statistics(self, root_value: float, child_visits: Dict[int, int]):
        """
        存储 MCTS 搜索后得到的根节点价值和策略（子节点访问次数）。
        参考 design_interface.md 中 GameHistory 的描述。
        """
        self.root_values.append(root_value)
        self.child_visits.append(child_visits)

    def get_stacked_observations(self, index: int, num_stacked_observations: int) -> np.ndarray:
        """
        根据索引获取堆叠的游戏观察状态。
        参考 design_interface.md 中 GameHistory 的描述。
        处理边界情况（游戏开始时）。
        """
        if num_stacked_observations <= 0:
            return self.observation_history[index]

        # 获取当前及之前的观测帧
        observations = [self.observation_history[index]]
        for i in range(1, num_stacked_observations):
            if index - i >= 0:
                observations.insert(0, self.observation_history[index - i])
            else:
                # 在游戏开始时，用零填充或复制第一帧进行填充
                # 这里使用零填充 (假设 observation 是 numpy array)
                zero_obs = np.zeros_like(self.observation_history[0])
                observations.insert(0, zero_obs)

        # 堆叠观测帧 (通常在通道维度)
        # 假设 observation_shape 是 (C, H, W) 或 (H, W, C)
        # 如果是 (H, W, C)，需要调整 axis
        # 假设是 (C, H, W)，则堆叠后是 (num_stacked * C, H, W)
        # 如果原始是 (H, W, C)，堆叠后是 (H, W, num_stacked * C)
        # 这里假设是 (C, H, W)
        # 注意：实际堆叠方式取决于网络输入要求
        if len(observations[0].shape) == 3 and observations[0].shape[0] < observations[0].shape[1]: # (C, H, W)
            stacked_obs = np.concatenate(observations, axis=0)
        elif len(observations[0].shape) == 3: # (H, W, C)
            stacked_obs = np.concatenate(observations, axis=2)
        else: # 其他情况，例如 1D 或 2D 状态
            stacked_obs = np.stack(observations, axis=0) # 简单堆叠

        return stacked_obs

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, discount: float):
        """
        为训练计算目标价值、奖励和策略。
        参考 MuZero 论文 Appendix G Training。
        """
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                # 计算 n-step 价值
                value = self.root_values[bootstrap_index] * (discount ** td_steps)
                for i, reward in enumerate(self.reward_history[current_index + 1 : bootstrap_index + 1]):
                    value += reward * (discount ** i)
            else:
                # 如果超出游戏长度，使用最终奖励（如果游戏结束）或 0
                value = 0.0
                if current_index < len(self.root_values): # 确保当前索引有效
                    # 累加到游戏结束的奖励
                    for i, reward in enumerate(self.reward_history[current_index + 1:]):
                         value += reward * (discount ** i)
                # 注意：如果游戏未结束但 bootstrap_index 超出，理论上不应发生（除非游戏很短）
                # 这里的处理方式可能需要根据具体情况调整，例如使用最后一步的价值

            # 奖励目标：下一步的实际奖励
            if current_index < len(self.reward_history) -1:
                reward = self.reward_history[current_index + 1]
            else:
                reward = 0.0 # 超出游戏长度

            # 策略目标：MCTS 搜索得到的策略分布
            if current_index < len(self.child_visits):
                visits = self.child_visits[current_index]
                total_visits = sum(visits.values())
                policy = {action: count / total_visits for action, count in visits.items()}
            else:
                # 超出游戏长度，使用均匀分布或零向量
                policy = {} # 或者根据 action_space_size 创建均匀分布

            targets.append((value, reward, policy))
        return targets

    def __len__(self):
        # 游戏历史的长度定义为采取的动作数量
        return len(self.action_history)


@dataclass
class TrainingBatch:
    """
    封装用于训练网络的一个批次数据。
    参考 design_structure.md 中 ReplayBuffer 部分的数据模型。
    """
    observation_batch: torch.Tensor  # [batch_size, C, H, W] or [batch_size, stack*C, H, W]
    action_batch: torch.Tensor       # [batch_size, num_unroll_steps] (注意：这里与设计文档略有不同，通常动作是 K 步)
    target_value: torch.Tensor       # [batch_size, num_unroll_steps + 1]
    target_reward: torch.Tensor      # [batch_size, num_unroll_steps + 1]
    target_policy: torch.Tensor      # [batch_size, num_unroll_steps + 1, action_space_size]
    # 可选：优先经验回放的权重
    weight_batch: Optional[torch.Tensor] = None # [batch_size]
    # 可选：梯度缩放因子 (MuZero 论文 Appendix G)
    gradient_scale_batch: Optional[torch.Tensor] = None # [batch_size, num_unroll_steps + 1]
    # 可选：用于掩码损失的有效步数标记
    mask_batch: Optional[torch.Tensor] = None # [batch_size, num_unroll_steps + 1]
