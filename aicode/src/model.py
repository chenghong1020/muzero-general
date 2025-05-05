import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict # <-- Add Dict here
import numpy as np

# 假设 config 和 utils 在同一目录下或已添加到 sys.path
from config import MuZeroConfig
from utils import support_to_scalar, scalar_to_support, mlp

class MuZeroNetwork(nn.Module):
    """
    MuZero 神经网络模型，包含表征、动态和预测三个核心网络。
    参考 design_structure.md 和 design_interface.md。
    """
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        self.full_support_size = 2 * config.support_size + 1 if config.support_size > 0 else 1

        # --- 1. 表征网络 (Representation Network) h ---
        # 输入: 堆叠的观察状态 (stacked_observations)
        # 输出: 编码状态 (encoded_state)
        # 根据 design_structure.md 中的描述，这里使用 MLP 作为示例
        # 输入维度需要根据 observation_shape 和 stacked_observations 计算
        obs_shape = config.observation_shape # (C, H, W) or (H, W, C) etc.
        obs_channels = obs_shape[0] if len(obs_shape) == 3 else 1 # 假设通道在第一维
        # 扁平化后的输入大小
        # 注意：这里假设观察是图像类数据，需要扁平化。如果观察是一维向量，则不需要 * obs_shape[1] * obs_shape[2]
        # 同时考虑堆叠帧数
        # TODO: 需要根据实际 observation_shape 调整输入大小计算方式
        if len(obs_shape) == 3: # 图像类 (C, H, W)
            # 观测通道数 C' = (k + 1) * C
            # 动作编码为 k 个 1 x H x W 的平面
            total_channels = (config.stacked_observations + 1) * obs_channels + config.stacked_observations
            representation_input_size = total_channels * obs_shape[1] * obs_shape[2]
        elif len(obs_shape) == 1: # 一维向量
            # 观测特征维度
            obs_dim = (config.stacked_observations + 1) * obs_shape[0]
            # 动作编码维度 (k个one-hot向量)
            action_dim = config.stacked_observations * config.action_space_size if config.stacked_observations > 0 else 0
            # 总输入维度
            representation_input_size = obs_dim + action_dim
        else: # 其他情况，需要用户定义
             raise ValueError(f"Unsupported observation shape format: {obs_shape}")

        # 使用 utils 中的 mlp 构建器，移除 DataParallel
        self.representation_network = mlp(
            representation_input_size,
            config.fc_representation_layers,
            config.encoding_size
        )

        self.representation_norm = nn.LayerNorm(config.encoding_size, eps=1e-5)

        # --- 2. 动态网络 (Dynamics Network) g ---
        # 输入: 编码状态 (encoded_state) + 动作 (action)
        # 输出: 下一编码状态 (next_encoded_state) + 奖励 (reward)
        # 动作需要进行 one-hot 编码
        dynamics_input_size = config.encoding_size + config.action_space_size
        
        self.dynamics_state_network = mlp(
                dynamics_input_size,
                config.fc_dynamics_layers,
                config.encoding_size
            )
        # 奖励预测输出维度是 full_support_size
        self.dynamics_reward_network = mlp(
                config.encoding_size, # 论文中奖励只依赖于状态 s'
                config.fc_reward_layers,
                self.full_support_size
            )

        # 动态网络的 LayerNorm
        self.dynamics_norm = nn.LayerNorm(config.encoding_size, eps=1e-5)
        
        # --- 3. 预测网络 (Prediction Network) f ---
        # 输入: 编码状态 (encoded_state)
        # 输出: 策略 (policy_logits) + 价值 (value)
        self.prediction_policy_network = mlp(
                config.encoding_size,
                config.fc_policy_layers,
                config.action_space_size
            )

        # 价值预测输出维度是 full_support_size
        self.prediction_value_network = mlp(
                config.encoding_size,
                config.fc_value_layers,
                self.full_support_size
            )

    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        """
        表征函数 h: 将观察映射到编码状态。
        Args:
            observation: 堆叠的观察状态，形状 (batch_size, C', H, W) 或 (batch_size, D')
                         其中 C' = (stacked_observations + 1) * C
                         D' = (stacked_observations + 1) * D
        Returns:
            编码状态，形状 (batch_size, encoding_size)
        """
        # 将输入扁平化以适应 MLP
        batch_size = observation.shape[0]
        flat_observation = observation.view(batch_size, -1)
        encoded_state = self.representation_network(flat_observation)
        
        # 直接使用已创建的 LayerNorm
        normalized_state = self.representation_norm(encoded_state)
        return normalized_state


    def dynamics(self, encoded_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        动态函数 g: 预测下一状态和奖励。
        Args:
            encoded_state: 当前编码状态，形状 (batch_size, encoding_size)
            action: 当前采取的动作 (one-hot 编码)，形状 (batch_size, action_space_size)
        Returns:
            next_encoded_state: 下一编码状态，形状 (batch_size, encoding_size)
            reward_logits: 预测的奖励 logits，形状 (batch_size, full_support_size)
        """
        # 将状态和动作拼接
        state_action = torch.cat((encoded_state, action), dim=1)
        next_encoded_state = self.dynamics_state_network(state_action)
        
        # 使用 LayerNorm 进行归一化
        normalized_next_state = self.dynamics_norm(next_encoded_state)
        
        # 使用归一化后的状态预测奖励
        reward_logits = self.dynamics_reward_network(normalized_next_state)
        
        return normalized_next_state, reward_logits

    def prediction(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测函数 f: 预测策略和价值。
        Args:
            encoded_state: 当前编码状态，形状 (batch_size, encoding_size)
        Returns:
            policy_logits: 策略 logits，形状 (batch_size, action_space_size)
            value_logits: 价值 logits，形状 (batch_size, full_support_size)
        """
        policy_logits = self.prediction_policy_network(encoded_state)
        value_logits = self.prediction_value_network(encoded_state)
        return policy_logits, value_logits

    def initial_inference(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        初始推理：从观察开始，计算初始编码状态、策略、价值。
        Args:
            observation: 初始观察状态 (已堆叠)，形状 (batch_size, C', H, W) 或 (batch_size, D')
        Returns:
            一个字典包含:
            - 'value': 预测的价值 (标量)，形状 (batch_size, 1)
            - 'value_logits': 预测的价值 logits，形状 (batch_size, full_support_size)
            - 'reward': 初始奖励 (标量，通常为0)，形状 (batch_size, 1)
            - 'reward_logits': 初始奖励 logits (通常为0)，形状 (batch_size, full_support_size)
            - 'policy_logits': 预测的策略 logits，形状 (batch_size, action_space_size)
            - 'encoded_state': 初始编码状态，形状 (batch_size, encoding_size)
        """
        batch_size = observation.shape[0]
        encoded_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(encoded_state)

        # 将 value_logits 转换为标量价值
        value = support_to_scalar(value_logits, self.config.support_size)

        # 初始奖励设为 0
        reward = torch.zeros(batch_size, 1).to(observation.device)
        # 初始奖励 logits 也设为 0 (对应于 support_to_scalar(logits) = 0)
        # 可以通过将标量 0 转换回 support 来实现
        reward_logits = scalar_to_support(reward, self.config.support_size)
        # 或者简单地创建一个零张量，如果不需要精确的 logits
        # reward_logits = torch.zeros(batch_size, self.full_support_size).to(observation.device)

        return {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "encoded_state": encoded_state,
        }

    def recurrent_inference(self, encoded_state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        循环推理：给定当前状态和动作，预测下一步的状态、奖励、策略、价值。
        Args:
            encoded_state: 当前编码状态，形状 (batch_size, encoding_size)
            action: 采取的动作 (整数索引)，形状 (batch_size, 1)
        Returns:
            一个字典包含:
            - 'value': 预测的价值 (标量)，形状 (batch_size, 1)
            - 'value_logits': 预测的价值 logits，形状 (batch_size, full_support_size)
            - 'reward': 预测的奖励 (标量)，形状 (batch_size, 1)
            - 'reward_logits': 预测的奖励 logits，形状 (batch_size, full_support_size)
            - 'policy_logits': 预测的策略 logits，形状 (batch_size, action_space_size)
            - 'encoded_state': 下一编码状态，形状 (batch_size, encoding_size)
        """
        batch_size = encoded_state.shape[0]
        # 将动作转换为 one-hot 编码
        action_one_hot = F.one_hot(action.squeeze(-1), num_classes=self.config.action_space_size).float()

        next_encoded_state, reward_logits = self.dynamics(encoded_state, action_one_hot)
        policy_logits, value_logits = self.prediction(next_encoded_state)

        # 将 logits 转换为标量值
        value = support_to_scalar(value_logits, self.config.support_size)
        reward = support_to_scalar(reward_logits, self.config.support_size)

        return {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "encoded_state": next_encoded_state, # 注意这里返回的是 next_encoded_state
        }

    def get_weights(self):
        """获取模型权重，用于保存到 SharedStorage。"""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        """从 SharedStorage 加载模型权重。"""
        self.load_state_dict(weights)


    def select_action(policy_logits: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        从策略logits中选择动作。
        Args:
            policy_logits: 策略网络输出的logits，形状 (batch_size, action_space_size)
            training: 是否处于训练模式
        Returns:
            选择的动作索引，形状 (batch_size, 1)
        """
        # 将logits转换为概率分布
        action_probs = F.softmax(policy_logits, dim=1)
        
        if training:
            # 训练时使用概率采样以保持探索
            action = torch.multinomial(action_probs, num_samples=1)
        else:
            # 评估时选择最优动作
            action = torch.argmax(action_probs, dim=1, keepdim=True)
        
        return action