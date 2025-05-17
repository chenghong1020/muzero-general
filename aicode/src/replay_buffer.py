from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import copy
import threading
import concurrent.futures
from dataclasses import dataclass

from structures import GameHistory, TrainingBatch
from config import MuZeroConfig

class ReplayBuffer:
    """
    回放缓冲区，用于存储自博弈游戏历史并生成训练批次。
    参考 design_structure.md 和 design_interface.md。
    """
    
    def __init__(self, config: MuZeroConfig):
        """
        初始化回放缓冲区
        
        Args:
            config: MuZeroConfig 配置对象
        """
        self.config = config

        #below need to be thread safe
        self.buffer = {}  # 游戏历史缓冲区 {game_id: GameHistory}
        self.num_played_games = 0  # 已玩游戏数量
        self.num_played_steps = 0  # 已玩步数
        self.total_samples = 0  # 总样本数
        
        # 创建线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # 创建锁来保护共享变量
        self.buffer_lock = threading.Lock()
        
        # 设置随机种子
        np.random.seed(self.config.seed)
    
    def save_game(self, game_history: GameHistory, shared_storage=None):
        """
        保存游戏历史到回放缓冲区
        
        Args:
            game_history: GameHistory对象，包含游戏历史数据
            shared_storage: SharedStorage对象，用于更新统计信息
        """
        # 使用线程池异步执行保存游戏的操作
        future = self.executor.submit(self._save_game, game_history)
        
        # 如果需要，可以等待操作完成
        # future.result()
        
        # 如果有共享存储，可以在保存完成后更新统计信息
        if shared_storage:
            future.add_done_callback(
                lambda _: shared_storage.set_info("num_played_games", self.num_played_games)
            )
            future.add_done_callback(
                lambda _: shared_storage.set_info("num_played_steps", self.num_played_steps)
            )
    
    def _save_game(self, game_history: GameHistory):
        """内部方法：实际保存游戏历史的逻辑"""
        # 如果启用了优先经验回放，计算初始优先级
        if self.config.use_priority:  # 原来是self.config.PER
            if game_history.priorities is None:
                # 初始化优先级 (参考 MuZero 论文附录 Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    # 计算目标值与预测值之间的差异作为优先级
                    target_value = self._compute_target_value(game_history, i)
                    priority = (np.abs(root_value - target_value) ** self.config.priority_alpha)  # 原来是self.config.PER_alpha
                    priorities.append(priority)
                
                game_history.priorities = np.array(priorities, dtype=np.float32)
                game_history.game_priority = np.max(game_history.priorities)
            else:
                # 确保优先级数组是可写的（避免从磁盘加载时的只读数组）
                game_history.priorities = np.copy(game_history.priorities)
        
        # 使用锁保护共享变量的修改
        with self.buffer_lock:
            # 将游戏历史添加到缓冲区
            self.buffer[self.num_played_games] = game_history
            self.num_played_games += 1
            self.num_played_steps += len(game_history.action_history)
            self.total_samples += len(game_history.root_values)
            
            # 如果缓冲区超过最大大小，删除最旧的游戏
            if len(self.buffer) > self.config.replay_buffer_size:
                del_id = self.num_played_games - len(self.buffer)
                self.total_samples -= len(self.buffer[del_id].root_values)
                del self.buffer[del_id]
    
    #TODO - make the sample lockless and not blocking
    def sample_batch(self, num_unroll_steps: int, batch_size: int) -> TrainingBatch:
        """
        从回放缓冲区采样训练批次
        
        Args:
            num_unroll_steps: 展开步数
            batch_size: 批次大小
            
        Returns:
            batch: 包含训练数据的字典
        """
        # 使用线程池执行采样操作并等待结果
        future = self.executor.submit(self._sample_batch, num_unroll_steps, batch_size)
        return future.result()
    
    def _sample_batch(self, num_unroll_steps: int, batch_size: int) -> TrainingBatch:
        """
        内部方法：实际采样批次的逻辑
        
        Args:
            num_unroll_steps: 展开步数
            batch_size: 批次大小
            
        Returns:
            batch: 包含训练数据的字典，其中：
                - observation_batch: [batch_size, C, H, W] 或 [batch_size, stack*C, H, W]
                - action_batch: [batch_size, num_unroll_steps]
                - target_value: [batch_size, num_unroll_steps + 1]
                - target_reward: [batch_size, num_unroll_steps + 1]
                - target_policy: [batch_size, num_unroll_steps + 1, action_space_size]
                - weight_batch: [batch_size] (如果使用优先经验回放)
                - gradient_scale_batch: [batch_size, num_unroll_steps + 1]
        """
        # 采样游戏和位置
        observation_batch = []
        action_batch = []
        target_value_batch = []
        target_reward_batch = []
        target_policy_batch = []
        weight_batch = [] if self.config.use_priority else None
        gradient_scale_batch = []
        
        # 使用锁保护对缓冲区的读取
        with self.buffer_lock:
            # 采样 batch_size 个游戏
            game_samples = self._sample_n_games(batch_size)
        
        for game_id, game_history, game_prob in game_samples:
            # 在游戏中采样位置
            game_pos, pos_prob = self._sample_position(game_history)
            
            # 获取堆叠的观察
            observation = game_history.get_stacked_observations(
                game_pos, 
                self.config.stacked_observations
            )
            
            # 计算目标值
            targets = game_history.make_target(
                game_pos,
                num_unroll_steps,
                self.config.td_steps,
                self.config.discount
            )
            # targets 的维度为 [(num_unroll_steps + 1)]，其中每个元素是一个元组 (value, reward, policy)
            # value: float - 目标价值
            # reward: float - 目标奖励
            # policy: Dict[int, float] - 目标策略，是一个动作到概率的映射字典 {action: probability}
            
            # 提取目标值、奖励和策略
            values, rewards, policies = [], [], []
            actions = []
            
            # 处理所有状态（初始状态 + 展开步骤）
            for current_index in range(num_unroll_steps + 1):
                # 获取目标索引
                target_index = current_index
                
                if target_index < len(targets):
                    # 提取目标值
                    target_value, target_reward, target_policy = targets[target_index]
                    values.append(target_value)
                    rewards.append(target_reward)
                    
                    # 将策略字典转换为向量
                    policy_vec = np.zeros(len(self.config.action_space), dtype=np.float32)
                    for action, prob in target_policy.items():
                        if action < len(policy_vec):
                            policy_vec[action] = prob
                    policies.append(policy_vec)
                    
                    # 添加动作（只为前num_unroll_steps个状态添加动作）
                    if current_index < num_unroll_steps:
                        action_pos = game_pos + current_index
                        action = game_history.action_history[action_pos] if action_pos < len(game_history.action_history) else 0
                        actions.append(action)
                else:
                    # 对于超出游戏长度的步骤，使用零值
                    values.append(0.0)
                    rewards.append(0.0)
                    policies.append(np.zeros(len(self.config.action_space), dtype=np.float32))
                    
                    # 添加默认动作（只为前num_unroll_steps个状态添加动作）
                    if current_index < num_unroll_steps:
                        actions.append(0)  # 默认动作
            
            # 添加到批次
            observation_batch.append(observation)
            action_batch.append(actions)
            target_value_batch.append(values)
            target_reward_batch.append(rewards)
            target_policy_batch.append(policies)
            
            # 计算梯度缩放因子
            gradient_scale = 1.0 / (num_unroll_steps)
            gradient_scale_batch.append([gradient_scale] * (num_unroll_steps + 1))
            
            # 如果使用优先经验回放，计算重要性采样权重
            if self.config.use_priority:
                # 计算重要性采样权重 (1 / (N * P(i)))
                weight = 1.0 / (self.total_samples * game_prob * pos_prob)
                weight_batch.append(weight)
        
        # 转换为张量
        observation_batch = torch.from_numpy(np.array(observation_batch)).float()
        action_batch = torch.from_numpy(np.array(action_batch)).long()
        target_value_batch = torch.from_numpy(np.array(target_value_batch)).float()
        target_reward_batch = torch.from_numpy(np.array(target_reward_batch)).float()
        target_policy_batch = torch.from_numpy(np.array(target_policy_batch)).float()
        gradient_scale_batch = torch.from_numpy(np.array(gradient_scale_batch)).float()
        
        # 如果使用优先经验回放，归一化权重
        if self.config.use_priority:
            weight_batch = np.array(weight_batch, dtype=np.float32)
            weight_batch = weight_batch / np.max(weight_batch)
            weight_batch = torch.from_numpy(weight_batch).float()
        
        # 创建训练批次
        batch = TrainingBatch(
            observation_batch=observation_batch,
            action_batch=action_batch,
            target_value=target_value_batch,
            target_reward=target_reward_batch,
            target_policy=target_policy_batch,
            weight_batch=weight_batch,
            gradient_scale_batch=gradient_scale_batch
        )
        
        return batch
    
    def _sample_n_games(self, n_games: int, force_uniform: bool = False) -> List[Tuple[int, GameHistory, float]]:
        """
        采样n个游戏
        
        Args:
            n_games: 要采样的游戏数量
            force_uniform: 是否强制均匀采样
            
        Returns:
            采样的游戏列表 [(game_id, game_history, game_prob)]
        """
        # 注意：此方法应在持有buffer_lock的情况下调用
        if self.config.use_priority and not force_uniform:
            # 根据优先级采样
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            
            game_probs = np.array(game_probs, dtype=np.float32)
            game_probs /= np.sum(game_probs)
            game_prob_dict = {game_id: prob for game_id, prob in zip(game_id_list, game_probs)}
            
            # 根据优先级采样游戏
            selected_games = np.random.choice(game_id_list, n_games, p=game_probs)
        else:
            # 均匀采样
            selected_games = np.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        
        # 返回采样的游戏
        return [(game_id, self.buffer[game_id], game_prob_dict.get(game_id, 1.0/len(self.buffer))) 
                for game_id in selected_games]
    
    def _sample_position(self, game_history: GameHistory, force_uniform: bool = False) -> Tuple[int, float]:
        """
        在游戏中采样位置
        
        Args:
            game_history: 游戏历史
            force_uniform: 是否强制均匀采样
            
        Returns:
            (position_index, position_prob)
        """
        position_prob = None
        if self.config.use_priority and not force_uniform and game_history.priorities is not None:
            # 根据优先级采样位置
            position_probs = game_history.priorities / np.sum(game_history.priorities)
            position_index = np.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            # 均匀采样位置
            position_index = np.random.choice(len(game_history.root_values))
            position_prob = 1.0 / len(game_history.root_values)
        
        return position_index, position_prob
    
    def _compute_target_value(self, game_history: GameHistory, index: int) -> float:
        """
        计算目标值
        
        Args:
            game_history: 游戏历史
            index: 当前位置索引
            
        Returns:
            目标值
        """
        # 目标值是未来 td_steps 步的折扣根节点值，加上所有奖励的折扣和
        bootstrap_index = index + self.config.td_steps
        
        if bootstrap_index < len(game_history.root_values):
            # 使用未来状态的值作为引导值
            value = game_history.root_values[bootstrap_index] * (self.config.discount ** self.config.td_steps)
             # 累加中间奖励
            for i, reward in enumerate(game_history.reward_history[index + 1 : bootstrap_index + 1]):
                value += reward * (self.config.discount ** i)
        else:
            # 如果超出游戏长度，使用0
            value = 0.0
            for i, reward in enumerate(game_history.reward_history[index + 1 :]):
                value += reward * (self.config.discount ** i)
         
        return value
    
    def update_priorities(self, priorities: np.ndarray, indices: List[Tuple[int, int]]):
        """
        更新优先级
        
        Args:
            priorities: 新的优先级值
            indices: 对应的索引 [(game_id, position)]
        """
        # 使用线程池异步执行更新优先级的操作
        self.executor.submit(self._update_priorities, priorities, indices)
    
    def _update_priorities(self, priorities: np.ndarray, indices: List[Tuple[int, int]]):
        """内部方法：实际更新优先级的逻辑"""
        if not self.config.use_priority:
            return
        
        with self.buffer_lock:
            for i, (game_id, position) in enumerate(indices):
                # 检查游戏是否仍在缓冲区中
                if game_id in self.buffer:
                    # 更新位置优先级
                    priority = priorities[i]
                    self.buffer[game_id].priorities[position] = priority
                    
                    # 更新游戏优先级
                    self.buffer[game_id].game_priority = np.max(self.buffer[game_id].priorities)
    
    def get_buffer(self):
        """获取缓冲区内容"""
        # 使用线程池执行获取缓冲区的操作并等待结果
        future = self.executor.submit(lambda: copy.deepcopy(self.buffer))
        return future.result()
    
    def close(self):
        """关闭线程池"""
        self.executor.shutdown()