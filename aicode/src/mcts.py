import math
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from config import MuZeroConfig
from model import MuZeroNetwork
from utils import support_to_scalar, scalar_to_support  # 添加导入
from game import Game

# 可以将 Player 简化为 int 类型的别名 （双人制 1 本玩家 -1 对手）
Player = int

# --- MCTS 核心数据结构 ---

class Node:
    """MCTS 树节点"""
    def __init__(self, prior: float, to_play: Player):
        # 节点被访问的次数，用于计算 UCB 分数和评估节点可信度
        self.visit_count: int = 0
        
        # 当前节点轮到哪个玩家行动
        # 在双人游戏中通常用 1 和 -1 表示两个玩家
        self.to_play: Player = to_play
        
        # 节点的先验概率，来自神经网络的策略预测
        # 表示在父节点状态下选择该动作的概率
        self.prior: float = prior
        
        # 所有访问该节点的模拟中累积的价值总和
        # 平均价值 = value_sum / visit_count
        self.value_sum: float = 0
        
        # 子节点字典，键为动作索引，值为对应的子节点
        # 例如在井字棋中，键为 0-8 表示九个位置
        self.children: Dict[int, Node] = {}
        
        # 神经网络编码的状态表示
        # 用于神经网络的动态预测和价值评估
        self.hidden_state: Optional[torch.Tensor] = None
        
        # 执行到达该节点的动作获得的即时奖励
        # 例如：获胜 = 1，失败 = -1，未结束 = 0
        self.reward: float = 0

    def expanded(self) -> bool:
        """检查节点是否已扩展"""
        return len(self.children) > 0

    def value(self) -> float:
        """计算节点的平均价值"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def expand(self,
               actions: List[int],
               to_play: Player,
               reward: float,
               policy_logits: torch.Tensor,
               hidden_state: torch.Tensor,
               config: MuZeroConfig):
        """
        扩展节点，创建子节点。
        Args:
            actions: 可用动作列表。
            to_play: 当前轮到的玩家。
            reward: 到达此节点的即时奖励。
            policy_logits: 预测网络输出的策略 logits。
            hidden_state: 动态网络输出的隐藏状态。
            config: MuZero配置。
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
    
        # 使用 softmax 获取策略概率
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
    
        # 使用配置中的玩家切换函数
        next_to_play = config.next_player_fn(to_play)
        
        for action in actions:
            self.children[action] = Node(prior=policy[action], to_play=next_to_play)

class MinMaxStats:
    """用于在 MCTS 搜索中跟踪和归一化价值"""
    def __init__(self):
        self.maximum: float = -float("inf")
        self.minimum: float = float("inf")

    def update(self, value: float):
        """更新最大值和最小值"""
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """将价值归一化到 [0, 1] 区间"""
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

# --- MCTS 搜索函数 ---

def run_mcts(config: MuZeroConfig, 
             root: Node, 
             action_space: List[int], 
             model: MuZeroNetwork, 
             legal_actions: List[int], 
             to_play: Player, 
             add_exploration_noise: bool, 
             observation: np.ndarray,  # 添加观察状态参数 
             override_root_with=None):
    """
    执行 MCTS 搜索。
    Args:
        config: MuZero 配置。
        root: 当前搜索树的根节点。
        action_space: 完整的动作空间列表。
        model: MuZero 网络模型。
        legal_actions: 当前状态下的合法动作列表。
        to_play: 当前轮到的玩家。
        add_exploration_noise: 是否在根节点添加探索噪声。
        observation: 当前游戏的观察状态，用于模型初始推理。
        override_root_with: (可选) 用于从外部提供根节点信息 (例如，来自之前的搜索)。

    Returns:
        更新后的根节点。
    """
    min_max_stats = MinMaxStats()
    max_tree_depth = 0

    # 如果提供了外部根节点信息，则使用它
    if override_root_with:
        root_predicted_value = override_root_with["value"]
        policy_logits = override_root_with["policy_logits"]
        hidden_state = override_root_with["encoded_state"]
        reward = override_root_with["reward"] # 初始奖励通常为0
        root.expand(legal_actions, to_play, reward, policy_logits, hidden_state, config)
    else:
        # 如果没有提供，需要模型进行初始推理
        # 需要将观察状态转换为张量并传递给模型
        observation_tensor = torch.tensor(observation).float().unsqueeze(0).to(next(model.parameters()).device)
        
        # 使用 torch.no_grad() 上下文管理器进行推理，避免梯度计算
        with torch.no_grad():
            network_output = model.initial_inference(observation_tensor)
        
        root_predicted_value = support_to_scalar(network_output["value_logits"], config.support_size).item()
        policy_logits = network_output["policy_logits"]
        hidden_state = network_output["encoded_state"]
        reward = 0.0  # 初始奖励为0
        
        root.expand(legal_actions, to_play, reward, policy_logits, hidden_state, config)


    if add_exploration_noise:
        add_dirichlet_noise(config, root)  # 移除 legal_actions 参数

    for _ in range(config.num_simulations):
        virtual_to_play = to_play
        node = root
        search_path = [node]
        current_tree_depth = 0

        # --- 1. 选择阶段 ---
        while node.expanded():
            current_tree_depth += 1
            action, node = select_child(config, node, min_max_stats)
            search_path.append(node)
            # 使用配置中的玩家切换函数
            virtual_to_play = config.next_player_fn(virtual_to_play)

        # --- 2. 扩展阶段 ---
        parent = search_path[-2]
        # 使用模型进行循环推理
        with torch.no_grad():
            network_output = model.recurrent_inference(parent.hidden_state, torch.tensor([[action]]).to(parent.hidden_state.device))

        value = support_to_scalar(network_output["value_logits"], config.support_size).item()
        reward = support_to_scalar(network_output["reward_logits"], config.support_size).item()
        policy_logits = network_output["policy_logits"]
        hidden_state = network_output["encoded_state"]

        # 扩展叶子节点
        node.expand(action_space, virtual_to_play, reward, policy_logits, hidden_state, config)

        max_tree_depth = max(max_tree_depth, current_tree_depth)

        # --- 3. 反向传播阶段 ---
        backpropagate(config, search_path, value, virtual_to_play, config.discount, min_max_stats)

    # 返回包含搜索统计信息的根节点
    # 可以在这里添加map结构来返回更详细的信息
    extra_info = {
        "max_tree_depth": max_tree_depth,
        "root_predicted_value": root_predicted_value,
    }
    return root, extra_info


def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    """
    使用 UCB 公式选择子节点。
    """
    best_score = -float('inf')
    best_action = -1
    best_child = None

    for action, child in node.children.items():
        score = ucb_score(config, node, child, min_max_stats)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    if best_child is None:
        # 如果没有子节点被访问（理论上不应发生，因为expand阶段会创建所有子节点）
        # 需要处理这种情况，例如随机选择一个动作或抛出错误
        print(f"Warning: No child selected for node with {node.visit_count} visits. Children actions: {list(node.children.keys())}")
        # 临时策略：从所有子节点中随机选择一个（如果存在）
        if node.children:
            best_action = random.choice(list(node.children.keys()))
            best_child = node.children[best_action]
        else: # 没有子节点，无法选择
            raise Exception("Cannot select child: Node has no children.")

    return best_action, best_child

def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    """
    计算 UCB 分数。
    UCB = Q(s,a) + P(s,a) * c * sqrt(N(s)) / (1 + N(s,a))
    Q(s,a) = (reward + discount * value)
    """
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    # 价值部分需要归一化
    if child.visit_count > 0:
        # Q(s,a) = R(s,a) + gamma * V(s')
        # R(s,a) 是从父节点采取动作 a 到达子节点的奖励 child.reward
        # V(s') 是子节点的平均价值 child.value()
        # 注意：这里的 to_play 是父节点的 to_play

        if config.is_two_player_game:
            # 对于双人游戏，子节点的价值 V(s') 需要从当前玩家（父节点）的角度来看，
            # 即 child.reward + discount * (-child.value())
            value_score = child.reward + config.discount * (-child.value())
        else:
            # 在单玩家游戏中，价值不需要调整
            value_score = child.reward + config.discount * child.value()
        # AlphaZero/MuZero 使用 MinMax 归一化
        normalized_value = min_max_stats.normalize(value_score)
    else:
        normalized_value = 0 # 未访问过的节点价值视为 0

    return prior_score + normalized_value


def backpropagate(config: MuZeroConfig, search_path: List[Node], value: float, to_play: Player, discount: float, min_max_stats: MinMaxStats):
    """
    反向传播价值和更新节点统计。
    Args:
        config: MuZero配置。
        search_path: 从根到叶节点的路径。
        value: 叶子节点的网络预测价值。
        to_play: 叶子节点的玩家。
        discount: 折扣因子。
        min_max_stats: 用于更新价值范围。
    """
    # 判断是单人游戏还是双人游戏
    is_two_player = config.is_two_player_game
    
    # 从叶子节点开始，向上反向传播
    for node in reversed(search_path):
        # 更新节点访问次数
        node.visit_count += 1
        
        if is_two_player:
            # 双人游戏（零和博弈）情况下的价值更新
            # 如果当前节点的玩家与叶子节点的玩家相同，则加上value；否则减去value
            node.value_sum += value if node.to_play == to_play else -value
            
            # 计算用于更新MinMaxStats的值
            # 在双人游戏中，从当前玩家角度看，子节点的价值需要取反
            value_for_minmax = node.reward + discount * (-node.value())
            min_max_stats.update(value_for_minmax)
            
            # 计算传递给父节点的值
            # 如果当前节点的玩家与叶子节点的玩家相同，则奖励取反；否则保持不变
            value = (-node.reward if node.to_play == to_play else node.reward) + discount * value
        else:
            # 单人游戏情况下的价值更新
            # 直接累加价值
            node.value_sum += value
            
            # 计算用于更新MinMaxStats的值
            value_for_minmax = node.reward + discount * node.value()
            min_max_stats.update(value_for_minmax)
            
            # 计算传递给父节点的值
            # 当前节点的奖励加上折扣后的子节点价值
            value = node.reward + discount * value

def add_dirichlet_noise(config: MuZeroConfig, node: Node):
    """在根节点的先验概率上添加 Dirichlet 噪声以促进探索"""
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    
    for idx, action in enumerate(actions):
        node.children[action].prior = node.children[action].prior * (1 - frac) + noise[idx] * frac

def select_action(config: MuZeroConfig, num_moves: int, node: Node, training: bool = True):
    """
    根据访问次数选择最终动作。
    Args:
        config: MuZero 配置。
        num_moves: 当前游戏进行的步数。
        node: MCTS 搜索后的根节点。
        training: 是否处于训练模式。
    Returns:
        选择的动作索引。
    """
    visit_counts = [(action, child.visit_count) for action, child in node.children.items()]
    if not visit_counts:
        # 如果没有子节点被访问（例如模拟次数为0或只有非法动作），随机选择一个合法动作
        # 需要从 Game 对象获取 legal_actions
        # return random.choice(game.legal_actions()) # 假设有 game 对象
        raise ValueError("Cannot select action: No children visited.")


    if training:
        # 根据温度参数计算访问次数的分布
        temperature = config.visit_softmax_temperature_fn(num_moves)
        actions = [vc[0] for vc in visit_counts]
        counts = np.array([vc[1] for vc in visit_counts], dtype=np.float32)

        if temperature == 0: # 确定性选择 (greedy)
            action = actions[np.argmax(counts)]
        elif temperature == float('inf'): # 均匀随机选择
             action = random.choice(actions)
        else:
            # 使用温度调整概率分布
            counts_temp = counts**(1./temperature)
            probs = counts_temp / np.sum(counts_temp)
            action = np.random.choice(actions, p=probs)
    else:
        # 评估模式：选择访问次数最多的动作 (greedy)
        action = max(visit_counts, key=lambda item: item[1])[0]

    return action


class MCTSFacade:
    """
    MCTS 搜索的外观模式实现，简化 MCTS 搜索的启动过程。
    只需要提供必要的参数如 game、model 和 config。
    """
    def __init__(self, config: MuZeroConfig, model: MuZeroNetwork, game: Game):
        """
        初始化 MCTS Facade。
        
        Args:
            config: MuZero 配置。
            model: MuZero 网络模型。
            game: 游戏环境实例。
        """
        self.config = config
        self.model = model
        self.game = game
    
    def run(self, override_root_with=None) -> tuple:
        """
        执行 MCTS 搜索。
        
        Args:
            override_root_with: (可选) 用于从外部提供根节点信息。
        
        Returns:
            tuple: (搜索后的根节点, 额外信息)
        """
        # 创建根节点
        root = Node(prior=1.0, to_play=self.game.to_play())
        
        # 从游戏环境获取当前状态
        observation = self.game.get_observation()
        legal_actions = self.game.legal_actions()
        to_play = self.game.to_play()
        
        # 确定完整的动作空间
        action_space = list(range(self.config.action_space_size))
        
        # 执行 MCTS 搜索
        return run_mcts(
            config=self.config,
            root=root,
            action_space=action_space,
            model=self.model,
            legal_actions=legal_actions,
            to_play=to_play,
            add_exploration_noise=self.config.add_exploration_noise,
            observation=observation,
            override_root_with=override_root_with
        )
    
    def select_action(self, root: Node, training: bool = True) -> int:
        """
        根据 MCTS 搜索结果选择动作。
        
        Args:
            root: MCTS 搜索后的根节点。
            training: 是否处于训练模式，默认为 True。
            
        Returns:
            int: 选择的动作索引。
        """
        # 获取当前游戏的步数（如果游戏类提供此信息）
        num_moves = getattr(self.game, 'num_moves', lambda: 0)()
        
        return select_action(self.config, num_moves, root, training)
    
    def search_and_play(self, training: bool = True) -> tuple:
        """
        执行完整的搜索并选择动作。
        
        Args:
            training: 是否处于训练模式，默认为 True。
            
        Returns:
            tuple: (选择的动作, 搜索树根节点)
        """
        # 执行搜索
        root, extra_info = self.run()
        
        # 选择动作
        action = self.select_action(root, training=training)
        
        return action, root