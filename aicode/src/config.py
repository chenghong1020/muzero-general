import math
from typing import List, Optional, Tuple

class MuZeroConfig:
    def __init__(self,
                 # --- 游戏环境相关 ---
                 seed: int = 0,
                 action_space_size: int = 0, # 需要根据具体游戏设置
                 observation_shape: Tuple[int, int, int] = (0, 0, 0), # (channels, height, width) 需要根据具体游戏设置
                 stacked_observations: int = 0, # 堆叠的观察帧数

                 # --- 网络结构相关 ---
                 encoding_size: int = 128,
                 fc_representation_layers: List[int] = [], # 表征网络全连接层
                 fc_dynamics_layers: List[int] = [64],     # 动态网络全连接层
                 fc_reward_layers: List[int] = [64],       # 奖励预测网络全连接层
                 fc_value_layers: List[int] = [64],        # 价值预测网络全连接层
                 fc_policy_layers: List[int] = [64],       # 策略预测网络全连接层
                 support_size: int = 10,                   # 价值和奖励分布的范围 [-support_size, support_size]

                 # --- MCTS 搜索相关 ---
                 num_simulations: int = 50,                # MCTS 模拟次数
                 discount: float = 0.997,                  # 折扣因子 gamma
                 # UCB 参数
                 root_dirichlet_alpha: float = 0.25,
                 root_exploration_fraction: float = 0.25,
                 pb_c_base: float = 19652,
                 pb_c_init: float = 1.25,
                 # 探索温度相关
                 temperature_threshold: Optional[int] = None, # 控制探索的步数阈值

                 # --- 训练过程相关 ---
                 training_steps: int = 1000000,
                 batch_size: int = 1024,
                 num_unroll_steps: int = 5,                # 网络展开步数 K
                 td_steps: int = 10,                       # TD 步数
                 lr_init: float = 0.05,                    # 初始学习率
                 lr_decay_steps: float = 350e3,
                 lr_decay_rate: float = 0.1,
                 weight_decay: float = 1e-4,               # 权重衰减
                 value_loss_weight: float = 0.25,          # 价值损失权重
                 reward_loss_weight: float = 1.0,          # 奖励损失权重 (通常设为1，其他损失相对调整)
                 policy_loss_weight: float = 1.0,          # 策略损失权重
                 max_grad_norm: float = 5.0,               # 梯度裁剪阈值

                 # --- 优先经验回放 (PER) 相关 ---
                 use_priority: bool = False,               # 是否使用 PER
                 priority_alpha: float = 1.0,              # PER alpha 参数
                 priority_beta_start: float = 1.0,         # PER beta 起始值
                 priority_beta_steps: int = 0,             # PER beta 线性增长步数

                 # --- 自我对弈 (Self-Play) 相关 ---
                 num_actors: int = 1,                      # 自我对弈 Actor 数量
                 replay_buffer_size: int = 100000,         # 回放缓冲区大小
                 checkpoint_interval: int = 1000,          # 模型保存间隔
                 games_per_evaluation: int = 10,           # 每次评估运行的游戏局数

                 # --- 存储相关 ---
                 results_path: str = "./results",          # 结果保存路径
                 ):

        # --- 游戏环境 ---
        self.seed = seed
        self.action_space_size = action_space_size
        self.observation_shape = observation_shape
        self.stacked_observations = stacked_observations

        # --- 网络结构 ---
        self.encoding_size = encoding_size
        self.fc_representation_layers = fc_representation_layers
        self.fc_dynamics_layers = fc_dynamics_layers
        self.fc_reward_layers = fc_reward_layers
        self.fc_value_layers = fc_value_layers
        self.fc_policy_layers = fc_policy_layers
        self.support_size = support_size

        # --- MCTS ---
        self.num_simulations = num_simulations
        self.discount = discount
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.temperature_threshold = temperature_threshold

        # --- 训练 ---
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.lr_init = lr_init
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.value_loss_weight = value_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.max_grad_norm = max_grad_norm

        # --- PER ---
        self.use_priority = use_priority
        self.priority_alpha = priority_alpha
        self.priority_beta_start = priority_beta_start
        self.priority_beta_steps = priority_beta_steps

        # --- 自我对弈 ---
        self.num_actors = num_actors
        self.replay_buffer_size = replay_buffer_size
        self.checkpoint_interval = checkpoint_interval
        self.games_per_evaluation = games_per_evaluation

        # --- 存储 ---
        self.results_path = results_path

        # --- 动态计算的参数 ---
        self.visit_softmax_temperature_fn = self.create_visit_softmax_temperature_fn()

    def create_visit_softmax_temperature_fn(self):
        """
        根据训练步数动态调整 MCTS 访问次数分布的温度参数。
        参考 design_config.md 中的描述。
        """
        threshold = self.temperature_threshold
        if threshold is None:
            # 如果未设置阈值，则始终使用温度 1.0
            return lambda training_step: 1.0
        else:
            # 在阈值内逐渐降低温度，之后固定为 0.0 (greedy)
            # 这里可以根据需要实现更复杂的调度逻辑
            # 例如: 前 50% 步数用 1.0, 接着 25% 用 0.5, 最后用 0.0 (greedy)
            # 以下是一个简单的线性衰减示例，具体实现可调整
            def temperature_fn(training_step: int) -> float:
                if training_step < threshold * 0.5:
                    return 1.0
                elif training_step < threshold * 0.75:
                    return 0.5
                else:
                    # 在实际应用中，可能不会直接降到 0，而是接近 0 的一个很小的值
                    # 或者根据评估效果决定是否切换到 greedy
                    return 0.25 # 或者更小的值，甚至 0 for greedy play
            return temperature_fn

    @staticmethod
    def get_atari_config() -> 'MuZeroConfig':
        """
        获取 Atari 游戏的预设配置。
        需要填充具体的 Atari 参数。
        """
        # 示例: 填充 Atari 特定参数
        config = MuZeroConfig(
            action_space_size=18, # 示例值
            observation_shape=(96, 96, 3), # 示例值
            stacked_observations=4, # 示例值
            discount=0.997,
            num_simulations=50,
            support_size=300, # Atari 通常用更大的 support size
            # ... 其他 Atari 特定参数 ...
        )
        print("Warning: get_atari_config() is using placeholder values. Please fill in actual Atari parameters.")
        return config

    @staticmethod
    def get_board_game_config() -> 'MuZeroConfig':
        """
        获取棋盘类游戏 (如围棋、象棋) 的预设配置。
        需要填充具体的棋盘游戏参数。
        """
         # 示例: 填充棋盘游戏特定参数 (以井字棋为例)
        config = MuZeroConfig(
            action_space_size=9, # 井字棋
            observation_shape=(3, 3, 2), # 棋盘状态 + 玩家信息
            stacked_observations=0, # 棋盘游戏通常不需要堆叠帧
            discount=1.0, # 棋盘游戏通常 discount 为 1
            num_simulations=50,
            support_size=10, # 棋盘游戏价值范围通常较小
            # ... 其他棋盘游戏特定参数 ...
        )
        print("Warning: get_board_game_config() is using placeholder values. Please fill in actual board game parameters.")
        return config

# 可以在这里添加一个主函数入口或者测试代码来验证配置类的使用
if __name__ == '__main__':
    # 创建一个默认配置实例
    config = MuZeroConfig(action_space_size=9, observation_shape=(3,3,1)) # 假设井字棋

    print(f"Action space size: {config.action_space_size}")
    print(f"Number of simulations: {config.num_simulations}")
    print(f"Initial learning rate: {config.lr_init}")

    # 测试温度函数
    temp_fn = config.visit_softmax_temperature_fn
    print(f"Temperature at step 0: {temp_fn(0)}")
    if config.temperature_threshold:
        print(f"Temperature at step {config.temperature_threshold // 2}: {temp_fn(config.temperature_threshold // 2)}")
        print(f"Temperature at step {config.temperature_threshold}: {temp_fn(config.temperature_threshold)}")

    # 获取预设配置 (注意需要填充实际参数)
    # atari_config = MuZeroConfig.get_atari_config()
    # board_game_config = MuZeroConfig.get_board_game_config()