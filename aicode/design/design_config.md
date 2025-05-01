# 配置参数

```python
class MuZeroConfig:
    """MuZero 算法配置参数"""
    
    def __init__(self):
        # 1. 游戏环境参数
        self.action_space_size: int          # 动作空间大小
        self.max_moves: int                  # 最大移动步数
        self.observation_shape: Tuple[int]   # 观察空间形状 [H, W, C]
        self.players: List[int]              # 玩家ID列表
        
        # 2. 网络结构参数
        self.encoding_size: int              # 编码状态维度
        self.num_channels: int               # 通道数
        self.fc_representation_layers: List[int]  # 表征网络隐藏层配置
        self.fc_dynamics_layers: List[int]       # 动态网络隐藏层配置
        self.fc_reward_layers: List[int]         # 奖励网络隐藏层配置
        self.fc_policy_layers: List[int]         # 策略网络隐藏层配置
        self.fc_value_layers: List[int]          # 价值网络隐藏层配置
        self.support_size: int               # 价值和奖励的分类数
        self.stacked_observations: int       # 堆叠的观察数量
        
        # 3. MCTS参数
        self.num_simulations: int           # MCTS模拟次数
        self.discount: float                # 奖励折扣因子
        self.pb_c_base: float              # UCB基础参数
        self.pb_c_init: float              # UCB初始化参数
        self.root_dirichlet_alpha: float    # 根节点Dirichlet噪声α参数
        self.root_exploration_fraction: float  # 根节点探索比例
        
        # 4. 训练参数
        self.num_unroll_steps: int          # 展开步数
        self.batch_size: int                # 训练批次大小
        self.training_steps: int            # 训练总步数
        self.checkpoint_interval: int        # 检查点保存间隔
        self.weight_decay: float            # 权重衰减
        self.learning_rate_init: float      # 初始学习率
        self.learning_rate_decay_steps: int  # 学习率衰减步数
        self.learning_rate_decay_rate: float # 学习率衰减率
        
        # 5. 优先经验回放参数
        self.priority_window_size: int       # 优先级窗口大小
        self.priority_alpha: float           # 优先级指数
        self.priority_beta: float            # 重要性采样指数
        
        # 6. 自博弈参数
        self.selfplay_on_gpu: bool          # 是否在GPU上进行自博弈
        self.num_actors: int                # 自博弈并行进程数
        self.temperature_threshold: float    # 温度阈值
        self.visit_softmax_temperature_fn: Callable  # 访问计数温度函数
        
        # 7. 评估参数
        self.muzero_player: int             # MuZero玩家ID
        self.opponent: str                  # 对手类型
        self.eval_episodes: int             # 评估回合数
        
        # 8. 存储参数
        self.replay_buffer_size: int        # 回放缓冲区大小
        self.window_size: int               # 游戏窗口大小
        self.save_path: str                 # 模型保存路径

    def visit_softmax_temperature(self, trained_steps: int) -> float:
        """根据训练步数返回访问计数的softmax温度"""
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    @staticmethod
    def get_atari_config() -> 'MuZeroConfig':
        """返回Atari游戏的默认配置"""
        config = MuZeroConfig()
        # 设置Atari特定的参数...
        return config

    @staticmethod
    def get_board_game_config() -> 'MuZeroConfig':
        """返回棋盘游戏的默认配置"""
        config = MuZeroConfig()
        # 设置棋盘游戏特定的参数...
        return config
```

配置参数的主要特点：

1. **模块化分组**：
   - 游戏环境参数
   - 网络结构参数
   - MCTS参数
   - 训练参数
   - 优先经验回放参数
   - 自博弈参数
   - 评估参数
   - 存储参数

2. **灵活性**：
   - 支持不同游戏类型的配置
   - 可动态调整的温度函数
   - 可配置的网络结构

3. **完整性**：
   - 覆盖算法所有组件
   - 包含所有必要的超参数
   - 支持完整的训练流程

4. **可扩展性**：
   - 静态方法生成特定配置
   - 支持自定义参数组合
   - 便于添加新的配置项

这些配置参数支持了 MuZero 算法的完整实现，包括：
- MCTS搜索过程
- 神经网络训练
- 自博弈数据生成
- 优先经验回放
- 模型评估和保存