### 1. Model 类接口定义

```python
class MuZeroNetwork:
    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始推理
        Args:
            observation: [batch_size, H, W, C*(stacked_observations+1)] - 堆叠的观察

        Returns:
            value: [batch_size, support_size] - 价值分布
            reward: [batch_size, support_size] - 奖励分布(初始为0)
            policy_logits: [batch_size, action_space_size] - 策略logits
            encoded_state: [batch_size, encoding_size] - 编码状态

        Args Example:
        Atari 先将输入从\(96×96×128\)（128帧，32* 3种颜色通道，32个动作相关特征 ）降采样到\(6×6×num\_channels\)。
        """

    def recurrent_inference(
        self, 
        encoded_state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """循环推理
        Args:
            encoded_state: [batch_size, encoding_size] - 编码状态
            action: [batch_size, 1] - 动作

        Returns:
            value: [batch_size, support_size] - 价值分布
            reward: [batch_size, support_size] - 奖励分布
            policy_logits: [batch_size, action_space_size] - 策略logits
            next_encoded_state: [batch_size, encoding_size] - 下一个编码状态
        """
```

### 2. Game 类接口定义

```python
class Game:
    def reset(self) -> np.ndarray:
        """重置游戏
        Returns:
            observation: [H, W, C] - 初始观察
        """

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """执行动作
        Args:
            action: int - 动作ID

        Returns:
            observation: [H, W, C] - 新的观察
            reward: float - 奖励值
            done: bool - 游戏是否结束
        """

    def to_play(self) -> int:
        """返回当前玩家
        Returns:
            player_id: int - 当前玩家ID
        """

    def legal_actions(self) -> List[int]:
        """返回合法动作
        Returns:
            actions: List[int] - 合法动作ID列表
        """
```

### 3. ReplayBuffer 类接口定义

```python
@ray.remote
class ReplayBuffer:
    def save_game(self, game_history: GameHistory, shared_storage: SharedStorage):
        """保存游戏历史
        Args:
            game_history: GameHistory对象，包含:
                - observation_history: List[np.ndarray] - [H, W, C]
                - action_history: List[int]
                - reward_history: List[float]
                - to_play_history: List[int]
                - child_visits: List[List[float]]
                - root_values: List[float]
            shared_storage: SharedStorage对象
        """

    def sample_batch(
        self,
        num_unroll_steps: int,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """采样训练批次
        Args:
            num_unroll_steps: int - 展开步数
            batch_size: int - 批次大小

        Returns:
            batch: Dict 包含:
                - observation_batch: [batch_size, channels, height, width]
                - action_batch: [batch_size, num_unroll_steps+1, 1]
                - target_value: [batch_size, num_unroll_steps+1]
                - target_reward: [batch_size, num_unroll_steps+1]
                - target_policy: [batch_size, num_unroll_steps+1, action_space_size]
                - weight_batch: [batch_size]
                - gradient_scale_batch: [batch_size, num_unroll_steps+1]
        """
```

### 4. SharedStorage 类接口定义

```python
@ray.remote
class SharedStorage:
    def set_info(self, info: Dict):
        """设置信息
        Args:
            info: Dict，可包含:
                - weights: NetworkWeights - 网络权重
                - training_step: int - 训练步数
                - episode_length: int - 回合长度
                - total_reward: float - 总奖励
                - mean_value: float - 平均价值
        """

    def get_info(self, keys: str) -> Any:
        """获取信息
        Args:
            keys: str - 信息键名

        Returns:
            value: Any - 对应的信息值
        """
```

### 5. GameHistory 类接口定义

```python
class GameHistory:
    def store_search_statistics(
        self,
        root: Node,
        action_space: List[int]
    ):
        """存储搜索统计
        Args:
            root: Node - MCTS根节点
            action_space: List[int] - 动作空间
        """

    def get_stacked_observations(
        self,
        index: int,
        num_stacked: int,
        action_space_size: int
    ) -> np.ndarray:
        """获取堆叠的观察
        Args:
            index: int - 当前索引
            num_stacked: int - 堆叠数量
            action_space_size: int - 动作空间大小

        Returns:
            stacked_observations: [H, W, C*(num_stacked+1)] - 堆叠的观察
        """
```