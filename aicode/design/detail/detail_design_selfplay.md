# MuZero Self-Play 详细设计

## 1. 概述

自我对弈 (Self-Play) 是 MuZero 算法中通过与自身或其他版本的智能体进行对弈来生成训练数据的核心过程。本设计详细描述了 `run_selfplay` 和 `play_game` 的逻辑，以及它们如何与 `Game`, `MuZeroNetwork`, `MCTS`, `ReplayBuffer`, `SharedStorage`, 和 `GameHistory` 等核心组件集成，以持续高效地生成高质量的训练样本。

## 2. 设计目标

-   实现一个健壮且高效的自我对弈流程，能够持续不断地生成游戏经验。
-   明确各组件在自我对弈过程中的职责和交互方式。
-   支持并行化自我对弈，以加速数据收集。
-   确保与现有组件接口的兼容性。

## 3. 核心组件与流程

### 3.1. `run_selfplay` (主循环/进程协调器)

`run_selfplay` 负责初始化和管理整个自我对弈过程，协调多个并行的游戏对弈 Actor。

**流程:**

1.  **初始化**:
    *   创建或连接到 `SharedStorage` 实例 (在主进程中)。
    *   创建或连接到 `ReplayBuffer` 实例 (作为独立的子进程)。
    *   根据配置 (`MuZeroConfig`)，确定要启动的 `play_game` Actor (子进程) 的数量。
    *   使用 `multiprocessing.Process` 或 `concurrent.futures.ProcessPoolExecutor` 启动指定数量的 `play_game` Actor。

2.  **主循环**:
    *   持续监控 `play_game` Actor 的状态。
    *   收集已完成游戏的 `GameHistory` 对象 (通过进程间通信，例如 `multiprocessing.Queue`)。
    *   将收集到的 `GameHistory` 对象发送给 `ReplayBuffer` 子进程进行存储。
    *   (可选) 根据系统负载或配置，动态调整 `play_game` Actor 的数量。
    *   (可选) 实现优雅的停止机制，允许在不丢失数据的情况下终止自我对弈过程。

### 3.2. `play_game` (单个游戏对弈 Actor - 子进程)

每个 `play_game` Actor 负责独立地进行一局或多局完整的游戏，并将游戏历史记录下来。

**流程:**

1.  **初始化 (在每个子进程中)**:
    *   创建游戏环境实例 (`Game`)，例如 `CartPole()`, `Atari("Pong")` 等，注意符合Game接口的开闭原则。
    *   创建 `MuZeroNetwork` 实例。该网络实例将用于 MCTS 中的模型推理。
    *   获取对主进程中 `SharedStorage` 的代理引用 (例如通过 `multiprocessing.Manager`)。
    *   获取与 `ReplayBuffer` 子进程通信的队列或管道。

2.  **游戏循环 (直到游戏结束 `done` 为 True)**:
    *   **加载/同步模型**:
        *   在每局游戏开始前（或根据配置每 N 步），通过 `SharedStorage` 代理调用 `get_weights()` 获取最新的网络权重。
        *   将获取到的权重加载到本地的 `MuZeroNetwork` 实例中 (`model.set_weights(weights)`)。
        *   (可选) 可以调用 `SharedStorage.wait_for_training_step()` 来等待训练器完成一定步数的训练后再获取新模型，以确保使用的是较新的模型。
    *   **重置游戏环境**: 调用 `game.reset()` 开始新的一局，获取初始观察 `observation`。
    *   **初始化 GameHistory**: 创建一个新的 `GameHistory` 对象来存储本局游戏的数据。
    *   **进行游戏步骤 (直到 `game.terminal()` 为 True)**:
        *   **获取当前状态**: 从 `Game` 环境获取当前观察 (`observation = game.get_observation()`)。
        *   **MCTS 搜索**:
            *   构建 MCTS 树的根节点 (`Node(prior=1.0, to_play=game.to_play())`)。
            *   使用本地 `MuZeroNetwork` 的 `initial_inference(observation_tensor)` 获取初始的 `encoded_state`, `reward_logits`, `policy_logits`, `value_logits`。其中 `observation_tensor` 是 `observation` 经过预处理（如堆叠、转换类型、调整维度、归一化）并转换为 `torch.Tensor` 的结果。
            *   执行 MCTS 搜索 (`run_mcts(config, root, action_space, model, legal_actions, to_play, add_exploration_noise, observation)`) 指定次数 (`config.num_simulations`)。在搜索过程中，树的扩展依赖 `MuZeroNetwork` 的 `recurrent_inference(encoded_state, action_tensor)`。
        *   **选择动作**:
            *   根据 MCTS 搜索结果 (`root` 节点) 和温度参数进行采样 (`action = select_action(config, num_moves, root, training=True)`) 选择一个动作 `action`。
            *   确保选择的动作在 `game.legal_actions()` 范围内 (MCTS 内部已处理)。
        *   **与环境交互**: 将选择的 `action` 输入到 `Game` 环境的 `step(action)` 方法，获取新的 `observation`, `reward`, `terminated` (done) 标志, `next_to_play`。
        *   **存储数据到 GameHistory**:
            *   调用 `game_history.store_search_statistics(root_value, child_visits)` 存储 MCTS 根节点的价值 (`root.value()`) 和子节点的访问计数 (从 `root.children` 提取)。
            *   调用 `game_history.append(action, observation, reward, terminated, game.to_play())` 存储动作、观察、奖励、结束标志和当前玩家。

3.  **游戏结束后**:
    *   (可选) 计算并存储目标价值 (例如，使用 n-step returns 或 discounted returns，这部分逻辑在 `GameHistory.make_target` 中实现)。
    *   将完整的 `GameHistory` 对象通过进程间通信机制 (如队列) 发送给 `run_selfplay` 主进程，由主进程再转发给 `ReplayBuffer`。

## 4. 组件接口兼容性与数据流

*   **`Game` -> `MuZeroNetwork` / `MCTS`**:
    *   `game.reset()`:
        *   **输出**: `observation: np.ndarray` (初始观察状态)。
    *   `game.step(action: int)`:
        *   **输入**: `action: int` (要执行的动作的索引)。
        *   **输出**: `Tuple[np.ndarray, float, bool, int]` (新的 `observation`, `reward`, `terminated`, `to_play` 下一个玩家)。
    *   `game.get_observation()`:
        *   **输出**: `observation: np.ndarray` (当前观察状态)。
    *   `game.legal_actions()`:
        *   **输出**: `List[int]` (当前状态下的合法动作列表)。此列表用于 `run_mcts` 中的 `legal_actions` 参数，并在 `Node.expand` 时过滤动作。
    *   `game.to_play()`:
        *   **输出**: `int` (当前轮到的玩家索引)。用于 `run_mcts` 中的 `to_play` 参数和 `Node` 初始化。
    *   `observation` (来自 `reset`, `step`, `get_observation`) 需要符合 `MuZeroNetwork.initial_inference()` 输入 `observation` 的形状和类型要求 (通常是 `torch.Tensor`)。这可能涉及到：
        *   数据类型转换 (e.g., `np.uint8` to `torch.float32`)。
        *   归一化 (e.g., 图像像素值 / 255.0)。
        *   维度调整 (e.g., HWC to CHW for PyTorch ConvNets, `unsqueeze(0)` 添加 batch 维度)。
        *   使用 `GameHistory.get_stacked_observations(index: int, num_stacked_observations: int)` 进行历史帧堆叠，以满足网络输入要求。

*   **`MuZeroNetwork` -> `MCTS`**:
    *   `model.initial_inference(observation: torch.Tensor)`:
        *   **输入**: `observation: torch.Tensor` (堆叠和预处理后的观察状态, e.g., `(batch_size, C', H, W)` or `(batch_size, D')`)。
        *   **输出**: `Dict[str, torch.Tensor]` 包含:
            *   `'value_logits': torch.Tensor` (预测的价值 logits)。
            *   `'reward_logits': torch.Tensor` (初始奖励 logits, 通常为0对应的 support)。
            *   `'policy_logits': torch.Tensor` (预测的策略 logits)。
            *   `'encoded_state': torch.Tensor` (初始编码状态)。
        *   这些输出用于 `run_mcts` 中初始化根节点 (`root.expand`)。
    *   `model.recurrent_inference(encoded_state: torch.Tensor, action: torch.Tensor)`:
        *   **输入**: `encoded_state: torch.Tensor` (当前编码状态), `action: torch.Tensor` (采取的动作，整数索引, e.g. `(batch_size, 1)`)。
        *   **输出**: `Dict[str, torch.Tensor]` 包含:
            *   `'value_logits': torch.Tensor`。
            *   `'reward_logits': torch.Tensor`。
            *   `'policy_logits': torch.Tensor`。
            *   `'encoded_state': torch.Tensor` (下一编码状态)。
        *   这些输出用于 `run_mcts` 中扩展 MCTS 树节点 (`node.expand`)。

*   **`MCTS` -> `play_game`**:
    *   `run_mcts(config: MuZeroConfig, root: Node, action_space: List[int], model: MuZeroNetwork, legal_actions: List[int], to_play: Player, add_exploration_noise: bool, observation: np.ndarray)`:
        *   **输出**: `Tuple[Node, Dict]` (更新后的根节点 `root`, `extra_info`)。
    *   `select_action(config: MuZeroConfig, num_moves: int, node: Node, training: bool = True)`:
        *   **输入**: `node: Node` (MCTS 搜索后的根节点), `num_moves: int` (当前游戏步数)。
        *   **输出**: `action: int` (选择的动作)。此动作需要是 `game.step()` 接受的格式。
    *   MCTS 根节点 (`root`) 的信息用于 `GameHistory`:
        *   `root.value()`: 根节点的平均价值，用于 `game_history.store_search_statistics()` 的 `root_value` 参数。
        *   `{action: child.visit_count for action, child in root.children.items()}`: 子节点的访问计数，用于 `game_history.store_search_statistics()` 的 `child_visits` 参数。

*   **`play_game` (`GameHistory`) -> `ReplayBuffer`**:
    *   `replay_buffer.save_game(game_history: GameHistory, shared_storage=None)`:
        *   **输入**: `game_history: GameHistory` 对象。该对象需要包含以下属性 (由 `GameHistory` 类定义):
            *   `observation_history: List[np.ndarray]`
            *   `action_history: List[int]`
            *   `reward_history: List[float]`
            *   `to_play_history: List[int]`
            *   `child_visits: List[Dict[int, int]]` (MCTS 搜索的目标策略)
            *   `root_values: List[float]` (MCTS 搜索的根节点价值)
            *   `config: MuZeroConfig`
            *   (可选) `priorities: np.ndarray`
            *   (可选) `game_priority: float`

*   **`SharedStorage` -> `play_game` (`MuZeroNetwork`)**:
    *   `shared_storage.get_weights()`:
        *   **输出**: `Dict` (网络权重, `state_dict` 格式)。
    *   `model.set_weights(weights: Dict)`:
        *   **输入**: `weights: Dict` (从 `shared_storage.get_weights()` 获取的权重)。该方法内部调用 `model.load_state_dict(weights)`。
    *   `shared_storage.wait_for_training_step(target_step: int, timeout: Optional[float] = None)`:
        *   **输入**: `target_step: int` (目标训练步数)。
        *   **输出**: `int` (实际达到的训练步数)。

## 5. 并行化与进程管理

*   **`play_game` Actors**: 每个 `play_game` 实例在独立的**子进程**中运行，以实现并行自我对弈，充分利用多核 CPU。
    *   **实现**: 使用 Python 的 `multiprocessing` 模块 (`Process`, `Queue`, `Manager`)。
*   **`ReplayBuffer`**: 运行在独立的**子进程**中，通过队列接收来自 `run_selfplay` 的 `GameHistory` 对象。
*   **`SharedStorage`**: 实例化于**主进程**中，通过 `multiprocessing.Manager().Namespace()` 或注册自定义类的方式，使其方法可以被 `play_game` 子进程远程调用。内部使用 `threading.RLock` 保证线程安全（因为 Manager 的代理调用可能在主进程中由不同线程处理）。
*   **`run_selfplay`**: 在主进程中运行，负责启动、监控和协调所有子进程。

## 6. 配置与参数

自我对弈的许多行为应通过 `MuZeroConfig` 进行配置，例如：

*   `num_actors`: 并行 `play_game` Actor 的数量。
*   `num_simulations`: MCTS 搜索的模拟次数。
*   `visit_softmax_temperature_fn`: 一个函数，根据游戏步数返回 MCTS 动作选择中的温度参数。
*   `sync_model_interval_steps` (或 `sync_model_interval_games`): `play_game` Actor 从 `SharedStorage` 同步模型的频率。
*   `stacked_observations`: 用于 `GameHistory.get_stacked_observations()` 的历史帧数量。
*   `action_space_size`: 动作空间大小。
*   `td_steps`: TD学习的步数，用于 `GameHistory.make_target()`。
*   `discount`: 折扣因子，用于 `GameHistory.make_target()` 和 MCTS 中的价值回溯。

## 7. 错误处理与日志

*   在 `play_game` Actor 中捕获异常，并通过日志记录或进程间通信报告给主进程。
*   主进程应能处理 Actor 异常退出的情况，例如重新启动 Actor。
*   详细的日志记录对于调试分布式自我对弈过程至关重要。应包含以下关键信息：
    *   **常规日志**: Actor 启动/停止，模型同步事件，游戏开始/结束。
    *   **MCTS 监控**: 
        *   每 N 步或每局游戏结束时，记录关键的 MCTS 统计数据，例如：
            *   根节点的价值 (`root.value()`)。
            *   选择的动作及其对应的访问计数和策略概率。
            *   MCTS 搜索深度和广度的一些指标（例如，平均访问节点数，最大深度）。
            *   (可选) 价值和策略网络输出的熵，以监控探索程度。
    *   **游戏奖励与表现监控**:
        *   每局游戏结束时，记录该局的总奖励、游戏步数。
        *   记录实际获得的奖励 (`reward`) 与模型预测的奖励 (`model.recurrent_inference` 输出中的 `reward_logits` 解码后的值) 之间的差异或相关性，以评估奖励模型的准确性。
        *   记录模型预测的价值 (`model.initial_inference` 或 `model.recurrent_inference` 输出中的 `value_logits` 解码后的值) 与游戏最终结果或 n-step return 之间的差异，以评估价值模型的准确性。
    *   **错误与警告**: 任何在游戏进行、MCTS 搜索、模型推理或数据存储过程中发生的异常或警告信息，包括堆栈跟踪。
    *   **性能指标**: (可选) MCTS 每次搜索耗时，模型推理耗时，游戏步耗时等。

## 8. 总结

该设计通过将计算密集型的 `play_game` 和数据处理密集型的 `ReplayBuffer` 作为独立子进程运行，并将状态同步中心 `SharedStorage` 置于主进程但支持并发访问，旨在构建一个高效、可扩展且相对易于管理的自我对弈系统。清晰的组件职责和接口定义是确保系统稳定运行的关键。