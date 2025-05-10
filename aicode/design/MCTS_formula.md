# MCTS (Monte Carlo Tree Search) 公式解析

本文档列举了 MuZero 实现中 MCTS 算法所使用的核心公式，并提供了简要解释。

## 1. UCB Score (Upper Confidence Bound for Trees)

UCB 分数用于在 MCTS 的选择阶段平衡探索（exploration）和利用（exploitation），即选择哪个子节点进行下一步模拟。

### 1.1. PUCT 常数 (pb_c)

`pb_c = (log((N(p) + C_base + 1) / C_base) + C_init) * sqrt(N(p)) / (N(c) + 1)`

*   **解释**:
    *   `N(p)`: 父节点的访问次数。
    *   `N(c)`: 子节点的访问次数。
    *   `C_base`, `C_init`: 控制探索程度的超参数。
    *   此项确保了访问次数较少的子节点或者父节点访问次数相对于子节点访问次数更多的子节点，会获得更高的探索奖励。

### 1.2. 先验分数 (Prior Score) - 探索项

`PriorScore(c) = pb_c * P(c)`

*   **解释**:
    *   `P(c)`: 子节点 `c` 的先验概率（通常由策略网络输出）。
    *   此项鼓励选择那些具有较高先验概率但可能尚未被充分探索的动作。

### 1.3. 价值分数 (Value Score) - 利用项

此分数代表了子节点的已知价值。

*   **单人游戏**:
    `ValueScore(c) = normalize(R(c) + γ * V(c))`
    *   `R(c)`: 到达子节点 `c` 后获得的即时奖励。
    *   `V(c)`: 子节点 `c` 的估计价值（即 `Q_sum(c) / N(c)`）。
    *   `γ`: 折扣因子，用于衡量未来奖励的重要性。
    *   `normalize(...)`: 将计算出的价值归一化到特定范围（例如 [0, 1]），以平衡不同幅度的值。

*   **双人游戏 (零和博弈)**:
    `ValueScore(c) = normalize(R(c) + γ * (-V(c)))`
    *   解释同单人游戏，但子节点的价值 `V(c)` 取负值。这是因为在零和博弈中，对手的收益即为我方的损失，所以从当前玩家的角度看，子节点的价值需要反转。

### 1.4. 总 UCB 分数

`UCB(c) = PriorScore(c) + ValueScore(c)`

*   **解释**:
    *   MCTS 在选择子节点时，会选择具有最高 UCB 分数的节点。

## 2. 价值回溯 (Backpropagation)

在一次模拟结束后，从模拟路径的叶节点开始，向上更新路径上所有节点的统计信息（访问次数和累积价值）。

### 2.1. 节点价值 V(s)

`V(s) = Q_sum(s) / N(s)` (当 `N(s) > 0` 时)

*   **解释**:
    *   `Q_sum(s)`: 节点 `s` 的累积价值总和（所有经过该节点的模拟路径所获得的未来奖励之和，根据玩家视角调整）。
    *   `N(s)`: 节点 `s` 的访问次数。
    *   这是节点 `s` 的平均观察价值。

### 2.2. 价值更新

令 `G` 为从当前模拟路径获得的（可能是递归计算的）未来累积折扣奖励。

*   **单人游戏**:
    *   更新节点的累积价值: `Q_sum_new(n) = Q_sum_old(n) + G_from_child`
    *   递归计算回溯价值: `G_for_parent(n) = R(n) + γ * G_from_child`
    *   **解释**:
        *   `Q_sum_old(n)`: 节点 `n` 更新前的累积价值。
        *   `G_from_child`: 从子节点回溯上来的价值。
        *   `R(n)`: 到达节点 `n` 时获得的奖励。
        *   路径上每个节点的累积价值会加上其子节点回溯上来的价值。回溯到父节点的价值是当前节点的奖励加上其子节点回溯价值的折扣和。

*   **双人游戏 (零和博弈)**:
    *   更新节点的累积价值:
        `Q_sum_new(n) = Q_sum_old(n) + (G_from_child if player(n) == current_turn_player else -G_from_child)`
    *   递归计算回溯价值:
        `G_for_parent(n) = (R_adjusted_for_player(n)) + γ * G_from_child`
        (其中 `R_adjusted_for_player(n)` 为 `-R(n)` 如果 `player(n) == current_turn_player`，否则为 `R(n)`)
    *   **解释**:
        *   `player(n)`: 节点 `n` 对应的玩家。
        *   `current_turn_player`: 当前回溯价值对应的玩家视角。
        *   累积价值的更新：如果节点 `n` 的玩家与当前回溯价值的视角玩家相同，则加上 `G_from_child`；否则，减去（因为是对手的价值）。
        *   递归价值的计算：奖励 `R(n)` 的符号会根据 `player(n)` 和 `current_turn_player` 是否一致进行调整。如果一致，意味着这个奖励是“对方”获得的，所以对当前玩家是负向的。

### 2.3 用于归一化的 MinMax 统计 (MinMax Statistics for Normalization)

`min_max_stats` 对象用于追踪 MCTS 搜索过程中遇到的特定值的最小和最大范围，这些统计数据随后用于归一化计算（例如在 `ValueScore` 中的 `normalize(...)` 函数）。在反向传播阶段，会使用以下公式更新这些统计数据：

*   **单人游戏**:
    `Value_tracked = R(n) + γ * V(n)`
    *   **解释**:
        *   `R(n)`: 节点 `n` 的即时奖励。
        *   `V(n)`: 节点 `n` 的估计价值 (即 `Q_sum(n) / N(n)`)。
        *   `γ`: 折扣因子。
        *   此值是节点 `n` 的原始（未归一化）Q值估计，用于更新 MinMax 统计。

*   **双人游戏 (零和博弈)**:
    `Value_tracked = R(n) + γ * (-V(n))`
    *   **解释**:
        *   解释同单人游戏，但节点 `n` 的价值 `V(n)` 取负值，以反映从当前玩家角度看的对手价值。
        *   此值是节点 `n` 从当前玩家角度看的原始（未归一化）Q值估计，用于更新 MinMax 统计。
