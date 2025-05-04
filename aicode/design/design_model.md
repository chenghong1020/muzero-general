# ResNet版本 Model 设计
本设计主要以应用于Atari游戏的场景，该场景下 MuZero网络（ResNet版本）示意图 ，其工作原理如下：

### 输入部分
- **原始输入**: 游戏画面（Image）通常为 RGB 图像。
- **观测处理**: 原始输入经过预处理（例如灰度化、缩放）得到观测值（Observation）。
- **状态构建**: 将当前观测值与过去的 `k` 个观测值以及 `k` 个动作堆叠起来。根据图示，输入维度为 `96x96x128`，其中 `128` 个通道由 `32` 帧历史观测（每帧 `3` 个颜色通道）加上 `32` 个历史动作（通常进行编码，例如 one-hot 平面）组成。
    - **输入维度**: `(Batch, 128, 96, 96)` （假设通道优先）

### Representation（表征）模块 `h`
此模块将高维的观测历史压缩成一个低维的隐藏状态（Encoded State）。
- **网络类型**: 卷积网络 (Downsample) + ResNet 块。
- **输入**: 堆叠的观测历史 `(Batch, 128, 96, 96)`。
- **处理**:
    1.  **降采样 (Downsample)**: 通过卷积层将输入 `96x96x128` 降采样到 `6x6xnum_channels`。
    2.  **ResNet 处理**: 应用一系列 ResNet 块处理降采样后的特征图。
    3.  **归一化**: 对 ResNet 输出的每个通道进行归一化，将值缩放到 `[0, 1]` 区间。
- **输出**:
    - **编码状态 (Encoded State)**: `(Batch, num_channels, 6, 6)`。
    - **初始奖励 (Initial Reward)**: 对于初始推理，奖励固定为 0。输出奖励支持向量 (Reward support)，其标量值为 0。

### Prediction（预测）模块 `f`
此模块根据当前的隐藏状态预测策略和价值。
- **网络类型**: ResNet 块 + 卷积网络 (Conv 1x1) + 全连接网络 (MLP)。
- **输入**: 编码状态 (Encoded State) `(Batch, num_channels, 6, 6)`。
- **处理**:
    1.  **共享 ResNet**: 输入的编码状态首先通过一个 ResNet 块。
    2.  **策略头 (Policy Head)**:
        a.  **通道缩减 (Conv 1x1)**: 使用 `1x1` 卷积调整通道数 (`reduced_channels_policy`)。
        b.  **展平 (Flatten)**: 将 `6x6` 特征图展平成向量。
        c.  **全连接层 (MLP)**: 通过 `fc_policy_layers` 输出策略 Logits。
        d.  **Softmax**: 将 Logits 转换为概率分布。
    3.  **价值头 (Value Head)**:
        a.  **通道缩减 (Conv 1x1)**: 使用 `1x1` 卷积调整通道数 (`reduced_channels_value`)。
        b.  **展平 (Flatten)**: 将 `6x6` 特征图展平成向量。
        c.  **全连接层 (MLP)**: 通过 `fc_value_layers` 输出价值支持向量。
        d.  **支持转标量**: 将价值支持向量转换为标量价值。
- **输出**:
    - **策略 (Policy)**: `(Batch, action_space_size)` - 每个动作的选择概率。
    - **价值 (Value)**: `(Batch, 1)` - 当前状态的预测价值（标量）。
    - **(中间输出)**: Policy Logits `(Batch, action_space_size)`, Value Support `(Batch, full_support_size)`。

### Dynamics（动态）模块 `g`
此模块模拟环境的动态，根据当前隐藏状态和采取的动作预测下一个隐藏状态和奖励。
- **网络类型**: 卷积网络 (Conv 1x1) + ResNet 块 + 卷积网络 (Conv 1x1 for reward) + 全连接网络 (MLP for reward)。
- **输入**:
    - 编码状态 (Encoded State): `(Batch, num_channels, 6, 6)`。
    - 动作 (Action): `(Batch, 1)` - 选择的动作索引。
- **处理**:
    1.  **动作编码**: 将动作索引 `Action` 编码为一个与 `Encoded State` 空间维度相同的平面（例如，每个位置填充 `action_id / action_space_size`），维度为 `(Batch, 1, 6, 6)`。
    2.  **拼接**: 将动作平面与 `Encoded State` 在通道维度上拼接，得到 `(Batch, num_channels + 1, 6, 6)`。
    3.  **通道调整 (Conv 1x1)**: 使用 `1x1` 卷积调整通道数，可能回到 `num_channels`。
    4.  **ResNet 处理**: 应用一系列 ResNet 块处理特征图。
    5.  **归一化**: 对 ResNet 输出进行归一化，得到下一个编码状态。
    6.  **奖励头 (Reward Head)**:
        a.  **通道缩减 (Conv 1x1)**: 对 ResNet 输出使用 `1x1` 卷积调整通道数 (`reduced_channels_reward`)。
        b.  **展平 (Flatten)**: 将 `6x6` 特征图展平成向量。
        c.  **全连接层 (MLP)**: 通过 `fc_reward_layers` 输出奖励支持向量。
        d.  **支持转标量**: 将奖励支持向量转换为标量奖励。
- **输出**:
    - **下一编码状态 (Next Encoded State)**: `(Batch, num_channels, 6, 6)`。
    - **奖励 (Reward)**: `(Batch, 1)` - 预测的即时奖励（标量）。
    - **(中间输出)**: Reward Support `(Batch, full_support_size)`。

### 循环推理 (Recurrent Inference)
在 MCTS（蒙特卡洛树搜索）过程中，重复使用 **Dynamics** 和 **Prediction** 模块进行多步预测：
1.  给定当前状态 `s_k` (Encoded State) 和选择的动作 `a_{k+1}`。
2.  使用 **Dynamics** 模块预测下一状态 `s_{k+1}` (Next Encoded State) 和奖励 `r_{k+1}`。
    - `s_{k+1}, r_{k+1} = g(s_k, a_{k+1})`
3.  使用 **Prediction** 模块基于预测的下一状态 `s_{k+1}` 预测其对应的策略 `p_{k+1}` 和价值 `v_{k+1}`。
    - `p_{k+1}, v_{k+1} = f(s_{k+1})`
这个过程可以迭代进行 `K` 步，模拟未来的轨迹。

---

# Fully Connected (FC) 版本 Model 设计
本设计适用于观测空间较小或可以有效扁平化的环境（例如 CartPole, GridWorld），或者作为处理图像输入的替代方案（将图像和动作平面扁平化）。

### 输入部分
- **原始输入**: 环境的观测值（Observation），可以是向量或需要扁平化的张量（如图像）。
- **状态构建**: 将当前观测值与过去的 `k` (`stacked_observations`) 个观测值以及 `k` 个历史动作堆叠起来。
    - **输入维度**:
        - 假设原始观测维度为 `observation_shape = (C, H, W)` (图像类) 或 `(D,)` (向量类)。
        - **图像类**: 堆叠后的输入维度为 `(Batch, C' + k, H, W)`，其中 `C' = (stacked_observations + 1) * C` 是观测通道数，`k` 是历史动作数。每个历史动作被编码为一个 `1 x H x W` 的平面。
        - **向量类**: 堆叠后的输入维度为 `(Batch, D' + k * action_space_size)`，其中 `D' = (stacked_observations + 1) * D` 是观测特征维度，`k` 是历史动作数。每个历史动作通常被编码为 one-hot 向量 `(action_space_size,)`。
    - **扁平化**: 输入在送入表征网络前会被扁平化。
        - **图像类**: 扁平化后的维度 `input_dim = (C' + k) * H * W`。
        - **向量类**: 扁平化后的维度 `input_dim = D' + k * action_space_size`。
- **注意**: 这种将动作编码为平面并与图像堆叠后扁平化的方式，是 `models.py` 中 FC 网络处理 3D 观测的一种实现方式。

### Representation（表征）模块 `h`
此模块将（可能高维的）扁平化观测历史压缩成一个低维的隐藏状态。
- **网络类型**: 全连接网络 (MLP)，层结构由 `fc_representation_layers` 定义。
- **输入**: 扁平化的观测历史 `(Batch, input_dim)`。
- **处理**:
    1.  通过 MLP 网络。
    2.  对输出进行归一化，将值缩放到 `[0, 1]` 区间。
- **输出**:
    - **编码状态 (Encoded State)**: `(Batch, encoding_size)`。
    - **初始奖励 (Initial Reward)**: 对于初始推理，奖励固定为 0。输出奖励支持向量 (Reward support)，其标量值为 0。

### Prediction（预测）模块 `f`
此模块根据当前的隐藏状态预测策略和价值。
- **网络类型**: 两个独立的全连接网络 (MLP)。
    - 策略头: 由 `fc_policy_layers` 定义。
    - 价值头: 由 `fc_value_layers` 定义。
- **输入**: 编码状态 (Encoded State) `(Batch, encoding_size)`。
- **处理**:
    1.  **策略头 (Policy Head)**: 输入 `Encoded State`，通过 MLP 输出策略 Logits。
    2.  **价值头 (Value Head)**: 输入 `Encoded State`，通过 MLP 输出价值支持向量。
- **输出**:
    - **策略 Logits (Policy Logits)**: `(Batch, action_space_size)`。
    - **价值支持向量 (Value Support)**: `(Batch, full_support_size)`，其中 `full_support_size = 2 * support_size + 1`。
    - (后续处理将 Logits 转换为概率分布，将支持向量转换为标量价值)。

### Dynamics（动态）模块 `g`
此模块模拟环境的动态，根据当前隐藏状态和采取的动作预测下一个隐藏状态和奖励。
- **网络类型**: 两个独立的全连接网络 (MLP)。
    - 状态预测头: 由 `fc_dynamics_layers` 定义。
    - 奖励预测头: 由 `fc_reward_layers` 定义。
- **输入**:
    - 编码状态 (Encoded State): `(Batch, encoding_size)`。
    - 动作 (Action): `(Batch, 1)` - 选择的动作索引。
- **处理**:
    1.  **动作编码**: 将动作索引 `Action` 转换为 one-hot 编码向量 `(Batch, action_space_size)`。
    2.  **拼接**: 将 `Encoded State` 与 one-hot 动作向量在特征维度上拼接，得到 `(Batch, encoding_size + action_space_size)`。
    3.  **状态预测**: 将拼接后的向量输入状态预测 MLP (`dynamics_encoded_state_network`)，得到未归一化的下一状态。
    4.  **奖励预测**: 将 **未归一化** 的下一状态输入奖励预测 MLP (`dynamics_reward_network`)，得到奖励支持向量。
    5.  **状态归一化**: 对未归一化的下一状态进行归一化，将值缩放到 `[0, 1]` 区间，得到最终的下一编码状态。
- **输出**:
    - **下一编码状态 (Next Encoded State)**: `(Batch, encoding_size)` (已归一化)。
    - **奖励支持向量 (Reward Support)**: `(Batch, full_support_size)`。
    - (后续处理将奖励支持向量转换为标量奖励)。

### 循环推理 (Recurrent Inference)
与 ResNet 版本类似，在 MCTS 过程中，重复使用 **Dynamics** 和 **Prediction** 模块进行多步预测：
1.  给定当前状态 `s_k` (Encoded State) 和选择的动作 `a_{k+1}` (索引)。
2.  将 `a_{k+1}` 转换为 one-hot 编码。
3.  使用 **Dynamics** 模块预测下一状态 `s_{k+1}` (Next Encoded State) 和奖励 `r_{k+1}` (的支持向量)。
    - `s_{k+1}, r_{k+1}_support = g(s_k, a_{k+1}_onehot)`
4.  使用 **Prediction** 模块基于预测的下一状态 `s_{k+1}` 预测其对应的策略 `p_{k+1}` (的 Logits) 和价值 `v_{k+1}` (的支持向量)。
    - `p_{k+1}_logits, v_{k+1}_support = f(s_{k+1})`
这个过程可以迭代进行 `K` 步，模拟未来的轨迹。


