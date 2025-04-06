# Structure
There are four components which are classes that run simultaneously in a dedicated thread. The shared storage holds the latest neural network weights, the self-play uses those weights to generate self-play games and store them in the replay buffer. Finally, those played games are used to train a network and store the weights in the shared storage. The circle is complete.

Those components are launched and managed from the MuZero class in muzero.py and the structure of the neural network is defined in models.py.

MuZero 通过以下几个主要步骤实现强化学习训练循环：
共享存储（Shared Storage）：存放最新的神经网络权重参数。这是整个系统的参数中心，存储着当前最优的模型参数，为其他组件提供模型基础。
自我对弈（Self - Play）：从共享存储中加载最新的神经网络权重，然后利用这些权重来进行自我对弈。在对弈过程中，生成大量的对弈游戏数据，这些数据包含了不同状态、采取的行动以及相应结果等信息。自我对弈过程利用多个 CPU 并行加速，将生成的游戏历史数据存储到回放缓冲区（Replay Buffer）中。
回放缓冲区（Replay Buffer）：存储自我对弈生成的游戏数据。训练器（Trainer）会从回放缓冲区中获取训练批次数据，用于训练神经网络。
训练器（Trainer）：使用 GPU 从回放缓冲区获取训练数据批次，对神经网络进行训练。训练完成后，将更新后的模型权重保存回共享存储。如此循环，不断优化神经网络。

# Network example 
Here is the diagram of the muzero network applied to the Atari game:

这是应用于Atari游戏的MuZero网络（ResNet版本）示意图 ，其工作原理如下：

### 输入部分
游戏画面以RGB图像形式输入，经过处理得到观测值（Observation）。将多个观测值堆叠（Stacked observations），并结合之前的观测值与动作（Previous observation + action），形成网络的输入。 

### Representation（表征）模块
- **降采样（Downsample）**：先将输入从\(96×96×128\)（\(128\)帧，\(3\)种颜色通道，\(32\)个动作相关特征 ）降采样到\(6×6×num\_channels\)。
- **ResNet处理**：通过ResNet网络进一步处理，对每个通道进行归一化到\([0, 1]\) 。此模块输出编码状态（Encoded state），并给出奖励支持（Reward support），初始奖励设为\(0\) 。

### Prediction（预测）模块（初次）
- **价值网络（Value network）**：将编码状态展平（Flatten）后，经全连接层（fc\_value\_layers）处理，输出价值支持（Value support），再转换为标量得到价值（Value） 。
- **策略网络（Policy network）**：先通过\(1×1\)卷积（Conv \(1×1\)）将通道数调整，展平后经全连接层（fc\_policy\_layers）处理，输出动作空间支持（Action space support），经Softmax函数得到策略（Policy），用于选择动作（Action） 。

### Dynamics（动态）模块
- **编码处理**：对选择的动作进行独热编码（One - hot encoding），与编码状态结合 ，经\(1×1\)卷积调整通道数后，通过ResNet网络处理。
- **奖励网络（Reward network）**：处理后经\(1×1\)卷积和全连接层（fc\_reward\_layers），输出奖励支持（Reward support），转换为标量得到奖励（Reward） 。同时，输出下一个编码状态（Next encoded state） 。

### Prediction（预测）模块（循环）
根据动态模块输出的编码状态，再次通过价值网络和策略网络，重复上述价值和策略计算过程 ，进行循环推理（Recurrent inference） 。每次循环都基于前一时刻的状态和动作，不断更新价值、奖励预测以及策略选择，以适应游戏状态变化，优化游戏决策。 


# Class Design
以下是各个类的作用和逻辑简要描述：

### 1. `MinMaxStats`类
 - **作用**：用于记录树结构中值的最小值和最大值，主要在蒙特卡洛树搜索（MCTS）过程中，帮助归一化节点价值，以便更合理地评估节点。
 - **逻辑**：初始化时可传入已知边界`KnownBounds`来设定初始的最大最小值，通过`update`方法更新当前记录的最大最小值，`normalize`方法用于在有实际最大最小值时对给定值进行归一化处理。 

### 2. `MuZeroConfig`类
 - **作用**：存储MuZero算法的各种配置参数，涵盖自博弈（Self - Play）和训练（Training）两部分的参数设置，是整个算法运行的参数基石。
 - **逻辑**：通过构造函数初始化一系列参数，如动作空间大小、最大移动步数、折扣因子、狄利克雷噪声参数、模拟次数、批量大小等。同时提供针对不同游戏（如棋盘游戏、围棋、象棋、将棋、Atari游戏等）的配置生成方法，根据不同游戏特点设置合适的参数。

### 3. `Action`类
 - **作用**：表示游戏中的动作，为动作提供一种简单的数值化索引表示方式。
 - **逻辑**：通过构造函数接收动作的索引值，定义了哈希、相等比较和大于比较等方法，方便在数据结构（如字典、集合）中对动作进行操作和比较。

### 4. `Player`类
 - **作用**：目前是一个空类，可能预留用于表示游戏中的玩家相关属性或行为，在当前代码中未体现具体逻辑。

### 5. `Node`类
 - **作用**：代表蒙特卡洛树搜索中的节点，存储节点相关的各种信息，是构建搜索树的基本单元。
 - **逻辑**：初始化时记录先验概率，在搜索过程中，通过属性记录访问次数、当前轮到哪个玩家行动、价值总和、子节点集合、隐藏状态以及奖励等信息。`expanded`方法判断节点是否已扩展（是否有子节点），`value`方法计算节点的平均价值。

### 6. `ActionHistory`类
 - **作用**：在搜索过程中，简单记录执行过的动作历史，便于追踪游戏进程和动作序列。
 - **逻辑**：构造函数接收动作历史列表和动作空间大小，提供克隆历史记录、添加动作、获取最后一个动作、获取动作空间以及确定当前玩家等方法。

### 7. `Environment`类
 - **作用**：表示MuZero与之交互的游戏环境，定义了环境与智能体交互的基本接口。
 - **逻辑**：目前仅定义了`step`方法，用于执行动作并返回奖励，但具体实现为空，需根据实际游戏环境进行具体编写。

### 8. `Game`类
 - **作用**：代表与环境交互的单个游戏回合，负责管理游戏过程中的各种信息和操作。
 - **逻辑**：初始化时创建游戏环境，记录动作历史、奖励、子节点访问情况、根节点价值等。包含判断游戏是否结束（`terminal`方法）、获取合法动作（`legal_actions`方法）、执行动作（`apply`方法）、存储搜索统计信息（`store_search_statistics`方法）、生成游戏图像特征（`make_image`方法）、生成训练目标（`make_target`方法）等功能，用于支持自博弈和训练过程。

### 9. `ReplayBuffer`类
 - **作用**：作为经验回放缓冲区，存储自博弈生成的游戏数据，为训练提供数据样本。
 - **逻辑**：初始化时设定缓冲区大小和批量大小，`save_game`方法用于保存游戏数据，当缓冲区满时按先进先出原则移除最早的数据。`sample_batch`方法从缓冲区中采样一批游戏数据用于训练，采样时需指定展开步数和时间差分步数，内部通过`sample_game`和`sample_position`方法获取具体的游戏和位置，但这两个方法目前只是简单返回默认值，实际应用中需按特定策略采样。

### 10. `NetworkOutput`类
 - **作用**：以具名元组的形式，封装神经网络的输出，包括价值、奖励、策略对数几率和隐藏状态。
 - **逻辑**：方便在代码中统一管理和传递神经网络的输出结果，使代码结构更清晰。

### 11. `Network`类
 - **作用**：定义神经网络相关的操作，是算法与神经网络交互的接口。
 - **逻辑**：包含`initial_inference`方法用于初始推理（结合表征和预测功能），`recurrent_inference`方法用于循环推理（结合动态和预测功能），`get_weights`方法获取网络权重，`training_steps`方法获取网络已训练的步数，但目前这些方法只是返回默认值，实际需根据具体神经网络实现。

### 12. `SharedStorage`类
 - **作用**：作为共享存储，保存最新的神经网络权重，实现网络模型在不同组件间的共享。
 - **逻辑**：通过字典存储不同训练步数对应的网络模型，`latest_network`方法返回最新的网络模型，`save_network`方法保存网络模型及其对应的训练步数。

结合整体工作思路：
 - **共享存储**：`SharedStorage`类负责保存最新的神经网络权重，是整个系统中模型参数的存储中心 。
 - **自博弈**：`MuZeroConfig`类配置自博弈相关参数，`run_selfplay`函数不断从`SharedStorage`获取最新网络，通过`play_game`函数进行自博弈。`play_game`中利用`Game`类管理游戏进程，通过`run_mcts`执行蒙特卡洛树搜索，`run_mcts`中`Node`类构建搜索树，`ActionHistory`记录动作历史，搜索结束后选择动作并更新游戏状态，最终将游戏数据保存到`ReplayBuffer` 。
 - **训练**：`train_network`函数从`ReplayBuffer`采样游戏数据，利用`Network`类对神经网络进行训练，训练过程中计算损失并更新权重，训练后的网络模型再保存回`SharedStorage` 。如此循环，实现了从共享存储获取模型 -> 自博弈生成数据 -> 训练更新模型 -> 保存回共享存储的闭环。 
