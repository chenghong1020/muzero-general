# design目录下各设计文件介绍
## design_structure.md
- 整体架构 : 描述了 MuZero 算法的四大核心组件：共享存储 (Shared Storage)、自我对弈 (Self-Play)、回放缓冲区 (Replay Buffer) 和训练器 (Trainer)，以及它们之间的交互流程（通过 Mermaid 图展示）。
- 数据模型 : 详细定义了各个组件中使用的数据结构：
- ReplayBuffer : 说明了存储的游戏历史数据 ( GameHistory ) 结构（包含观察、动作、奖励、玩家、MCTS统计信息等），以及输出给训练器的批处理数据 ( TrainingBatch ) 格式（包含 observation, action, target_value, target_reward, target_policy, weight, gradient_scale 等张量）。
- SharedStorage : 定义了用于存储网络权重和训练统计信息的数据结构。
- Model : 描述了模型内部的三个核心网络（表征网络 Representation, 动态网络 Dynamics, 预测网络 Prediction）的输入输出，以及推理过程（初始推理和循环推理）。还提到了为保持训练稳定性的措施（如状态归一化）。
- Game : 定义了游戏环境的基本交互方式，并展示了 Game 与 Model 、 Game 与 ReplayBuffer 之间的数据流转和格式。

## design_interface.md
- 接口定义 : 提供了核心 Python 类的接口定义（方法签名和文档字符串），明确了各个组件交互时所需遵循的规范。
- 涵盖的类 :
- MuZeroNetwork ( `MuZeroNetwork` ): 定义了模型的 initial_inference 和 recurrent_inference 方法。
- Game ( `Game` ): 定义了 reset , step , to_play , legal_actions 等与环境交互的方法。
- ReplayBuffer ( `ReplayBuffer` ): 定义了 save_game 和 sample_batch 方法。
- SharedStorage ( `SharedStorage` ): 定义了 set_info 和 get_info 方法。
- GameHistory ( `GameHistory` ): 定义了 store_search_statistics 和 get_stacked_observations 方法。

## design_MCTS.md
- MCTS 算法 : 详细阐述了蒙特卡洛树搜索 (MCTS) 的过程。
- 流程图 : 使用 Mermaid 图展示了 MCTS 的完整流程，包括选择 (Selection)、扩展 (Expansion)、反向传播 (Backpropagation) 等阶段，以及 UCB 分数计算、探索噪声添加、温度参数控制等细节。
- 数据结构 : 定义了 MCTS 过程中使用的核心数据结构：
- Node ( `Node` ): 树节点结构，包含访问次数、先验概率、价值总和、子节点、隐藏状态等信息。
- MinMaxStats ( `MinMaxStats` ): 用于归一化节点价值。
- SearchPath ( `SearchPath` ): 搜索路径记录。
- MCTSConfig ( `MCTSConfig` ): MCTS 算法的配置参数。
- MCTSResult ( `MCTSResult` ): MCTS 搜索的结果。

## design_config.md
MuZeroConfig ( `MuZeroConfig` ) 的 Python 类。

这个类的作用是 集中管理 MuZero 算法的所有超参数和配置选项 。它将参数分成了几大类，例如游戏环境、网络结构、MCTS 搜索、训练过程、优先经验回放、自博弈、评估和存储等。

此外，该类还包含了一些辅助方法，比如根据训练步数动态调整 MCTS 温度的函数 ( `MuZeroConfig.visit_softmax_temperature` )，以及用于获取 Atari 或棋盘游戏预设配置的静态方法 ( `MuZeroConfig.get_atari_config` , `MuZeroConfig.get_board_game_config` )。

总的来说，这个文件提供了一个结构化、可配置的方式来定义和管理 MuZero 实验的各种设置。

## design_model.md
- 模型架构 : 详细描述了 MuZero 算法的核心网络架构，包括表征网络、动态网络和预测网络。
- 网络结构 : 提供了每个网络的详细结构，包括网络类型、输入输出维度、网络层数、激活函数等。
- 网络参数 : 列出了每个网络的参数数量，以及网络的总参数量。


