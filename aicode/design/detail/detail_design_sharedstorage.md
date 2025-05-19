# MuZero SharedStorage 详细设计

## 1. 概述

SharedStorage 是 MuZero 算法中的关键组件，负责存储和同步最新的网络权重及训练统计信息。它作为训练器和自博弈 Actor 之间的桥梁，确保所有组件能够访问到最新的模型和训练数据。

## 2. 设计目标

- 提供高效的模型权重存储和检索机制
- 支持训练统计信息的记录和查询
- 确保在分布式环境中的数据一致性
- 支持模型版本管理和检查点保存

## 3. 数据结构

SharedStorage 主要管理以下数据：

1. **网络权重**：MuZero 神经网络的最新参数
2. **训练统计信息**：包括训练步数、奖励、损失等指标
3. **模型版本**：用于追踪模型更新

- weights 对象具体类型是一个嵌套字典结构，用于存储神经网络的参数
```python 
weights = {
    "representation": representation_network_weights,  # 表征网络的权重
    "dynamics": dynamics_network_weights,             # 动态网络的权重
    "prediction": prediction_network_weights          # 预测网络的权重
}
```
其中：

- representation_network_weights 是表征网络的参数，通常是 PyTorch 的 torch.Tensor 对象或包含多个张量的字典/列表
- dynamics_network_weights 是动态网络的参数，同样是 PyTorch 张量或其集合
- prediction_network_weights 是预测网络的参数，结构与上述类似


```python
class SharedStorage:
    def __init__(self, checkpoint, config):
        """
        初始化共享存储。
        
        参数:
            checkpoint: 包含初始网络权重和统计信息的字典
            config: MuZero配置对象
        """
        self.config = config
        self._checkpoint = copy.deepcopy(checkpoint)
        self._model_version = 0
```

## 4. 核心接口

### 4.1 检查点管理

```python
def save_checkpoint(self, path=None):
    """
    保存当前检查点到指定路径。
    
    参数:
        path: 保存路径，默认为配置中的结果路径
    
    返回:
        保存的文件路径
    """
    
def get_checkpoint(self):
    """
    获取完整的检查点数据。
    
    返回:
        包含所有权重和统计信息的检查点字典
    """
```

### 4.2 权重管理

```python
def get_weights(self):
    """
    获取最新的网络权重。
    
    返回:
        最新的网络权重
    """
    
def get_weights_version(self):
    """
    获取当前权重的版本号。
    
    返回:
        权重版本号
    """
    
def set_weights(self, weights):
    """
    更新网络权重。
    
    参数:
        weights: 新的网络权重
    """
```

### 4.3 统计信息管理
SharedStorage 组件存储的训练统计信息主要包括以下内容：
- training_step ：当前训练步数，表示模型已经训练了多少步
- total_reward ：累计奖励，反映模型在自博弈中获得的平均奖励
- mean_value ：平均价值，表示模型对状态价值的平均预测
- lr ：当前学习率，训练过程中可能会动态调整
- total_loss ：总损失值，反映模型整体训练情况
- value_loss ：价值网络的损失，反映价值预测的准确性
- policy_loss ：策略网络的损失，反映动作选择的准确性

- 可以结合可视化工具（如 TensorBoard、Matplotlib）将这些统计信息以图表形式展示，使训练监测更加直观和高效。

```python
def get_info(self, keys):
    """
    获取指定的统计信息。
    
    参数:
        keys: 字符串或字符串列表，表示要获取的信息键
        
    返回:
        单个值或包含请求键值对的字典
    """
    
def set_info(self, key, value):
    """
    更新指定的统计信息。
    
    参数:
        key: 要更新的信息键
        value: 新值
    """
    
def update_info(self, update_dict):
    """
    批量更新多个统计信息。
    
    参数:
        update_dict: 包含要更新的键值对的字典
    """
```

### 4.4 同步机制
- 这个接口允许自博弈 Actor 等待直到训练步数达到指定的目标值，主要用于实现训练器和自博弈 Actor 之间的模型同步

```python
def wait_for_training_step(self, target_step, timeout=None):
    """
    等待直到训练步数达到目标值。
    
    用于自博弈Actor等待新模型时的同步。
    
    参数:
        target_step: 目标训练步数
        timeout: 超时时间（秒），None表示无限等待
        
    返回:
        达到目标步数时的训练步数
    """
```

## 5. 分布式支持

为了支持多线程/多进程环境，SharedStorage 将使用 Python 的 concurrent.futures 和 threading 模块
- 使用 threading.RLock 实现线程安全
- 使用 concurrent.futures 实现并行处理
- 暂不使用Ray框架

## 6. 训练器更新模型

```python
# 获取当前模型权重
current_weights = {
    "representation": representation_model.state_dict(),
    "dynamics": dynamics_model.state_dict(),
    "prediction": prediction_model.state_dict()
}

# 更新共享存储中的权重
shared_storage.set_weights.remote(current_weights)


shared_storage.set_info.remote("training_step", current_step)

# 更新训练统计信息
shared_storage.update_info.remote({
    "total_reward": avg_reward,
    "mean_value": mean_value,
    "training_step": steps,
    "lr": current_lr,
    "total_loss": total_loss,
    "value_loss": value_loss,
    "policy_loss": policy_loss
})
```

## 7 自博弈Actor获取最新模型

```python
# 自博弈Actor获取最新权重
# 获取最新权重
weights = get(shared_storage.get_weights.remote())

# 加载到模型
representation_model.load_state_dict(weights["representation"])
dynamics_model.load_state_dict(weights["dynamics"])
prediction_model.load_state_dict(weights["prediction"])
weights = get(shared_storage.get_weights())

# 等待新模型
current_step = get(shared_storage.wait_for_training_step(target_step, timeout=60))
```

## 总结

SharedStorage 组件是 MuZero 算法中连接训练器和自博弈 Actor 的关键桥梁。通过提供高效的模型权重存储和检索机制，以及训练统计信息的记录和查询功能，它确保了分布式环境中的数据一致性和模型同步。

该组件的设计充分考虑了分布式环境的需求，使用 Ray 框架提供的远程类机制，确保在多进程/多节点环境中能够高效地共享和同步数据。同时，通过版本管理和检查点保存功能，提供了模型训练过程中的容错和恢复能力。

        