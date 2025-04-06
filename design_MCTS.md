# MCTS 算法
```
graph TD
    A[开始MCTS搜索] --> B[初始化根节点]
    B --> C{是否使用已有根节点?}
    
    C -->|否| D[使用表示网络获取初始状态]
    C -->|是| E[使用提供的根节点]
    
    D --> F[扩展根节点]
    E --> G[添加探索噪声]
    F --> G
    
    G --> H[初始化MinMaxStats]
    H --> I[开始模拟循环]
    
    I --> J[选择阶段]
    J --> K[使用UCB公式选择动作]
    K --> L{节点是否已扩展?}
    
    L -->|是| M[更新树深度<br/>temperature控制]
    L -->|否| N[扩展阶段]
    
    M --> K
    
    N --> O[使用动态网络预测]
    O --> P[扩展新节点]
    P --> Q[反向传播]
    
    Q --> R[更新节点统计]
    R --> S[更新MinMaxStats]
    S --> T{是否达到模拟次数?}
    
    T -->|否| I
    T -->|是| U[结束搜索]
    
    subgraph 探索噪声控制
        G --> G1[生成Dirichlet噪声]
        G1 --> G2[混合先验概率]
    end
    
    subgraph UCB分数计算
        K --> K1[计算探索项pb_c]
        K1 --> K2[计算先验分数]
        K2 --> K3[计算价值分数]
        K3 --> K4[归一化处理]
    end
    
    subgraph 温度参数控制
        M --> M1[temperature=0<br/>确定性选择]
        M --> M2[temperature=inf<br/>随机选择]
        M --> M3[0<temperature<inf<br/>软化选择]
    end
    
    subgraph 最大树深度跟踪
        I --> I1[初始化当前深度]
        M --> I2[更新最大深度]
        U --> I3[记录搜索统计]
    end
```
### 算法解释
1. **初始化阶段**
   - 创建根节点或使用提供的根节点
   - 使用表示网络获取初始状态
   - 添加 Dirichlet 探索噪声

2. **选择阶段**
   - 使用 UCB 公式选择动作
   - 计算探索奖励和利用价值
   - 通过温度参数控制选择策略

3. **扩展阶段**
   - 使用动态网络预测新状态
   - 扩展叶子节点
   - 更新树深度统计

4. **反向传播阶段**
   - 更新节点访问计数和价值
   - 更新 MinMaxStats

### 算法数据结构
- 节点结构
```
class Node:
    def __init__(self, prior: float):
        self.visit_count: int = 0          # 访问次数
        self.to_play: int = -1             # 当前玩家ID
        self.prior: float = prior          # 先验概率
        self.value_sum: float = 0          # 价值总和
        self.children: Dict[int, Node] = {} # 子节点映射 {action: Node}
        self.hidden_state: torch.Tensor = None  # 隐藏状态
        self.reward: float = 0             # 即时奖励

    def value(self) -> float:
        """计算节点的平均价值"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
```
- MinMaxState 最大最小值统计
```
class MinMaxStats:
    def __init__(self):
        self.maximum: float = -float("inf")  # 树中的最大值
        self.minimum: float = float("inf")   # 树中的最小值
    
    def normalize(self, value: float) -> float:
        """归一化值到[0,1]区间"""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
```

- 搜索路径
```
class SearchPath:
    """搜索路径数据结构"""
    search_path: List[Node]       # 从根到叶的节点列表
    actions: List[int]           # 选择的动作序列
    virtual_to_play: List[int]   # 每个节点的玩家ID
```

- UCB 分数计算
```
class UCBParams:
    """UCB计算相关参数"""
    pb_c_base: float     # UCB基础参数
    pb_c_init: float     # UCB初始化参数
    discount: float      # 折扣因子
    
    def calculate_ucb(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """计算UCB分数
        pb_c = log((parent.visit_count + pb_c_base + 1)/pb_c_base) + pb_c_init
        pb_c *= sqrt(parent.visit_count)/(child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = child.reward + discount * child.value()
        """
```

- 探索噪声参数
```
class ExplorationParams:
    """根节点探索噪声参数"""
    dirichlet_alpha: float    # Dirichlet分布的α参数
    exploration_fraction: float  # 探索噪声比例
```
- MCTS配置结构
```
class MCTSConfig:
    """MCTS算法配置"""
    num_simulations: int          # 模拟次数
    discount: float               # 奖励折扣因子
    pb_c_base: float             # UCB基础参数
    pb_c_init: float             # UCB初始化参数
    root_dirichlet_alpha: float   # 根节点Dirichlet噪声α参数
    root_exploration_fraction: float  # 根节点探索比例
    players: List[int]            # 玩家ID列表
    action_space: List[int]       # 动作空间
```
- 搜索结果结构
```
class MCTSResult:
    """MCTS搜索结果"""
    root: Node                    # 根节点
    max_tree_depth: int          # 最大树深度
    root_predicted_value: float  # 根节点预测价值
    visit_counts: Dict[int, int] # 访问计数 {action: count}
    search_policy: Dict[int, float]  # 搜索策略 {action: probability}
```
