import abc
from typing import List, Tuple, Optional
import numpy as np

class Game(abc.ABC):
    """
    游戏环境的抽象基类 (ABC)。
    所有具体游戏实现都需要继承此类并实现其抽象方法。
    参考 design_interface.md 和 design_structure.md。
    """

    @abc.abstractmethod
    def __init__(self, seed: Optional[int] = None):
        """初始化游戏环境，可选择设置随机种子。"""
        pass

    @property
    @abc.abstractmethod
    def action_space_size(self) -> int:
        """返回动作空间的大小。"""
        pass

    @property
    @abc.abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """返回观察空间的形状 (例如 C, H, W)。"""
        pass

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """
        重置游戏到初始状态。
        返回: 初始观察状态。
        """
        pass

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, int]:
        """
        在环境中执行一个动作。
        参数:
            action: 要执行的动作的索引。
        返回:
            observation: 新的观察状态。
            reward: 执行动作后获得的奖励。
            terminated: 游戏是否结束。
            to_play: 下一个轮到的玩家索引。
        """
        pass

    @abc.abstractmethod
    def legal_actions(self) -> List[int]:
        """返回当前状态下的合法动作列表。"""
        pass

    @abc.abstractmethod
    def to_play(self) -> int:
        """返回当前轮到的玩家索引 (例如 0 或 1)。"""
        pass

    @abc.abstractmethod
    def get_observation(self) -> np.ndarray:
        """获取当前的观察状态。"""
        pass

    @abc.abstractmethod
    def terminal(self) -> bool:
        """检查当前游戏状态是否为终止状态。"""
        pass

    @abc.abstractmethod
    def render(self) -> None:
        """(可选) 渲染游戏当前状态，用于可视化。"""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """(可选) 关闭环境并释放资源。"""
        pass


class TicTacToeGame(Game):
    """
    井字棋游戏环境实现。
    玩家 1 (X) 用 1 表示，玩家 2 (O) 用 -1 表示。
    玩家索引: 0 代表玩家 1 (X)， 1 代表玩家 2 (O)。
    """
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        if seed is not None:
            np.random.seed(seed)
        self._board = np.zeros((3, 3), dtype=np.int8)
        self._current_player = 0 # 0 for X, 1 for O
        self._winner = None # None: ongoing, 0: X wins, 1: O wins, -1: draw

    @property
    def action_space_size(self) -> int:
        return 9 # 3x3 grid

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        # (C, H, W) - Channel 0: Player X's pieces, Channel 1: Player O's pieces
        return (2, 3, 3)

    def reset(self) -> np.ndarray:
        self._board = np.zeros((3, 3), dtype=np.int8)
        self._current_player = 0
        self._winner = None
        return self.get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, int]:
        if not (0 <= action < 9):
            raise ValueError(f"Invalid action: {action}. Action must be between 0 and 8.")
        row, col = divmod(action, 3)

        if self._board[row, col] != 0:
            # raise ValueError(f"Invalid action: Cell ({row}, {col}) is already occupied.")
            # 在强化学习中，非法动作通常返回负奖励并保持状态不变，或由 MCTS 过滤掉
            # 这里我们假设调用者会传入合法动作，或者返回一个惩罚
            # 为简单起见，我们先假设传入的是合法动作
             print(f"Warning: Illegal move {action} attempted on occupied cell ({row}, {col}). State unchanged.")
             # 或者返回一个大的负奖励和终止状态？取决于设计
             # return self.get_observation(), -1.0, False, self.to_play() # 简单处理：状态不变，小惩罚
             # 严格处理：抛出异常
             raise ValueError(f"Illegal move: Cell ({row}, {col}) is already occupied.")


        player_mark = 1 if self._current_player == 0 else -1
        self._board[row, col] = player_mark

        terminated = self._check_termination()
        reward = 0.0
        if terminated:
            if self._winner == self._current_player: # 当前玩家获胜
                reward = 1.0
            elif self._winner == -1: # 平局
                reward = 0.0
            else: # 对手获胜 (理论上不会在这里发生，因为是当前玩家移动后检查)
                 # 但如果考虑对手获胜的情况，奖励应为 -1
                 # 为了清晰，我们在检查终止状态时就确定了赢家
                 # 如果游戏结束但不是当前玩家赢，也不是平局，说明是对手赢了
                 if self._winner is not None and self._winner != -1:
                     reward = -1.0 # 输了给对手

        # 切换玩家
        if not terminated:
            self._current_player = 1 - self._current_player

        next_player = self.to_play() # 获取下一个玩家
        # 如果游戏结束，to_play 返回的是游戏结束时的玩家，这可能需要调整
        # MuZero 通常需要游戏结束时的价值，所以这里的 to_play 可能需要反映这一点
        # 或者在 GameHistory 中单独记录

        return self.get_observation(), reward, terminated, next_player


    def _check_termination(self) -> bool:
        """检查游戏是否结束 (胜利或平局)。"""
        player_mark = 1 if self._current_player == 0 else -1

        # 检查行、列、对角线
        for i in range(3):
            if np.all(self._board[i, :] == player_mark) or \
               np.all(self._board[:, i] == player_mark):
                self._winner = self._current_player
                return True
        if np.all(np.diag(self._board) == player_mark) or \
           np.all(np.diag(np.fliplr(self._board)) == player_mark):
            self._winner = self._current_player
            return True

        # 检查平局 (棋盘已满)
        if np.all(self._board != 0):
            self._winner = -1 # Draw
            return True

        return False # Game not terminated

    def legal_actions(self) -> List[int]:
        if self.terminal():
            return [] # 游戏结束没有合法动作
        return [i for i, cell in enumerate(self._board.flatten()) if cell == 0]

    def to_play(self) -> int:
        # 如果游戏结束，返回谁是赢家或特定值可能更有用？
        # MuZero论文中 to_play 在游戏结束时似乎没有明确定义，
        # 但通常在 MCTS 节点中会存储游戏结果。
        # 这里我们简单返回当前理论上的玩家，即使游戏已结束。
        return self._current_player

    def get_observation(self) -> np.ndarray:
        """
        将棋盘状态转换为 (2, 3, 3) 的 NumPy 数组。
        Channel 0: 玩家 1 (X) 的棋子位置 (1 表示有棋子, 0 表示无)
        Channel 1: 玩家 2 (O) 的棋子位置 (1 表示有棋子, 0 表示无)
        """
        obs = np.zeros((2, 3, 3), dtype=np.float32)
        obs[0, :, :] = (self._board == 1).astype(np.float32) # Player X (1)
        obs[1, :, :] = (self._board == -1).astype(np.float32) # Player O (-1)
        return obs

    def terminal(self) -> bool:
        return self._winner is not None

    def render(self) -> None:
        """在控制台打印棋盘状态。"""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("-------------")
        for row in range(3):
            print("|", end=" ")
            for col in range(3):
                print(symbols[self._board[row, col]], end=" | ")
            print("\n-------------")
        if self.terminal():
            if self._winner == 0:
                print("Player X wins!")
            elif self._winner == 1:
                print("Player O wins!")
            elif self._winner == -1:
                print("It's a draw!")
        else:
             print(f"Player {'X' if self._current_player == 0 else 'O'}'s turn.")


    def close(self) -> None:
        pass # 对于简单环境，可能不需要特殊清理

# 可以在这里添加一些简单的测试代码
if __name__ == '__main__':
    game = TicTacToeGame()
    obs = game.reset()
    print("Initial Observation Shape:", obs.shape)
    game.render()

    done = False
    while not done:
        legal = game.legal_actions()
        print("Legal actions:", legal)
        if not legal:
            break
        # 随机选择一个合法动作
        action = np.random.choice(legal)
        print(f"Player {game.to_play()} takes action: {action}")

        obs, reward, terminated, next_player = game.step(action)
        print(f"Reward: {reward}, Terminated: {terminated}, Next Player: {next_player}")
        game.render()
        done = terminated

    print("Game Over.")