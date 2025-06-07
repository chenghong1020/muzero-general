import math
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch

from config import MuZeroConfig
from game import Game
from model import MuZeroNetwork
from mcts import MCTSFacade, Node, select_action
from structures import GameHistory
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer


def run_selfplay(config: MuZeroConfig, shared_storage: SharedStorage, replay_buffer: ReplayBufferProcess):
    """
    自我对弈主循环，负责初始化和管理整个自我对弈过程，协调多个并行的游戏对弈 Actor。
    
    Args:
        config: MuZero 配置。
        shared_storage: 共享存储实例，用于获取最新的网络权重。
        replay_buffer: 回放缓冲区进程实例，用于存储游戏历史。
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"selfplay_{time.strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    logger = logging.getLogger("selfplay")
    
    # 创建游戏历史队列，用于从子进程收集完成的游戏
    game_history_queue = mp.Queue()
    
    # 创建停止事件，用于优雅地停止自我对弈过程
    stop_event = mp.Event()
    
    # 确定要启动的 play_game Actor 数量
    num_actors = config.num_actors if hasattr(config, 'num_actors') else mp.cpu_count()
    logger.info(f"启动 {num_actors} 个自我对弈 Actor")
    
    # 启动 play_game Actor 进程
    processes = []
    for actor_id in range(num_actors):
        process = mp.Process(
            target=play_game_actor,
            args=(actor_id, config, shared_storage, game_history_queue, stop_event),
            daemon=True
        )
        process.start()
        processes.append(process)
        logger.info(f"Actor {actor_id} 已启动，进程 ID: {process.pid}")
    
    try:
        # 主循环：收集游戏历史并发送到回放缓冲区
        games_collected = 0
        while not stop_event.is_set():
            try:
                # 非阻塞方式从队列获取游戏历史，超时 0.1 秒
                game_history = game_history_queue.get(timeout=0.1)
                
                # 将游戏历史发送到回放缓冲区进程
                replay_buffer.save_game(game_history, shared_storage)
                
                games_collected += 1
                if games_collected % 10 == 0:  # 每收集 10 局游戏记录一次日志
                    logger.info(f"已收集 {games_collected} 局游戏")
                    
                # 检查是否达到训练步数上限
                if hasattr(config, 'training_steps') and \
                   shared_storage.get_training_step() >= config.training_steps:
                    logger.info(f"已达到训练步数上限 {config.training_steps}，停止自我对弈")
                    stop_event.set()
                    break
                    
            except mp.queues.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"处理游戏历史时出错: {str(e)}")
                
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在停止自我对弈...")
        stop_event.set()
    finally:
        # 等待所有进程结束
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                logger.warning(f"进程 {process.pid} 未能正常结束，强制终止")
                process.terminate()
        
        logger.info(f"自我对弈结束，共收集 {games_collected} 局游戏")


def play_game_actor(actor_id: int, config: MuZeroConfig, shared_storage: SharedStorage,  # 参数改回config
                   game_history_queue: mp.Queue, stop_event: mp.Event):
    """
    游戏对弈 Actor，在独立进程中运行，负责进行一局或多局完整的游戏。
    
    Args:
        actor_id: Actor 的唯一标识符。
        game_name: 游戏名称。
        shared_storage: 共享存储实例，用于获取最新的网络权重。
        game_history_queue: 用于发送完成的游戏历史到主进程的队列。
        stop_event: 用于接收停止信号的事件。
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format=f"Actor-{actor_id} %(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"selfplay_actor_{actor_id}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    logger = logging.getLogger(f"selfplay_actor_{actor_id}")
    
    # 设置随机种子，确保每个 Actor 生成不同的游戏
    seed = actor_id + int(time.time()) % 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建游戏环境 
    # 直接使用config中的游戏配置（新增）
    game_class = config.game_class  # 假设config已包含game_class属性
    game = game_class(seed)
    
    # 创建模型实例（使用传入的config）
    model = MuZeroNetwork(config)  # 改回使用config
    
    # 游戏计数器
    games_played = 0
    
    # 主循环：不断进行游戏，直到收到停止信号
    while not stop_event.is_set():
        try:
            # 从共享存储获取最新的网络权重
            weights = shared_storage.get_weights()
            model.set_weights(weights)
            
            # 可选：等待训练器完成一定步数的训练
            if hasattr(config, 'wait_for_training_step') and config.wait_for_training_step > 0:
                current_step = shared_storage.get_training_step()
                if current_step < config.wait_for_training_step:
                    logger.info(f"等待训练步数达到 {config.wait_for_training_step}，当前: {current_step}")
                    time.sleep(1)
                    continue
            
            # 进行一局游戏
            game_history = play_game(config, model, game, shared_storage)
            
            # 将游戏历史发送到主进程
            game_history_queue.put(game_history)
            
            games_played += 1
            logger.info(f"完成第 {games_played} 局游戏，总奖励: {sum(game_history.reward_history)}")
            
            # 可选：控制自我对弈速率
            if hasattr(config, 'self_play_delay') and config.self_play_delay > 0:
                time.sleep(config.self_play_delay)
                
        except Exception as e:
            logger.error(f"游戏过程中出错: {str(e)}")
            # 短暂暂停后继续
            time.sleep(1)


def play_game(config: MuZeroConfig, model: MuZeroNetwork, game: Game, 
              shared_storage: Optional[SharedStorage] = None) -> GameHistory:
    """
    进行一局完整的游戏，使用 MCTS 搜索选择动作。
    
    Args:
        config: MuZero 配置。
        model: MuZero 网络模型。
        game: 游戏环境实例。
        shared_storage: (可选) 共享存储实例，用于记录游戏统计信息。
        
    Returns:
        GameHistory: 包含完整游戏历史的对象。
    """
    # 重置游戏环境
    observation = game.reset()
    
    # 创建游戏历史对象
    game_history = GameHistory(config=config)
    
    # 记录初始状态
    game_history.observation_history.append(observation)
    game_history.action_history.append(0)  # 初始动作设为 0（无动作）
    game_history.reward_history.append(0.0)  # 初始奖励设为 0
    game_history.to_play_history.append(game.to_play())
    
    # 创建 MCTS Facade
    mcts_facade = MCTSFacade(config, model, game)
    
    # 游戏主循环
    terminated = False
    max_moves = config.max_moves if hasattr(config, 'max_moves') else 1000
    
    with torch.no_grad():  # 禁用梯度计算以提高性能
        while not terminated and len(game_history.action_history) <= max_moves:
            # 执行 MCTS 搜索
            root, extra_info = mcts_facade.run()
            
            # 根据温度参数选择动作
            # 获取当前游戏步数
            num_moves = len(game_history.action_history) - 1  # 减去初始的无动作
            
            # 选择动作
            action = select_action(config, num_moves, root, training=True)
            
            # 执行动作
            observation, reward, terminated, next_player = game.step(action)
            
            # 存储搜索统计信息
            # 将子节点访问计数转换为字典格式 {action: visit_count}
            child_visits = {action: child.visit_count for action, child in root.children.items()}
            game_history.store_search_statistics(root.value(), child_visits)
            
            # 存储游戏步骤信息
            game_history.observation_history.append(observation)
            game_history.action_history.append(action)
            game_history.reward_history.append(reward)
            game_history.to_play_history.append(next_player)
            
            # 记录 MCTS 统计信息（用于调试）
            if shared_storage is not None and hasattr(shared_storage, 'set_info'):
                # 每 10 步或游戏结束时记录统计信息
                if num_moves % 10 == 0 or terminated:
                    shared_storage.atomic_update_info(lambda info: {
                        **info,
                        "num_played_steps": info.get("num_played_steps", 0) + 1,
                        "num_played_games": info.get("num_played_games", 0) + (1 if terminated else 0),
                        "max_tree_depth": extra_info["max_tree_depth"],
                        "root_value": root.value(),
                    })
    
    # 设置游戏结束标志
    game_history.terminated = terminated
    
    return game_history