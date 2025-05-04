import torch
import numpy as np
from typing import List

def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    将网络输出的离散支持分布转换为标量值。
    参考 MuZero 论文 Appendix F Network Architecture。
    Args:
        logits: 网络输出的原始 logits，形状为 (batch_size, num_atoms)。
        support_size: 离散支持的范围 [-support_size, support_size]。
    Returns:
        标量值，形状为 (batch_size, 1)。
    """
    if support_size == 0:
        # 如果 support_size 为 0，则假定 logits 直接是标量值
        return logits  # 直接返回，保持 (batch_size, 1) 形状

    # 生成支持值 [-support_size, ..., support_size]
    support = torch.linspace(-support_size, support_size, 2 * support_size + 1).to(logits.device)
    support = support.view(1, -1) # 形状变为 (1, num_atoms)

    # 应用 softmax 获取概率分布
    probabilities = torch.softmax(logits, dim=1) # 在 num_atoms 维度上应用 softmax

    # 计算期望值
    scalar = torch.sum(support * probabilities, dim=1, keepdim=True)
    return scalar

def scalar_to_support(scalar: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    将标量值转换为离散支持分布的目标。
    参考 MuZero 论文 Appendix F Network Architecture。
    Args:
        scalar: 目标标量值，形状为 (batch_size, 1)。
        support_size: 离散支持的范围 [-support_size, support_size]。
    Returns:
        目标分布，形状为 (batch_size, num_atoms)。
    """
    if support_size == 0:
        # 如果 support_size 为 0，直接返回标量值
        return scalar

    # 将标量值限制在支持范围内
    scalar = torch.clamp(scalar, -support_size, support_size)

    # 计算离散支持值
    num_atoms = 2 * support_size + 1

    # 计算每个标量值在支持上的位置比例
    scalar = scalar.view(-1, 1) # (batch_size, 1)

    # 创建目标分布
    target_support = torch.zeros(scalar.shape[0], num_atoms).to(scalar.device)

    # 处理浮点数标量
    is_not_integer = (scalar != torch.floor(scalar))

    # --- 对非整数进行插值 ---
    if torch.any(is_not_integer):
        # 找出非整数标量及其对应的行索引
        rows_f = torch.where(is_not_integer.view(-1))[0]
        scalar_f = scalar[rows_f] # 获取非整数标量值

        lower_bound_f = torch.floor(scalar_f).long()
        upper_bound_f = torch.ceil(scalar_f).long() # 对于非整数，ceil > floor

        upper_weight_f = scalar_f - lower_bound_f.float()
        lower_weight_f = 1.0 - upper_weight_f

        lower_idx_f = (lower_bound_f + support_size).clamp(0, num_atoms - 1)
        upper_idx_f = (upper_bound_f + support_size).clamp(0, num_atoms - 1)

        # 直接使用索引将权重添加到 target_support 的对应位置
        target_support[rows_f, lower_idx_f.view(-1)] = lower_weight_f.view(-1)
        target_support[rows_f, upper_idx_f.view(-1)] = upper_weight_f.view(-1)

    # --- 对整数直接赋值 ---
    is_integer = ~is_not_integer
    if torch.any(is_integer):
        # 找出整数标量及其对应的行索引
        rows_i = torch.where(is_integer.view(-1))[0]
        scalar_i = scalar[is_integer] # 获取整数标量值
        
        # 计算整数标量对应的支撑点索引
        exact_idx_i = (scalar_i + support_size).long().clamp(0, num_atoms - 1)
        
        # 直接使用索引将 1.0 赋值到对应位置
        target_support[rows_i, exact_idx_i.view(-1)] = 1.0

    return target_support

# 可以在这里添加其他工具函数，例如 MLP 构建器等
def mlp(input_size: int, layer_sizes: List[int], output_size: int, activation=torch.nn.ReLU, output_activation=torch.nn.Identity):
    """构建一个简单的多层感知机 (MLP)"""
    layers = []
    in_size = input_size
    for size in layer_sizes:
        layers.append(torch.nn.Linear(in_size, size))
        layers.append(activation())
        in_size = size
    layers.append(torch.nn.Linear(in_size, output_size))
    layers.append(output_activation())
    return torch.nn.Sequential(*layers)