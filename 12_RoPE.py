# 通常将RoPE实现为一个函数，直接作用于自注意力机制中的Q和K向量
from typing import Tuple

import torch


def freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    计算 RoPE 所需的频率张量 (以复数形式 torch.polar 表示)。

    :param dim: int 头的维度
    :param end: int 序列的最大长度
    :param theta: RoPE 的基数
    :return: torch.Tensor 形状为 (end, dim//2 ) 的复数张量。complex64
    """

    # 计算频率基数，shape：(dim//2, )
    # 每个维度基频为：1 / (theta^(2i/dim))
    freq_base = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 创建时间步张量 t_p, shape：(end, ), 范围是 (0, 1, 2, ..., seq_len - 1)
    t = torch.arange(end, dtype=torch.float32)

    # 计算每个位置和每个维度的旋转角度(外积)，shape: (end, dim//2)
    freqs = torch.outer(t, freq_base)  # 对应 t_p * freq_base_i

    # 将频率(角度)转化为复数形式(polar): cos(freqs) + i*sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在输入的 Q 或 K 张量上应用 RoPE

    :param xq: 输入的 Q (B, S, H, D)
    :param xk: 输入的 K (B, H, S, D)
    :param freqs_cis: 频率复数张量 (S, D//2)
    :return: 旋转后的张量，维度与输入相同
    """

    # 将 x 的最后一个维度 D 拆成 D/2 个复数对 [x0, x1], [x2, x3], ...
    # x: (B, S, H, D) -> (B, S, H, D/2, 2)
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_complex = torch.view_as_complex(xq_reshaped)
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_complex = torch.view_as_complex(xk_reshaped)

    # 调整 freqs_cis 的形状以进行广播
    seq_len = xq.shape[1]
    freqs_cis = freqs_cis[:seq_len].view(1, seq_len, 1, -1)

    # 执行复数乘法，实现旋转
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis

    # 转换回实数形式
    xq_out_reshaped = torch.view_as_real(xq_rotated)
    # 在最后一个维度上进行拼接
    xq_out = xq_out_reshaped.flatten(3)
    xk_out = torch.view_as_real(xk_rotated).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# 极简代码
def apply_rotary_emb1(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # 将实数张量转换为复数形式
    seq_len = x.shape[1]
    rotary_factor = freqs_cis[:seq_len].view(1, seq_len, 1, -1)  # 调整形状以匹配 x
    x_rotated = x_complex * rotary_factor  # 逐元素相乘，应用旋转
    x_out = torch.view_as_real(x_rotated).flatten(3)  # 将复数转换回实数，并调整形状
    return x_out.type_as(x)


# 测试代码
if __name__ == '__main__':
    d_model = 64
    max_len = 128
    n_heads = 4
    head_dim = d_model // n_heads

    # 1. 预计算频率
    freqs_cis = freqs_cis(dim=head_dim, end=max_len)  # 预计算旋转角度的正弦和余弦值

    # 2. 模拟 Q/K 向量: (B, S, H, D)
    xq = torch.randn(2, 50, n_heads, head_dim)  # 随机生成查询向量 (batch_size=2, seq_len=50, n_heads=4, head_dim=16)

    # 3. 应用 RoPE
    xq_rotated = apply_rotary_emb1(xq, freqs_cis)  # 对查询向量应用旋转位置编码

    print(f"旋转 Q' 形状: {xq_rotated.shape}")  # 输出形状应为 (2, 50, 4, 16)
