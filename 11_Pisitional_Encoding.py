import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        绝对位置编码：正余弦位置编码

        :param d_model: int 模型维度
        :param max_len: int 预计算的最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 初始化一个 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 创建位置张量：(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率项：div_term = 1 / (10000^(2i/d_model)), 维度: (d_model/2, )
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 广播并计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 使用register_buffer 将 pe 注册为非参数状态，避免随反向传播更新
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        为输入张量添加位置编码
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


# 测试代码
if __name__ == '__main__':
    max_len = 200
    batch_size = 4
    seq_len = 100
    d_model = 512

    # 词嵌入的输入
    input_tensor = torch.randn(batch_size, seq_len, d_model)

    # 实例化位置编码模块
    pe_module = PositionalEncoding(d_model, max_len)

    # 前向传播
    output_tensor = pe_module(input_tensor)

    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")

    # 输入张量形状: torch.Size([4, 100, 512])
    # 输出张量形状: torch.Size([4, 100, 512])
