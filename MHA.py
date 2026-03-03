import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (seq_len, seq_len), 需广播为: (batch, num_heads, seq_len, seq_len)
            if mask.dim == 2:
                mask = mask.unsueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))  # mask:(1, 1, seq_len, seq_len)

        attn_scores = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_scores, V)
        # contiguous 用于确保交换后的张量在内存中是连续存储的，为下一步的 view 操作做准备
        # 即 transpose 操作和 view 操作之间一定要加一个 contiguous
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)
        return output


if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    num_heads = 8
    d_model = 512

    # 随机生成一个输入张量
    x = torch.randn(batch_size, seq_len, d_model)

    # 初始化模型
    mha = MultiHeadAttention(num_heads, d_model)

    # 不带掩码的测试
    no_mask = mha(x)
    print("不带掩码的输出形状：", no_mask.shape)  # (4, 10, 512)

    # 带掩码的输出形状
    casual_mask = torch.ones(seq_len, seq_len)
    casual_mask = torch.tril(casual_mask)  # 生成下三角掩码矩阵
    with_mask = mha(x, mask=casual_mask)
    print("带掩码的输出形状：", with_mask.shape)  # (4, 10, 512)
