import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 多头注意力：transformer的核心
# 要求：d_model 必须是 num_heads 的倍数
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # 每个头的维度

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 线性投影  x: (batch_size, seq_len, d_model)
        # 拆分成多头  x: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # 维度调整  x: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力分数
        # Q: (batch_size, num_heads, seq_len, d_k)
        # V.transpose: (batch_size, num_heads, d_k, seq_len)
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask: (seq_len, seq_len) 需广播为: (batch_size, num_heads, seq_len, seq_len)
            if mask.dim == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 在最后一个维度上进行归一化，对于单个query，在其所有key上计算概率分布
        attn_scores = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_scores, V)

        # 拼接多头并进行最终线性变换
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)
        return output


if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    num_heads = 8
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(num_heads, d_model)

    no_mask = mha(x)
    print("不带掩码的输出形状：", no_mask.shape)

    casual_mask = torch.ones(seq_len, seq_len)
    casual_mask = torch.tril(casual_mask)  # 生成下三角掩码矩阵
    with_mask = mha(x, casual_mask)
    print("带掩码的输出形状：", with_mask.shape)
