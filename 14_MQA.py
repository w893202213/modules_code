import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# MQA 是对 MHA 的改进，旨在减少推理时 KV Cache 的显存占用
# 方法是所有的 Query 共享同一 Key 和 Value, 可以理解为“头数为1”
# 代码层面，只有两个修改点
# 要求：d_model 必须是 num_heads 的倍数
class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        # 修改点1：W_K 和 W_V 的输出维度变成 d_k
        self.W_K = nn.Linear(d_model, self.d_k)
        self.W_V = nn.Linear(d_model, self.d_k)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 修改点2: K 和 V 只被投影成一个头，其形状为: (batch_size, seq_len, 1, d_k)
        K = self.W_K(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_scores = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_scores, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)

        return output


if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    num_heads = 8
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiQueryAttention(num_heads, d_model)

    no_mask = mha(x)
    print("不带掩码的输出形状：", no_mask.shape)

    casual_mask = torch.ones(seq_len, seq_len)
    casual_mask = torch.tril(casual_mask)  # 生成下三角掩码矩阵
    with_mask = mha(x, casual_mask)
    print("带掩码的输出形状：", with_mask.shape)
