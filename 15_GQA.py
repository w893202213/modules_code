import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# GQA 是介于 MHA 和 MQA 之间的一种折中方案，旨在平衡 MQA 的推理加速和 MHA 的模型性能。
# 核心做法是将 num_heads 个头分成 kv_groups 组，每组内的查询头共享一个 K 和 V 头。可以理解为“头数为 kv_groups”
# 代码层面，在 MHA 的代码基础上有 5 处修改点
# 要求：d_model 必须是 num_heads 的倍数，且 num_heads 必须是 kv_groups 的倍数
class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, d_model, kv_groups):  # 修改点1: 传入参数多了一个 kv_groups
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.kv_groups = kv_groups  # 修改点2: self 参数多了一个 self.kv_groups

        self.W_Q = nn.Linear(d_model, d_model)
        # 修改点3: GQA 的关键, W_K 和 W_V 的投影维度为 kv_groups * d_k
        self.W_K = nn.Linear(d_model, self.kv_groups * self.d_k)
        self.W_V = nn.Linear(d_model, self.kv_groups * self.d_k)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 修改点4: K 和 V 被投影成 kv_groups 个头。
        K = self.W_K(x).view(batch_size, seq_len, self.kv_groups, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.kv_groups, self.d_k).transpose(1, 2)

        # 修改点5：复制 K/V 头以匹配 Q 头
        if self.kv_groups != self.num_heads:  # 如果 kv_groups == num_heads 的话，退化为 MHA
            repeat_factor = self.num_heads // self.kv_groups
            # repeat_interleave 是 PyTorch 中的一个方法，用于将张量的元素沿指定维度重复多次。
            K = K.repeat_interleave(repeat_factor, dim=1)
            V = V.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-INF'))

        attention_scores = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_scores, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)

        return output


if __name__ == '__main__':
    d_model = 512
    num_heads = 8

    print("--- GQA with kv_groups = 4 ---")
    kv_groups = 4
    x = torch.randn(2, 20, d_model)
    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, kv_groups=kv_groups)
    out = gqa(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print("----------------------------\n")

    print("--- GQA with kv_groups = 1 (等价于MQA) ---")
    kv_groups = 1
    x = torch.randn(2, 20, d_model)
    gqa_mqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, kv_groups=kv_groups)
    out_mqa = gqa_mqa(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out_mqa.shape}")
    print("----------------------------\n")

    print("--- GQA with kv_groups = 8 (等价于MHA) ---")
    kv_groups = 8
    x = torch.randn(2, 20, d_model)
    gqa_mha = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, kv_groups=kv_groups)
    out_mha = gqa_mha(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out_mha.shape}")
    print("----------------------------\n")

