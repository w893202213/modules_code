# KV Cache 是对 MHA 的优化，思路是空间换时间
#   模型预处理整个输入 prompt , 并计算出所有 token 的 K 和 V 矩阵, 将它们缓存起来
#   对于每个新生成的 token, 只需计算它自己的 K 和 V 向量, 然后将其与缓存中的 K 和 V 矩阵拼接起来, 形成当前步的 K/V 矩阵。
# 这样, 模型的计算量只与新生成的 token 数量(通常是 1 ) 相关, 而与序列总长度无关, 大大提高了推理效率.

# 代码层面, 只与最初版本的 MHA 有 3 处修改.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionWithCacha(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttentionWithCacha, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divided by num_heads."

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, past_key_value=None):  # 修改点1: 传入参数多一个历史的key_value
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 修改点2: 拼接历史 KV Cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 此前 past_key 和 K 的维度均为 (batch_size, num_heads, seq_len, d_k)
            # 需要沿着 seq_len, 即维度 2 的方向做拼接, 只需使用 torch.cat 即可完成拼接
            # past_value 和 V 的拼接同理
            K = torch.cat([past_key, K], dim=2)
            V = torch.cat([past_value, V], dim=2)
        # 将当前的 K, V 作为新的 cache 返回
        present_key_value = (K, V)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_scores = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_scores, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)

        return output, present_key_value  # 修改点3: 返回时将当前的 KV Cache 一同返回


# 测试代码
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    mha = MultiHeadAttentionWithCacha(num_heads, d_model)

    # 预填充
    prefill_len = 5
    x_prefill = torch.randn(1, prefill_len, d_model)
    print("预填充：")
    prefill_mask = torch.tril(torch.ones(prefill_len, prefill_len))
    output_prefill, kv_cache = mha(x_prefill, prefill_mask)
    print(f"预填充输入形状：{x_prefill.shape}")
    print(f"预填充输出形状：{output_prefill.shape}")
    print(f"预填充后 K Cache 形状：{kv_cache[0].shape}")
    print(f"预填充后 K Cache 序列长度：{kv_cache[0].size(2)}")
    print("-"*20)

    # 解码
    decoder_len = 1
    x_decoder = torch.randn(1, decoder_len, d_model)
    print("解码：")

    # 掩码只针对新 token 的注意力计算
    # scores 的形状是 (batch_size, seq_len, 1, total_seq_len)
    # 我们需要的形状是 (1, 1, 1, total_seq_len)
    past_len = kv_cache[0].size(2)
    new_len = past_len + decoder_len

    # 创建一个下三角矩阵作为掩码矩阵，然后只取最后一行
    decoder_mask = torch.tril(torch.ones(new_len, new_len))[-decoder_len:, :]
    output_decode, new_kv_cache = mha(x_decoder, mask=decoder_mask, past_key_value=kv_cache)

    print(f"解码输入形状：{x_decoder.shape}")
    print(f"解码输出形状：{output_decode.shape}")
    print(f"解码后 K Cache 形状：{new_kv_cache[0].shape}")
    print(f"解码后 K Cache 序列长度：{new_kv_cache[0].size(2)}")
    print("-"*20)

