import torch.nn as nn
from MHA import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, d_model, ffn_hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()

        # 子层1: 带掩码的多头自注意力
        self.self_attention = MultiHeadAttention(num_heads, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        # 子层2: 交叉注意力
        self.cross_attention = MultiHeadAttention(num_heads, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

        # 子层3: 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out, self_attn_mask=None, cross_attn_mask=None):
        # 带掩码的自注意力
        self_attn_out = self.self_attention(x, mask=self_attn_mask)
        x = x + self.dropout1(self_attn_out)
        x = self.ln1(x)

        # 交叉注意力
        # 这里其实需要修改 MultiHeadAttention 的 forward 函数以接受独立的 Q、K、V 输入
        # 假设我们已经将 MHA 类修改为 forward(self, q, k, v, mask=None)
        # 则 cross_attn_out = self.cross_attention(q=x, k=encoder_out, v=encoder_out, mask=cross_attn_mask)
        # 此处使用简化的调用，实际使用时需要修改
        cross_attn_out = self.cross_attention(x, mask=cross_attn_mask)
        x = x + self.dropout2(cross_attn_out)
        x = self.ln2(x)

        # 前馈网络
        ffn_out = self.ffn(x)
        x = x + self.dropout3(ffn_out)
        x = self.ln3(x)

        return x
