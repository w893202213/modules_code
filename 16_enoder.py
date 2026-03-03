import torch.nn as nn
from MHA import MultiHeadAttention


class EncoderBlcok(nn.Module):
    def __init__(self, num_heads, d_model, ffn_hidden_dim, dropout=0.1):
        super(EncoderBlcok, self).__init__()

        # 子层1: 多头自注意力
        self.attention = MultiHeadAttention(num_heads, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        # 子层2: 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 自注意力 + 残差&归一化
        attn_out = self.attention(x, mask)
        x = x + self.dropout1(attn_out)
        x = self.ln1(x)

        # 前馈网络 + 残差&归一化
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.ln2(x)

        return x

