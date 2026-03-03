# 前馈神经网络, 公式为: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    input_dim = 512
    hidden_dim = 2048
    output_dim = 512

    ffn = FeedForwardNetwork(input_dim, hidden_dim, output_dim)
    x = torch.randn(2, 10, input_dim)  # batch_size=2, seq_len=10, input_dim=512

    output = ffn(x)
    print(output.shape)
