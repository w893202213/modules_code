import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, elementwise_affine: bool = True):
        """
        RMSNorm 实现。(简化版本的LayerNorm)

        :param d_model: 模型的特征维度
        :param eps: 防止除以 0 的小常数
        :param elementwise_affine: 是否使用可学习的缩放参数
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根，-1 表示最后一个维度，keepdim用于压缩维度为 1，方便后续广播
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # 标准化
        x_norm = x / rms

        # 缩放
        if self.elementwise_affine:
            output = x_norm * self.weight
        else:
            output = x_norm

        return output


# 测试代码
if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 4, 8
    x = torch.randn(batch_size, seq_len, d_model)
    rmsnorm = RMSNorm(d_model)
    output = rmsnorm(x)

    print("输入数据维度：", x.shape)
    print("输出数据维度：", output.shape)
    print(output)
