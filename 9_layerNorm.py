import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        LayerNorm 实现

        :param normalized_shape: (tuple或int) 需要进行归一化的维度
        :param eps: (float) 防止除以0的小常熟
        :param elementwise_affine: (bool) 是否使用可学习的缩放和平移参数
        """
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))  # 缩放参数
            self.beta = nn.Parameter(torch.zeros(self.normalized_shape))  # 平移参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确认输入张量的最后维度与 normalized_shape 匹配
        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape

        # 获取需要归一化的维度索引，例如 shape=(C,H,W) -> dims=(-3,-2,-1)
        dims = tuple(range(-len(self.normalized_shape), 0))

        # 计算均值和方差
        # unbiased=False 表示计算总体方差（分母是N），而不是样本方差（分母是N-1）
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 可学习的缩放和平移
        if self.elementwise_affine:
            return self.gamma * x_norm + self.beta

        return x_norm


# 测试代码
if __name__ == "__main__":
    # 示例1 NLP风格的输入
    print("NLP风格：")

    # (batch_size, sequence_length, embedding_dim)
    batch_size, seq_len, d_model = 2, 4, 8
    x_nlp = torch.randn(batch_size, seq_len, d_model)

    # 只对最后一个维度做归一化 (d_model, )
    ln_nlp = LayerNorm(normalized_shape=(d_model, ))
    output_nlp = ln_nlp(x_nlp)

    print("输入维度：", x_nlp.shape)
    print("归一化维度：", ln_nlp.normalized_shape)
    print("输出维度：", output_nlp.shape)
    print("输出示例：\n", output_nlp)

    print("-" * 20)

    # 示例2 视觉风格的输入
    print("视觉风格：")

    # (batch_size, channels, height, width)
    batch_size, channels, height, width = 2, 3, 4, 4
    x_vision = torch.randn(batch_size, channels, height, width)

    # 对最后三个维度做归一化 (C, H, W)
    ln_vision = LayerNorm(normalized_shape=(channels, height, width))
    output_vision = ln_vision(x_vision)

    print("输入维度：", x_vision.shape)
    print("归一化维度：", ln_vision.normalized_shape)
    print("输出维度：", output_vision.shape)
    print("输出示例：\n", output_vision)


