import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """LoRA旁路模块"""
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank

        # 初始化：A 使用 KaiMing 均匀分布, B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))  # a 是激活函数的负斜率, ReLU 为 0, LeakyReLU 需指定
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        x = self.lora_A(x)
        x = self.lora_B(x)
        return x * self.scaling


class LinearWithLoRA(nn.Module):
    """将 LoRA 应用于 nn.Linear 的包装层"""
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            alpha
        )

        # 冻结原始线性层参数
        self.linear.weights.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):  # 原有路径 + LoRA 旁路
        return self.linear(x) + self.lora(x)


# 辅助函数: 替换模型中所有的线性层
def apply_lora_to_model(model, rank, alpha):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 替换当前线性层
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 不是线性层就说明还有子模块, 需要递归遍历
            apply_lora_to_model(module, rank, alpha)
