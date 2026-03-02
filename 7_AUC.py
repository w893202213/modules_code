import numpy as np
from typing import List, Tuple


def calculate_AUC(labels: List[int], scores: List[float]) -> float:
    """
    计算AUC
    M 为正样本数，N为负样本数
    AUC值定义为正样本预测概率大于负样本预测概率的个数。
    相等的情况记为半个。

    :param labels:真实标签列表(0或1)
    :param scores:模型预测得分/概率列表
    :return:AUC值
    """
    # 分离正样本和负样本的预测分数
    positive_scores: List[float] = []
    negative_scores: List[float] = []

    for i in range(len(labels)):
        if labels[i] == 1:
            positive_scores.append(scores[i])
        else:
            negative_scores.append(scores[i])

    # M: 正样本数量  N： 负样本数量
    M = len(positive_scores)
    N = len(negative_scores)

    # 边界条件：没有正样本或负样本，AUC无法计算，返回0.5(随即猜测)
    if M == 0 or N == 0:
        return 0.5

    # 遍历所有正负样本对并计数
    count = 0  # 对应P_pos > P_neg 的情况
    ties = 0  # 对应P_pos = P_neg 的情况

    for p_score in positive_scores:
        for n_score in negative_scores:
            if p_score > n_score:
                count += 1
            elif p_score == n_score:
                ties += 1

    # 计算AUC
    auc = (count + 0.5 * ties) / (M + N)

    return auc


# 测试代码
if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_scores = [0.1, 0.3, 0.4, 0.8, 0.2, 0.9, 0.5, 0.7]

    auc_result = calculate_AUC(y_true, y_scores)
    print(f"AUC: {auc_result:.4f}")
