import numpy as np


def compute_f1(y_true, y_pred):
    """
    计算精确率，召回率和F1_score
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算TP，FP，FN
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # 计算精确率，处理分母为0的情况
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    # 计算召回率，处理分母为0的情况
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    # 计算F1_score，处理分母为0的情况
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1


# 测试代码
if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]

    precision, recall, f1 = compute_f1(y_true, y_pred)

    print(f"precision:{precision:.4f}")
    print(f"recall:{recall:.4f}")
    print(f"F1 score:{f1:.4f}")
