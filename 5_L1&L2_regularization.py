import numpy as np


# L2正则化实现
def linear_regression_with_L2(X, y, lr=0.1, lambd=0.1, epochs=100):
    """
    带L2正则化的线性回归

    :param X:特征矩阵
    :param y:标签向量
    :param lr:学习率
    :param lambd:正则化强度
    :param epochs:迭代次数
    :return:tuple: 训练好的权重 w 和 偏置 b
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        # 计算梯度，L2正则化的梯度是 lambda * w
        dw = (X.T @ error) / m + lambd * w
        db = np.mean(error)

        # 更新参数
        w -= lr * dw
        b -= lr * db

    return w, b


# L1 正则化的实现
def linear_regression_with_L1(X, y, lr=0.1, lambd=0.1, epochs=100):
    """
    带L1正则化的线性回归

    :param X:特征矩阵
    :param y:标签向量
    :param lr:学习率
    :param lambd:正则化强度
    :param epochs:迭代次数
    :return:tuple: 训练好的权重 w 和 偏置 b
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        # 计算梯度，L1正则化的梯度是lambd * sign(w)
        dw = (X.T @ error) / m + lambd * np.sign(w)
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db

    return w, b


# 主程序
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(100, 3)
    true_w = np.array([2.0, -3.0, 1.0])
    y = X @ true_w + np.random.randn(100) * 0.5

    print("L2正则化")
    w_l2, b_l2 = linear_regression_with_L2(X, y)
    print(f"学到的权重：{w_l2}")

    print("L1正则化")
    w_l1, b_l1 = linear_regression_with_L1(X, y)
    print(f"学到的权重：{w_l1}")
