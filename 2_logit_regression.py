import numpy as np


def sigmoid(x):
    """
    sigmoid函数
    """
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    """
    使用numpy实现的逻辑回归分类器
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        初始化模型
        :param learning_rate:学习率
        :param num_iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练逻辑回归模型
        :param X:训练数据特征，维度(num_samples, num_features)
        :param y:训练数据标签，维度(num_samples, 1)
        """
        num_samples, num_features = X.shape
        # 初始化权重和偏置
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降
        for _ in range(self.num_iterations):
            # 线性模型：z = X*w + b
            linear_model = np.dot(X, self.weights) + self.bias
            # 使用sigmoid函数得到预测概率
            y_pred = sigmoid(linear_model)

            # 计算梯度
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict_prob(self, X):
        """
        预测样本属于正例的概率
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        根据概率和阈值进行预测
        """
        y_pred_prob = self.predict_prob(X)
        y_pred_cls = [1 if i > threshold else 0 for i in y_pred_prob]
        return np.array(y_pred_cls)
