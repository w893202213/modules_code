import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):  # k (int): 邻居的数量
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X_train, y_train):  # 训练KNN模型(实际上只是存储训练数据)
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        对新样本进行分类预测。

        :param X_test:(np.ndarray)待预测的数据
        :return:(np.ndarray)预测的类别标签数组
        """
        predictions = []
        for x in X_test:
            # 计算当前测试点到所有训练点的欧氏距离(特征方向,即axis=1)
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # 找到距离最近的前 k 个点的索引
            nearest_indices = np.argsort(distances)[:self.k]

            # 获取这 k 个邻居的标签
            nearest_labels = self.y_train[nearest_indices]

            # 投票决定最终类别
            # most_common(1)返回出现次数最多的前1个元素。格式[(label,count)]。取[0][0]拿到label
            most_common = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)


# 测试代码
if __name__ == "__main__":
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 4],  # 这一簇属于类别 0
        [5, 6],
        [7, 8],
        [8, 9]  # 这一簇属于类别 1
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [2, 2],
        [6, 6]
    ])

    # 运行模型
    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("测试数据：")
    print(X_test)
    print("\n预测结果：")
    print(predictions)
