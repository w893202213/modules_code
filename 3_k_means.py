import numpy as np


def kmeans(data, k, thresh=1, max_iterations=100):
    # 随机初始化 k 个中心点
    # data.shape[0] 表示数据点数量
    centers = data[np.random.choice(data.shape[0], k, replace=False)]

    labels = None

    for _ in range(max_iterations):
        # 计算每个样本到各中心点的距离
        # data[:, None] 将形状为 (N, M) 的数据数组扩展为 (N, 1, M)，(样本数, 特征数)
        # data[:, None] - centers 利用 NumPy 的广播机制，计算 N 个数据点与 k 个中心点之间的向量差，得到形状为（N,k,M）的数组。
        # np.linalg.norm(..., axis=2) 沿着最后一个轴（特征轴）计算 L2 范数（欧几里得距离）。
        # 结果是形状为 (N, k) 的 distances 数组，distances[i, j] 表示第 i 个数据点到第 j 个族中心的距离。
        distance = np.linalg.norm(data[:, None] - centers, axis=2)

        # 根据距离最近的中心点将样本分类到最近的族
        labels = np.argmin(distance, axis=1)  # 所得矩阵的大小为 (N, )

        # 更新中心点为每个族的平均值
        # data[labels == i] 提取所有属于簇 i 的数据子集
        # .mean(axis=0) 计算这个子集沿着特征轴（第 0 轴）的均值，得到簇 i 的新中心点坐标
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 判断是否收敛
        # np.all()返回bool数组中是否所有元素均为true
        if np.all(centers == new_centers) or np.linalg.norm(new_centers - centers) < thresh:
            break

        centers = new_centers

    return labels, centers


# 测试代码
data = np.random.rand(100, 2)

k = 3

labels, centers = kmeans(data, k)

print("族标签：", labels)
print("聚类中心点：", centers)
