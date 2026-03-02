import numpy as np


def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = X.dot(theta)
        err = predictions - y

        gradient = (1 / m) * X.T.dot(err)
        theta = theta - alpha * gradient
        history[i] = compute_cost(X, y, theta)

    return theta, history


if __name__ == "__main__":
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    X_b = np.c_[np.ones((100, 1)), X]
    theta = np.random.randn(2, 1)

    alpha = 0.01
    num_iters = 1000

    final_theta, history = gradient_descent(X_b, y, theta, alpha, num_iters)

    print("优化后的参数：")
    print(final_theta)
