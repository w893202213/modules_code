import numpy as np


def gradient_descent_for_linear_regression(X, y, lr=0.01, epochs=100):
    w = 0.0
    b = 0.0
    m = len(X)  # 样本数
    loss_history = []

    for epoch in range(epochs):
        y_pred = w * X + b
        error = y_pred - y

        dw = (1 / m) * np.dot(error, X)
        db = (1 / m) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        # 记录损失
        loss = (1 / (2 * m)) * np.sum(error ** 2)
        loss_history.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")

    return w, b, loss_history


if __name__ == "__main__":
    # y = 2 * x + 1
    np.random.seed(0)
    X = np.linspace(0, 10, 100)
    y = 2 * X + 1 + np.random.randn(*X.shape) * 1.5

    w_final, b_final, _ = gradient_descent_for_linear_regression(X, y)
    print(f"最终学到的参数 -> w: {w_final:.4f} b: {b_final:.4f}")

