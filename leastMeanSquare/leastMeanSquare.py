import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    """
    sklearn 风格的线性回归（支持 GD 和 SGD）
    -----------------------------------------------------
    参数：
        step: float，学习率
        epochs: int，迭代次数
        method: 'gd' 或 'sgd'
    -----------------------------------------------------
    方法：
        fit(X, y)
        predict(X)
        get_params()
        set_params()
    """
    
    def __init__(self, step=0.001, epochs=20000, method="gd"):
        self.step = step
        self.epochs = epochs
        self.method = method.lower()
        self.w = None
        self.loss_history = []

    def _compute_loss(self, X, y):
        pred = np.matmul(X, self.w)
        return 0.5 * np.mean((pred - y) ** 2)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        N, d = X.shape

        # 初始化 w
        self.w = np.zeros(d) + 1.0

        self.loss_history = []

        for epoch in range(self.epochs):

            if self.method == "gd":
                # 批梯度下降
                y_pred = np.matmul(X, self.w)
                grad = (1/N) * np.matmul(X.T, (y_pred - y))
                self.w = self.w - self.step * grad

            elif self.method == "sgd":
                # 随机梯度下降
                i = np.random.randint(0, N)
                x_i = X[i]
                y_i = y[i]

                y_pred = np.dot(x_i, self.w)
                grad = (y_pred - y_i) * x_i
                self.w = self.w - self.step * grad

            else:
                raise ValueError("method 必须是 'gd' 或 'sgd'")

            # 记录 loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.array(X) @ self.w

    def get_params(self):
        return {"step": self.step, "epochs": self.epochs, "method": self.method}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ============================
# 测试程序 + 可视化
# ============================

if __name__ == "__main__":
    # 数据
    X = np.array([[1, 2],
                  [2, 5],
                  [5, 1],
                  [4, 2]], dtype=float)
    y = np.array([19, 26, 19, 20], dtype=float)

    # GD
    model_gd = LinearRegressionGD(step=0.001, epochs=20000, method="gd")
    model_gd.fit(X, y)

    # SGD
    model_sgd = LinearRegressionGD(step=0.001, epochs=20000, method="sgd")
    model_sgd.fit(X, y)

    # 打印结果
    print("GD w =", model_gd.w)
    print("SGD w =", model_sgd.w)

    # 可视化 loss 曲线
    plt.figure(figsize=(8,5))
    plt.plot(model_gd.loss_history, label="GD Loss")
    plt.plot(model_sgd.loss_history, label="SGD Loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss vs Epoch (GD vs SGD)")
    plt.legend()
    plt.grid(True)
    plt.show()
