"""
实现一个简单的多类逻辑回归模型来训练 MNIST 数据集
使用 numpy 进行数值计算
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def load_data(mat_file_path):
    """
    读取 mnist_all.mat 文件并整合成训练集和测试集
    mnist_all.mat 结构通常包含 train0, train1... test0, test1...
    """
    data = sio.loadmat(mat_file_path)
    
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    
    # 遍历 0 到 9 类
    for i in range(10):
        # 读取训练数据
        curr_train_data = data[f'train{i}']
        train_X.append(curr_train_data)
        train_y.append(np.full(curr_train_data.shape[0], i)) # 创建对应的标签
        
        # 读取测试数据
        curr_test_data = data[f'test{i}']
        test_X.append(curr_test_data)
        test_y.append(np.full(curr_test_data.shape[0], i))
        
    # 将列表堆叠成大的 numpy 数组
    train_X = np.vstack(train_X).astype(np.float32)
    train_y = np.concatenate(train_y)
    test_X = np.vstack(test_X).astype(np.float32)
    test_y = np.concatenate(test_y)
    
    # 归一化：将像素值从 0-255 缩放到 0-1
    train_X /= 255.0
    test_X /= 255.0
    
    return train_X, train_y, test_X, test_y

def softmax(z):
    """
    Softmax 函数: 将分数转换为概率分布
    为了数值稳定，通常减去最大值
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    """将标签转换为 one-hot 向量"""
    return np.eye(num_classes)[y]

def train_logistic_regression():
    # 1. 加载数据
    print("正在加载数据...")
    # 请确保 mnist_all.mat 在你的当前目录下
    try:
        X_train, y_train, X_test, y_test = load_data('mnist_all.mat')
    except FileNotFoundError:
        print("Error: missing 'mnist_all.mat' file.")
        return

    print(f"训练集维度: {X_train.shape}, 测试集维度: {X_test.shape}")
    
    # 2. 初始化参数
    # 输入维度 784，输出维度 10 (0-9)
    input_dim = 784
    num_classes = 10
    
    # 权重 W 初始化为小的随机数
    W = 0.01 * np.random.randn(input_dim, num_classes) 
    # 偏置 b 初始化为 0
    b = np.zeros((1, num_classes))
    
    # 超参数
    learning_rate = 0.1
    epochs = 30  # 训练轮数
    batch_size = 64  # 每一批处理 64 张图片
    
    N = X_train.shape[0]
    
    # 3. 开始训练循环
    loss_history = []
    
    print(f"开始训练 (Batch Size: {batch_size})...")
    for epoch in range(epochs):
        # 每轮开始前，打乱数据！(Shuffle)
        # 如果不打乱，模型每次都按固定顺序学习（比如先学一堆0，再学一堆1），会导致梯度震荡严重
        permutation = np.random.permutation(N)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0 # 记录这一轮的平均损失
        num_batches = 0

        # 内层循环：按 batch_size 切分数据
        for i in range(0, N, batch_size):
            # 获取当前批次的数据 (Slicing)
            # Python 切片很智能，即使最后剩的数据不足 64 个，它也会自动取到末尾
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            # 当前 batch 的实际大小 (最后一个 batch 可能小于 batch_size)
            current_batch_size = X_batch.shape[0]

            # --- 前向传播 (Forward Pass) ---
            z = np.dot(X_batch, W) + b # Z = X * W + b
            probs = softmax(z) # 计算预测概率 (Softmax)

            # --- 计算 Loss (仅用于监控) ---
            # 为了计算方便，加上一个极小值 1e-8 防止 log(0)
            y_batch_onehot = one_hot_encode(y_batch, num_classes)
            batch_loss = -np.mean(np.sum(y_batch_onehot * np.log(probs + 1e-8), axis=1))
            epoch_loss += batch_loss
            num_batches += 1

            # --- 反向传播 ---
            error = probs - y_batch_onehot
            dw = np.dot(X_batch.T, error) / current_batch_size
            db = np.sum(error, axis=0, keepdims=True) / current_batch_size

            # --- 参数更新 ---
            W = W - learning_rate * dw
            b = b - learning_rate * db

        # 记录每轮的平均 Loss
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # 每轮打印一次测试集准确率
        test_z = np.dot(X_test, W) + b
        pred_y = np.argmax(test_z, axis=1)
        accuracy = np.mean(pred_y == y_test)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Acc: {accuracy:.4f}")

    print("训练完成！")
    
    # 简单的可视化损失曲线
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    train_logistic_regression()