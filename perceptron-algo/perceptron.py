import re
import sys
import numpy as np

def parse_line(line):
    """从一行文本中解析出样本点"""
    pattern = re.compile(r'\((.*?)\)')
    matches = pattern.findall(line)
    samples = []
    for m in matches:
        nums = [float(x.strip()) for x in m.split(',')]
        samples.append(nums)
    return samples

def perceptron_train(X, max_iter=1000, C=1.0):
    """感知器训练算法
    X: 所有样本（已增广，ω2 已乘 -1）
    max_iter: 最大迭代次数
    返回 (w, epoch) 或 (None, max_iter)
    """
    n_features = X.shape[1]
    # w = np.zeros(n_features) # 使用零向量初始化权重
    # w = np.ones(n_features)  # 使用全1向量初始化权重
    w = np.random.rand(n_features)  # 使用随机向量初始化权重

    print(f"初始权重向量 w = ({', '.join(f'{v:.2g}' for v in w)})")
    print(f"最大迭代次数 = {max_iter}, 学习率 C = {C}")

    for epoch in range(1, max_iter + 1):
        error_count = 0
        for x in X:
            if np.dot(w, x) <= 0:
                w = w + C * x
                error_count += 1
        if error_count == 0:
            return w, epoch
    return None, max_iter  # 未收敛

def main():
    # === 检查命令行参数 ===
    if len(sys.argv) != 2:
        print("用法: python perceptron.py <input_file>")
        sys.exit(1)

    filename = sys.argv[1]

    # === 读取输入文件 ===
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"文件未找到: {filename}")
        sys.exit(1)

    if len(lines) < 2:
        print("输入文件格式错误，应包含两行。")
        sys.exit(1)

    omega1 = parse_line(lines[0])
    omega2 = parse_line(lines[1])

    # === 构造增广向量并区分两类 ===
    X = []
    for x in omega1:
        X.append(np.array(x + [1.0]))  # ω1 增广
    for x in omega2:
        X.append(-1 * np.array(x + [1.0]))  # ω2 乘以 -1

    X = np.array(X)

    # === 训练感知器 ===
    w, epoch = perceptron_train(X)

    if w is not None:
        formatted_w = "(" + ", ".join(f"{v:.2g}" for v in w) + ")"
        print(f"解向量 w = {formatted_w}")
        print(f"收敛迭代次数 = {epoch}")
    else:
        print("failed")

if __name__ == "__main__":
    main()
