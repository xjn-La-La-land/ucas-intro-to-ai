"""
实现决策树 (Decision Tree) 与 Bagging/Boosting 集成学习模型来训练 MNIST 数据集
"""

import numpy as np
import scipy.io as sio
import time
import random
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from typing import Union, Optional
from skimage.feature import hog # 导入 HOG 特征提取器

def load_data(mat_file_path):
    """
    读取 mnist_all.mat 文件并整合成训练集和测试集
    进行简单的归一化 (0-1)
    """
    if not os.path.exists(mat_file_path):
        print(f"Error: file '{mat_file_path}' not found.")
        return None, None, None, None

    print(f"正在加载数据: {mat_file_path} ...")
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
        train_y.append(np.full(curr_train_data.shape[0], i))
        
        # 读取测试数据
        curr_test_data = data[f'test{i}']
        test_X.append(curr_test_data)
        test_y.append(np.full(curr_test_data.shape[0], i))
        
    train_X = np.vstack(train_X).astype(np.float32)
    train_y = np.concatenate(train_y)
    test_X = np.vstack(test_X).astype(np.float32)
    test_y = np.concatenate(test_y)
    
    # 归一化 0-1
    train_X /= 255.0
    test_X /= 255.0
    
    print(f"数据加载完成: 训练集 {train_X.shape}, 测试集 {test_X.shape}")
    return train_X, train_y, test_X, test_y


def extract_hog_features(images, hog_orientations=9, hog_pixels_per_cell=(4, 4), hog_cells_per_block=(2, 2)):
    """
    将一批灰度图像转换为 HOG 特征向量
    """
    print(f"正在提取 HOG 特征 (Orientations={hog_orientations}, "
            f"Pixels per Cell={hog_pixels_per_cell}, "
            f"Cells per Block={hog_cells_per_block})...")
    
    hog_features = []
    for img in images:
        # 重塑为 28x28 图像
        img_2d = img.reshape(28, 28)
        
        # 提取 HOG 特征
        # visualize=True 用于调试，feature_vector=True 返回扁平的特征向量
        features = hog(img_2d, 
                        orientations=hog_orientations, 
                        pixels_per_cell=hog_pixels_per_cell,
                        cells_per_block=hog_cells_per_block, 
                        visualize=False,
                        feature_vector=True)
        hog_features.append(features)
        
    print(f"HOG 特征提取完成，维度: {hog_features[0].shape[0]}")
    return np.array(hog_features)


class MNISTDTClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, pca_components: Optional[Union[int, float]] = None):
        """
        初始化决策树分类器
        Args:
            criterion: 'gini' (基尼系数) 或 'entropy' (信息增益)
            max_depth: 树的最大深度，None 表示不限制（容易过拟合）
            min_samples_split: 拆分内部节点所需的最小样本数
            pca_components: PCA 降维参数。如果是 int，表示保留的维度数；如果是 float (0-1)，表示保留的方差比例。
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.pca_components = pca_components
        self.model = None
        self.pca = None

    def train(self, X, y):
        # HOG 特征提取
        X = extract_hog_features(X)
        print(f"HOG 特征数据维度: {X.shape}")
        # PCA 预处理
        if self.pca_components:
            print(f"应用 PCA 降维 (components={self.pca_components})...")
            self.pca = PCA(n_components=self.pca_components)
            X = self.pca.fit_transform(X)
            print(f"PCA 完成，保留的累计方差: {np.sum(self.pca.explained_variance_ratio_):.4f}")
            print(f"降维后特征维度: {X.shape[1]}")

        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        
        print(f"开始训练决策树 (Criterion={self.criterion}, Max Depth={self.max_depth})...")
        start_time = time.time()
        self.model.fit(X, y)
        print(f"训练完成！耗时: {time.time() - start_time:.2f} 秒")

    def evaluate(self, X, y):
        if self.model is None:
            print("错误：模型尚未训练")
            return

        X = extract_hog_features(X)
        # 如果训练时用了 PCA，测试时也要用同样的 PCA 转换
        if self.pca:
            X = self.pca.transform(X)
            
        print("正在评估...")
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print("-" * 30)
        print(f"测试集准确率: {acc:.4f}")
        print("-" * 30)

class MNISTBaggingClassifier:
    """
    Bagging(Bootstrap Aggregating) 集成学习分类器
    原理：并行训练多个决策树，通过投票机制降低方差 (Variance)
    """
    def __init__(self, n_estimators=10, max_depth=None, pca_components=None):
        """
        初始化 Bagging 分类器
        Args:
            n_estimators: 集成中基模型的数量（树的棵数）
            max_depth: 每个基模型（决策树）的最大深度
            pca_components: PCA 降维参数
        """
        self.n_estimators = n_estimators

        self.max_depth = max_depth
        
        self.pca_components = pca_components
        self.model = None
        self.pca = None

    def train(self, X, y):
        # HOG 特征提取
        X = extract_hog_features(X)
        print(f"HOG 特征数据维度: {X.shape}")
        # PCA 预处理 (可选)
        if self.pca_components:
            print(f"应用 PCA 降维 (components={self.pca_components})...")
            self.pca = PCA(n_components=self.pca_components)
            X = self.pca.fit_transform(X)
            print(f"PCA 完成，保留的累计方差: {np.sum(self.pca.explained_variance_ratio_):.4f}")
            print(f"降维后特征维度: {X.shape[1]}")
        
        # 定义基分类器 (Base Estimator): Bagging 的核心是若干个“弱”但“不稳定”的分类器
        base_dt = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion='gini',
            min_samples_split=random.randint(10, 20),
            random_state=None # 注意：这里不要固定随机种子，否则每棵树都一样了！
        )

        # 定义 Bagging 模型
        self.model = BaggingClassifier(
            estimator=base_dt,                # 基模型是决策树
            n_estimators=self.n_estimators,   # 树的数量
            max_samples=random.uniform(0.5, 1.0),  # 随机选择样本比例
            max_features=random.uniform(0.5, 1.0), # 随机选择特征比例
            bootstrap=True,                   # 开启自助采样 (Bootstrap)
            n_jobs=-1,                        # [系统架构重点] 使用所有 CPU 核心并行训练
            random_state=None                 # 不固定随机种子，保证多样性
        )

        print(f"开始并行训练 {self.n_estimators} 个基模型 (n_jobs=-1)...")
        start_time = time.time()
        self.model.fit(X, y)
        print(f"训练完成！耗时: {time.time() - start_time:.2f} 秒")

    def evaluate(self, X, y):
        if self.model is None:
            print("错误：模型尚未训练")
            return
        X = extract_hog_features(X)
        # 如果训练时用了 PCA，测试时也要用同样的 PCA 转换
        if self.pca: X = self.pca.transform(X)
        
        print("正在评估...")
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print("-" * 30)
        print(f"测试集准确率: {acc:.4f}")
        print("-" * 30)


class MNISTAdaBoostClassifier:
    """
    AdaBoost (Adaptive Boosting) 集成学习分类器
    原理：通过迭代训练多个弱分类器（如决策树），并根据分类错误率调整样本权重，逐步提高整体分类性能。
    """
    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=1, pca_components=None):
        """
        初始化 AdaBoost 分类器
        Args:
            n_estimators: 弱分类器的数量
            learning_rate: 学习率，用于缩放每个弱分类器的贡献
            max_depth: 每个弱分类器（决策树）的最大深度
            pca_components: PCA 降维参数
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.pca_components = pca_components
        self.model = None
        self.pca = None
    
    def train(self, X, y):
        # HOG 特征提取
        X = extract_hog_features(X)
        print(f"HOG 特征数据维度: {X.shape}")
        # PCA 预处理 (可选)
        if self.pca_components:
            print(f"应用 PCA 降维 (components={self.pca_components})...")
            self.pca = PCA(n_components=self.pca_components)
            X = self.pca.fit_transform(X)
            print(f"PCA 完成，保留的累计方差: {np.sum(self.pca.explained_variance_ratio_):.4f}")
            print(f"降维后特征维度: {X.shape[1]}")

        # 定义基分类器 (Base Estimator)
        base_dt = DecisionTreeClassifier(
            max_depth=self.max_depth,  # 弱分类器的深度通常较小
            criterion='gini',
            min_samples_split=random.randint(10, 20),
            random_state=None # 注意：这里不要固定随机种子，否则每棵树都一样了！
        )

        # 定义 AdaBoost 模型
        self.model = AdaBoostClassifier(
            estimator=base_dt,              # 基分类器是决策树
            n_estimators=self.n_estimators, # 弱分类器的数量
            learning_rate=self.learning_rate, # 学习率
        )

        print(f"开始训练 AdaBoost 模型 (n_estimators={self.n_estimators}, learning_rate={self.learning_rate})...")
        start_time = time.time()
        self.model.fit(X, y)
        print(f"训练完成！耗时: {time.time() - start_time:.2f} 秒")

    def evaluate(self, X, y):
        if self.model is None:
            print("错误：模型尚未训练")
            return

        X = extract_hog_features(X)
        # 如果训练时用了 PCA，测试时也要用同样的 PCA 转换
        if self.pca:
            X = self.pca.transform(X)

        print("正在评估...")
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print("-" * 30)
        print(f"测试集准确率: {acc:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_data('mnist_all.mat')
    
    if train_X is not None:
        # 单棵决策树
        # dt = MNISTDTClassifier(
        #     criterion='entropy',       # 使用信息增益，通常在多分类任务上比 gini 稍好
        #     max_depth=20,              # 限制树深度，防止过拟合（死记硬背）
        #     min_samples_split=20,      # 增加节点分裂所需的最小样本数，减少对噪声的敏感度
        #     pca_components=0.85        # PCA 降维：保留 85% 方差。对决策树来说，特征正交化比保留所有细节更重要
        # )
        
        # dt.train(train_X, train_y)
        # dt.evaluate(test_X, test_y)

        # print("\n" + "="*40 + "\n")

        # Bagging 集成学习
        # bagging = MNISTBaggingClassifier(
        #     n_estimators=100,       # 训练 100 棵树
        #     max_depth=25,           # 稍微加深深度，允许基模型更复杂
        #     pca_components=0.85
        # )

        # bagging.train(train_X, train_y)
        # bagging.evaluate(test_X, test_y)

        # print("\n" + "="*40 + "\n")

        # AdaBoost 集成学习
        adaboost = MNISTAdaBoostClassifier(
            n_estimators=200,      # 弱分类器数量
            learning_rate=0.2,     # 学习率
            max_depth=10,          # 弱分类器的最大深度
            pca_components=None
        )

        adaboost.train(train_X, train_y)
        adaboost.evaluate(test_X, test_y)