"""
实现线性支持向量机（SVM）和高斯核支持向量机来训练和评估 MNIST 数据集
"""


import numpy as np
import scipy.io as sio
import time
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Literal, Union
from skimage.feature import hog # 导入 HOG 特征提取器
from skimage import exposure # 用于 HOG 图像增强（可选）

def load_data(mat_file_path, preprocesser:Union[None, StandardScaler, MinMaxScaler]=None):
    """
    读取 mnist_all.mat 文件并整合成训练集和测试集
    并且进行归一化处理（0-1）
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
    
    if preprocesser:
        print(f"应用预处理器: {preprocesser.__class__.__name__}...")
        train_X = preprocesser.fit_transform(train_X)
        test_X = preprocesser.transform(test_X)

    print(f"训练集样本数: {train_X.shape[0]}, 测试集样本数: {test_X.shape[0]}")
    
    return train_X, train_y, test_X, test_y


class MNISTSVMClassifier:
    def __init__(self, mode='linear', C=1.0, 
                 gamma: Union[float, Literal['scale', 'auto']] = 'scale', 
                 cache_size=1000, max_iter=2000,
                 use_hog=False, hog_orientations=9, 
                 hog_pixels_per_cell=(4, 4), hog_cells_per_block=(2, 2)):
        """
        初始化 SVM 分类器
        
        Args:
            mode (str): 'linear' (线性核) 或 'rbf' (高斯核)
            C (float): 惩罚系数。越大越不能容忍错误（易过拟合），越小越宽容（易欠拟合）。
            gamma (str/float): RBF 核的参数。'scale' 是推荐的默认值。
            cache_size (int): 指定使用的内存缓存大小 (MB)，仅对 RBF 核有效。
            max_iter (int): 最大迭代次数，防止不收敛死循环。
            use_hog (bool): 是否使用 HOG 特征。
            hog_orientations (int): HOG 参数，方向数。
            hog_pixels_per_cell (tuple): HOG 参数，每个 cell 的像素数。
            hog_cells_per_block (tuple): HOG 参数，每个 block 的 cell 数。
        """
        self.mode = mode
        self.C = C
        self.gamma = gamma
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.model = None # 稍后在 train 中实例化
        self.training_time = 0

        self.use_hog = use_hog
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        
        print(f"初始化分类器: Mode={self.mode}, C={self.C}, Use HOG={self.use_hog}")
    
    def _build_model(self):
        """
        内部方法：根据 mode 构建对应的 sklearn 模型对象
        系统优化视角：LinearSVC 使用 liblinear，SVC 使用 libsvm。
        """
        if self.mode == 'linear':
            # LinearSVC 对线性分类做了极致优化，比 SVC(kernel='linear') 快得多
            # dual=False: 当 n_samples > n_features 时推荐设为 False
            return svm.LinearSVC(C=self.C, dual=False, max_iter=self.max_iter)
        
        elif self.mode == 'rbf':
            # 标准的核 SVM
            return svm.SVC(kernel='rbf', C=self.C, max_iter=self.max_iter, gamma=self.gamma, cache_size=self.cache_size)
        
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
        
    def _extract_hog_features(self, images):
        """
        将一批灰度图像转换为 HOG 特征向量
        """
        print(f"正在提取 HOG 特征 (Orientations={self.hog_orientations}, "
              f"Pixels per Cell={self.hog_pixels_per_cell}, "
              f"Cells per Block={self.hog_cells_per_block})...")
        
        hog_features = []
        for img in images:
            # 重塑为 28x28 图像
            img_2d = img.reshape(28, 28)
            
            # 提取 HOG 特征
            # visualize=True 用于调试，feature_vector=True 返回扁平的特征向量
            features = hog(img_2d, 
                           orientations=self.hog_orientations, 
                           pixels_per_cell=self.hog_pixels_per_cell,
                           cells_per_block=self.hog_cells_per_block, 
                           visualize=False,
                           feature_vector=True)
            hog_features.append(features)
            
        print(f"HOG 特征提取完成，维度: {hog_features[0].shape[0]}")
        return np.array(hog_features)
    
    def load_data(self, mat_file_path,
                  preprocesser: Union[None, StandardScaler, MinMaxScaler]=None,
                  pca_components: Union[None, int]=None):
        """
        读取并预处理数据，可选是否提取 HOG 特征
        Args:
            mat_file_path (str): mnist_all.mat 文件路径
            preprocesser: (可选) sklearn 预处理器实例，如 StandardScaler 或 MinMaxScaler
            pca_components: (可选) PCA 降维后的维度数
        """
        try:
            train_X, train_y, test_X, test_y = load_data(mat_file_path, preprocesser)
        except FileNotFoundError:
            print(f"Error: missing {mat_file_path} file.")
            return None, None, None, None
        
        if self.use_hog:
            train_X = self._extract_hog_features(train_X)
            test_X = self._extract_hog_features(test_X)
            print(f"HOG 特征数据维度: 训练集 {train_X.shape}, 测试集 {test_X.shape}")
        
        if pca_components:
            print(f"应用 PCA 降维到 {pca_components} 维...")
            pca = PCA(n_components=pca_components)
            train_X = pca.fit_transform(train_X)
            test_X = pca.transform(test_X)
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"PCA 降维完成！保留的累计方差：{explained_variance:.4f}")
        
        return train_X, train_y, test_X, test_y

        

    def train(self, X, y, sample_limit=None):
        """
        训练模型
        Args:
            X, y: 训练数据和标签
            sample_limit: (int, 可选) 仅使用前 N 个样本训练，用于快速调试 RBF 核
        """
        self.model = self._build_model()
        
        # 如果指定了样本限制（主要针对 RBF 核太慢的问题）
        if sample_limit and sample_limit < X.shape[0]:
            print(f"警告：仅使用前 {sample_limit} 个样本进行训练...")
            X_subset = X[:sample_limit]
            y_subset = y[:sample_limit]
        else:
            X_subset = X
            y_subset = y

        print(f"开始训练 ({self.mode})...")
        start_time = time.time()
        
        self.model.fit(X_subset, y_subset)
        
        self.training_time = time.time() - start_time
        print(f"训练完成！耗时: {self.training_time:.2f} 秒")

    def evaluate(self, test_X, test_y):
        """
        评估模型性能
        """
        if self.model is None:
            print("错误：模型尚未训练，请先调用 train()")
            return
            
        print(f"正在评估 ({self.mode})...")
        y_pred = self.model.predict(test_X)
        acc = accuracy_score(test_y, y_pred)
        
        print("-" * 30)
        print(f"模式: {self.mode}")
        print(f"测试集准确率: {acc:.4f}")
        print("-" * 30)
        return acc
    

if __name__ == "__main__":

    # svm_linear = MNISTSVMClassifier(mode='linear', C=1.0, max_iter=2000)
    # train_X, train_y, test_X, test_y = svm_linear.load_data('mnist_all.mat', preprocesser=MinMaxScaler())
    # svm_linear.train(train_X, train_y)
    # svm_linear.evaluate(test_X, test_y)

    # print("\n" + "="*40 + "\n")

    svm_rbf = MNISTSVMClassifier(
        mode='rbf', 
        C=50.0,            # (通常 RBF 需要更大的 C)
        gamma="scale", 
        max_iter=20000, 
        use_hog=True,      # 启用 HOG 特征
        hog_orientations=9,
        hog_pixels_per_cell=(4, 4),
        hog_cells_per_block=(2, 2)
    ) 
    train_X, train_y, test_X, test_y = svm_rbf.load_data(
        'mnist_all.mat', 
        preprocesser=StandardScaler(), 
        pca_components=200 # 用 PCA 降维加速 rbf 训练
    ) 
    svm_rbf.train(train_X, train_y)
    svm_rbf.evaluate(test_X, test_y)

