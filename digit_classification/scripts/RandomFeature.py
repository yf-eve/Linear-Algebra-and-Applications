import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix

# --- 1. 全局路径配置 ---
# 获取当前脚本所在目录，并定义相关文件夹路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "results")

# 确保结果保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 数据加载 ---
def load_data():
    train_path = os.path.join(DATA_DIR, "usps")
    test_path = os.path.join(DATA_DIR, "usps.t")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到数据集，请确认路径: {train_path}")

    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)
    
    # 标签从 [1, 10] 修正为 [0, 9]
    return X_train.toarray(), y_train.astype(int) - 1, X_test.toarray(), y_test.astype(int) - 1

# --- 3. 随机 ReLU 特征映射器 ---
class RandomReLUProjector:
    def __init__(self, d_in, d_out):
        # 使用 He 初始化缩放随机权重 R
        self.R = np.random.randn(d_out, d_in) * np.sqrt(2.0 / d_in)
        self.b = np.random.randn(d_out) * 0.1

    def transform(self, X):
        # 执行非线性激活: max(0, Rx + b)
        projected = X @ self.R.T + self.b
        return np.maximum(0, projected)

# --- 4. 十分类最小二乘求解器 ---
class MultiClassLSClassifier:
    def __init__(self, lmbda):
        self.lmbda = lmbda
        self.W_tilde = None

    def fit(self, X, y):
        N, d = X.shape
        A = np.hstack([X, np.ones((N, 1))])
        Y_multi = np.ones((N, 10)) * -1
        for k in range(10):
            Y_multi[y == k, k] = 1
        
        reg = self.lmbda * np.eye(d + 1)
        reg[-1, -1] = 0 # 不惩罚偏置项
        self.W_tilde = np.linalg.solve(A.T @ A + reg, A.T @ Y_multi)

    def predict(self, X):
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        scores = A @ self.W_tilde
        return np.argmax(scores, axis=1)

# --- 5. 实验流程 ---
if __name__ == "__main__":
    X_train_raw, y_train, X_test_raw, y_test = load_data()

    # 特征扩张：从 256 维增加到 2000 维
    print("正在执行随机 ReLU 特征映射 (256 -> 2000)...")
    projector = RandomReLUProjector(d_in=256, d_out=2000)
    X_train = projector.transform(X_train_raw)
    X_test = projector.transform(X_test_raw)

    # 扫描 Lambda
    lambdas = np.logspace(-1, 4, 15)
    train_errors = []
    test_errors = []

    print("开始 Lambda 扫描以寻找最优正则化系数...")
    for l in lambdas:
        clf = MultiClassLSClassifier(lmbda=l)
        clf.fit(X_train, y_train)
        train_errors.append(np.mean(clf.predict(X_train) != y_train))
        test_errors.append(np.mean(clf.predict(X_test) != y_test))
        print(f"Lambda: {l:8.2f} | Test Error: {test_errors[-1]*100:5.2f}%")

    # --- 6. 绘图与保存 ---
    best_idx = np.argmin(test_errors)
    best_lambda = lambdas[best_idx]

    # 图像 A: 错误率曲线
    plt.figure(figsize=(10, 6))
    plt.semilogx(lambdas, train_errors, 'r-o', label='Train Error')
    plt.semilogx(lambdas, test_errors, 'b-s', label='Test Error')
    plt.plot(best_lambda, test_errors[best_idx], 'go', markersize=12, label='Best Performance')
    
    plt.xlabel('Lambda')
    plt.ylabel('Error Rate')
    plt.title(f'Random ReLU Features: Error vs Lambda\n(Min Test Error: {test_errors[best_idx]*100:.2f}%)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    curve_path = os.path.join(SAVE_DIR, "random_relu_error_curve.png")
    plt.savefig(curve_path)
    print(f"错误率曲线已保存至: {curve_path}")
    plt.show()

    # 图像 B: 混淆矩阵
    print(f"\n生成最优 Lambda ({best_lambda:.2f}) 下的混淆矩阵...")
    final_clf = MultiClassLSClassifier(lmbda=best_lambda)
    final_clf.fit(X_train, y_train)
    y_pred = final_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.title(f'Confusion Matrix (Random ReLU)\nTest Error: {test_errors[best_idx]*100:.2f}%')
    
    cm_path = os.path.join(SAVE_DIR, "random_relu_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"混淆矩阵已保存至: {cm_path}")
    plt.show()