import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_svmlight_file

# --- 1. 数据加载 (包含标签修正 [1,10]->[0,9]) ---
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    train_path = os.path.join(base_dir, "data", "usps")
    test_path = os.path.join(base_dir, "data", "usps.t")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到数据集文件，请检查路径: {train_path}")

    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)
    
    # 转换为稠密矩阵并修正标签偏移
    return X_train.toarray(), y_train.astype(int) - 1, X_test.toarray(), y_test.astype(int) - 1

# --- 2. 二元最小二乘分类器类 ---
class BinaryLeastSquaresClassifier:
    def __init__(self, target_digit, lmbda):
        self.target_digit = target_digit
        self.lmbda = lmbda
        self.w_tilde = None

    def fit(self, X, y):
        N, d = X.shape
        # 构造增强矩阵 A = [X, 1]
        A = np.hstack([X, np.ones((N, 1))])
        # 目标数字为 +1, 其他所有数字为 -1
        y_bin = np.where(y == self.target_digit, 1, -1)
        
        # 岭回归解析解: theta = (A^T A + lambda * I)^-1 * A^T * y
        reg = self.lmbda * np.eye(d + 1)
        reg[-1, -1] = 0  # 按照惯例，不对偏移项 v 进行正则化惩罚
        
        # 使用 solve 比直接求逆更稳定
        self.w_tilde = np.linalg.solve(A.T @ A + reg, A.T @ y_bin)

    def evaluate(self, X, y):
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        y_bin = np.where(y == self.target_digit, 1, -1)
        # 预测公式: sign(A * w_tilde)
        y_pred = np.sign(A @ self.w_tilde)
        # 返回分类错误率 (Classification Error)
        return np.mean(y_pred != y_bin)

# --- 3. 核心实验与绘图逻辑 ---
def run_detailed_experiment(target, chosen_lmbda):
    X_train, y_train, X_test, y_test = load_data()
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    # A. 扫描 Lambda 范围以生成对比曲线
    lambdas = np.logspace(0, 5, 30)
    train_errors = []
    test_errors = []
    
    print(f"\n正在为数字 {target} 扫描 Lambda 空间...")
    for l in lambdas:
        clf = BinaryLeastSquaresClassifier(target, l)
        clf.fit(X_train, y_train)
        train_errors.append(clf.evaluate(X_train, y_train))
        test_errors.append(clf.evaluate(X_test, y_test))

    # B. 针对指定的 lambda 进行最终训练与评估
    final_clf = BinaryLeastSquaresClassifier(target, chosen_lmbda)
    final_clf.fit(X_train, y_train)
    
    final_train_err = final_clf.evaluate(X_train, y_train)
    final_test_err = final_clf.evaluate(X_test, y_test)

    # --- 关键输出：在终端打印错误率 ---
    print("="*50)
    print(f"数字 {target} 二元分类结果 (Lambda = {chosen_lmbda}):")
    print(f"训练集错误率 (Train CE): {final_train_err * 100:.4f}%")
    print(f"测试集错误率 (Test CE):  {final_test_err * 100:.4f}%")
    print("="*50)

    # C. 保存【权重图】
    w_img = final_clf.w_tilde[:-1].reshape(16, 16)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(w_img, cmap='RdBu_r')
    plt.colorbar(im)
    plt.title(f"Weight Vector\nDigit: {target}, Lambda: {chosen_lmbda}, TestErr: {final_test_err*100:.2f}%")
    
    weight_path = os.path.join(results_dir, f"digit{target}_lambda{int(chosen_lmbda)}_weight.png")
    plt.savefig(weight_path)
    plt.close()

    # D. 保存【错误率曲线图】(红线为训练，蓝线为测试)
    plt.figure(figsize=(8, 6))
    plt.semilogx(lambdas, train_errors, 'r-o', markersize=4, label='Train Classification Error')
    plt.semilogx(lambdas, test_errors, 'b-s', markersize=4, label='Test Classification Error')
    
    # 用绿色圆点标出当前选择的 Lambda
    plt.plot(chosen_lmbda, final_test_err, 'go', markersize=10, label=f'Current Lambda ({chosen_lmbda})')
    
    plt.xlabel('Regularization Parameter (lambda)')
    plt.ylabel('Error Rate')
    plt.title(f"Error vs Regularization (Target Digit: {target})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    curve_path = os.path.join(results_dir, f"digit{target}_lambda{int(chosen_lmbda)}_curve.png")
    plt.savefig(curve_path)
    plt.close()

    print(f"图片保存成功：\n1. {weight_path}\n2. {curve_path}\n")

# --- 4. 运行 ---
if __name__ == "__main__":
    # 你可以尝试修改数字 target (0-9) 和 lambda 值
    run_detailed_experiment(target=9, chosen_lmbda=100)