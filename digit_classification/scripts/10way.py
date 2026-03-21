import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix

# --- 1. 数据加载 ---
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "usps")
    test_path = os.path.join(base_dir, "data", "usps.t")
    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)
    return X_train.toarray(), y_train.astype(int) - 1, X_test.toarray(), y_test.astype(int) - 1

# --- 2. 十分类逻辑 ---
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
        reg[-1, -1] = 0
        self.W_tilde = np.linalg.solve(A.T @ A + reg, A.T @ Y_multi)

    def predict(self, X):
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        scores = A @ self.W_tilde
        return np.argmax(scores, axis=1)

# --- 3. 运行 Lambda 扫描 ---
X_train, y_train, X_test, y_test = load_data()
lambdas = np.logspace(0, 4, 20)
train_errors = []
test_errors = []

print("开始十分类 Lambda 扫描...")
for l in lambdas:
    clf = MultiClassLSClassifier(lmbda=l)
    clf.fit(X_train, y_train)
    train_errors.append(np.mean(clf.predict(X_train) != y_train))
    test_errors.append(np.mean(clf.predict(X_test) != y_test))

# --- 4. 绘制错误率曲线 ---
plt.figure(figsize=(10, 6))
plt.semilogx(lambdas, train_errors, 'r-o', label='Train Multi-class Error')
plt.semilogx(lambdas, test_errors, 'b-s', label='Test Multi-class Error')
best_idx = np.argmin(test_errors)
best_lambda = lambdas[best_idx]
plt.plot(best_lambda, test_errors[best_idx], 'go', markersize=10)
plt.title(f'10-Way Classification Error (Best Lambda: {best_lambda:.1f})')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()

# --- 5. 重点：使用最优 Lambda 计算并绘制混淆矩阵 ---
print(f"\n正在使用最优 Lambda ({best_lambda:.2f}) 生成混淆矩阵...")
final_clf = MultiClassLSClassifier(lmbda=best_lambda)
final_clf.fit(X_train, y_train)
y_pred = final_clf.predict(X_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label (预测结果)')
plt.ylabel('True Label (真实结果)')
plt.title(f'Confusion Matrix at Lambda = {best_lambda:.1f}\n(Test Error: {test_errors[best_idx]*100:.2f}%)')

# 保存混淆矩阵图片
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "multiclass_confusion_matrix.png"))
plt.show()