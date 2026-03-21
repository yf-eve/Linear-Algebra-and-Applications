import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 路径自动处理
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 固定随机种子 (确保实验可复现)
np.random.seed(42)

def load_data():
    train_path = os.path.join(DATA_DIR, "usps")
    test_path = os.path.join(DATA_DIR, "usps.t")
    
    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)
    
    # 核心修正：全体减 1，使范围从 [1, 10] 变为 [0, 9]
    y_train = y_train.astype(int) - 1
    y_test = y_test.astype(int) - 1
    
    return X_train.toarray(), y_train, X_test.toarray(), y_test

# --- 2. K-Means 核心算法 ---
class KMeansClassifier:
    def __init__(self, k=20):
        self.k = k
        self.centroids = None

    def fit(self, X):
        best_msd = float('inf')
        for attempt in range(10): # 运行10次取最优
            idx = np.random.choice(len(X), self.k, replace=False)
            curr_centroids = X[idx]
            for _ in range(50):
                dist = np.linalg.norm(X[:, np.newaxis] - curr_centroids, axis=2)
                labels = np.argmin(dist, axis=1)
                new_centroids = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 
                                         else curr_centroids[j] for j in range(self.k)])
                if np.allclose(curr_centroids, new_centroids): break
                curr_centroids = new_centroids
            
            msd = np.mean(np.min(np.linalg.norm(X[:, np.newaxis] - curr_centroids, axis=2), axis=1)**2)
            if msd < best_msd:
                best_msd = msd
                self.centroids = curr_centroids
        print(f"K-Means 训练完成，最优 MSD: {best_msd:.4f}")

# --- 3. 实验主逻辑 ---
X_train, y_train, X_test, y_test = load_data()
model = KMeansClassifier(k=20)
model.fit(X_train)

# 保存质心图预览
plt.figure(figsize=(12, 10))
for i in range(model.k):
    plt.subplot(4, 5, i+1)
    plt.imshow(model.centroids[i].reshape(16, 16), cmap='gray')
    plt.title(f"ID: {i}")
    plt.axis('off')
plt.savefig(os.path.join(RESULTS_DIR, "centroids_preview.png"))
print(f"\n[预览] 20个质心图已保存至: {RESULTS_DIR}/centroids_preview.png")


# --- 模式选择 ---
mode = input("\n请选择模式 (输入 1 为手动标记，输入 2 为自动投票): ").strip()

if mode == '1':
    print("\n--- 手动标记模式 ---")
    # 关键修改：使用非阻塞方式显示图片
    plt.show(block=False) 
    print("[提示] 请查看弹出的图片窗口，根据 ID 0-19 认字。")
    print("[提示] 输入完成后请按回车，最后再手动关闭图片窗口。")
    
    # 此时窗口不会消失，你可以看着图输入
    input_str = input("\n请依次输入 20 个数字（0-9），用空格隔开: ")
    
    try:
        centroid_labels = np.array([int(x) for x in input_str.split()])
        if len(centroid_labels) != 20:
            raise ValueError(f"数量不对！需要 20 个，你输入了 {len(centroid_labels)} 个。")
    except ValueError as e:
        print(f"输入错误: {e}")
        exit()
else:
    print("\n--- 自动聚类打标模式 ---")
    # 计算训练集到质心的距离来确定每个质心的“语义”
    dist_to_train = np.linalg.norm(X_train[:, np.newaxis] - model.centroids, axis=2)
    train_cluster_idx = np.argmin(dist_to_train, axis=1)
    
    centroid_labels = np.zeros(20, dtype=int)
    print("\n[机器诊断结果] ID vs 自动识别数字:")
    for j in range(20):
        # 找到属于这个簇的所有真实标签
        members = y_train[train_cluster_idx == j]
        if len(members) > 0:
            centroid_labels[j] = np.bincount(members).argmax()
        print(f"ID {j:02d}: 数字 {centroid_labels[j]}")

# --- 4. 预测与评估 ---
test_dist = np.linalg.norm(X_test[:, np.newaxis] - model.centroids, axis=2)
y_pred = centroid_labels[np.argmin(test_dist, axis=1)]

error_rate = np.mean(y_pred != y_test)
print(f"\n" + "="*30)
print(f"最终测试集错误率: {error_rate * 100:.2f}%")
print("="*30)

# --- 5. 生成混淆矩阵 ---
# 根据模式动态设置文件名
filename = "manual_cm.png" if mode == '1' else "auto_cm.png"
title_suffix = "Manual Mapping" if mode == '1' else "Automatic Voting"

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
# 注意：display_labels 确保是 0-9
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(ax=ax, cmap='Blues', values_format='d')

plt.title(f"Confusion Matrix [{title_suffix}]\n(Total Error: {error_rate*100:.2f}%)")
save_path = os.path.join(RESULTS_DIR, filename)
plt.savefig(save_path)

print(f"混淆矩阵图已保存至: {save_path}")
plt.show()