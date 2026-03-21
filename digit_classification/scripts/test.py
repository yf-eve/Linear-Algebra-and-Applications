import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_svmlight_file

# --- 1. 数据加载逻辑 (确保路径正确) ---
def load_data():
    # 自动获取路径，请确保你的数据在 scripts/../data/ 目录下
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
    test_path = os.path.join(DATA_DIR, "usps.t")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集文件: {test_path}")

    X_test, y_test = load_svmlight_file(test_path)
    X_test = X_test.toarray()
    y_test = y_test.astype(int)

    # 关键：暂时先注释掉 10->0 的映射，看看原始标签到底是什么
    # y_test[y_test == 10] = 0 
    
    return X_test, y_test

# --- 2. 核心审计逻辑 ---
def audit_dataset_labels(X, y, num_samples=10):
    unique_labels = np.unique(y)
    unique_labels.sort()
    
    print(f"检测到数据集中的唯一标签: {unique_labels}")
    
    n_classes = len(unique_labels)
    # 创建画布
    fig, axes = plt.subplots(n_classes, num_samples, figsize=(num_samples * 1.2, n_classes * 1.5))
    
    for row, label in enumerate(unique_labels):
        indices = np.where(y == label)[0]
        selected_indices = indices[:num_samples]
        
        for col in range(num_samples):
            ax = axes[row, col]
            if col < len(selected_indices):
                idx = selected_indices[col]
                ax.imshow(X[idx].reshape(16, 16), cmap='gray')
            ax.axis('off')
            
            # 在每行开头添加标签文字
            if col == 0:
                ax.text(-10, 8, f"Label: {int(label)}", fontsize=12, fontweight='bold', ha='right')

    plt.suptitle("USPS 数据集原始标签对照表 (每一行应该是同一个数字)", fontsize=16)
    plt.tight_layout(rect=[0.1, 0.03, 1, 0.95])
    plt.show()

# --- 3. 执行 ---
try:
    X_test, y_test = load_data()
    print("正在生成标签审计图...")
    audit_dataset_labels(X_test, y_test)
except Exception as e:
    print(f"运行失败: {e}")