import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import convolve
import os

def run_edge_experiment(image_path):
    # 1. 加载并标准化
    img_raw = io.imread(image_path)
    
    # 通道预处理 (处理 RGBA)
    if img_raw.ndim == 3 and img_raw.shape[-1] == 4:
        img_raw = img_raw[:, :, :3]
    
    # 转为灰度图：边缘提取通常在灰度图上进行
    img = color.rgb2gray(img_raw)
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # --- 方法一：NumPy 矩阵切片 (对应讲义的 D 矩阵逻辑) ---
    # 水平差分：Yij = Xi,j+1 - Xi,j
    # 注意：结果矩阵会比原图少一列
    diff_h_slice = img[:, 1:] - img[:, :-1]
    
    # 垂直差分：Zij = Xi+1,j - Xi,j
    # 注意：结果矩阵会比原图少一行
    diff_v_slice = img[1:, :] - img[:-1, :]

    # --- 方法二：卷积实现 (更通用的工程做法) ---
    # 水平算子：[-1, 1]
    kernel_h = np.array([[-1, 1]]) 
    # 垂直算子：[[-1], [1]]
    kernel_v = np.array([[-1], [1]])
    
    img_diff_h = convolve(img, kernel_h)
    img_diff_v = convolve(img, kernel_v)
    
    # --- 进阶：计算梯度幅值 (Combined Edge) ---
    # 结合水平和垂直边缘，看到完整的轮廓
    # 类似于几何上的勾股定理：sqrt(dh^2 + dv^2)
    img_edge = np.sqrt(img_diff_h**2 + img_diff_v**2)

    # --- 2. 存储结果 ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 这里的 cmap='gray' 非常重要，因为差分会有负数
    '''plt.imsave(os.path.join(results_dir, f"{file_base}_Edge_H.png"), img_diff_h, cmap='gray')
    plt.imsave(os.path.join(results_dir, f"{file_base}_Edge_V.png"), img_diff_v, cmap='gray')
    plt.imsave(os.path.join(results_dir, f"{file_base}_Edge_Full.png"), img_edge, cmap='gray')

    # --- 3. 可视化对比 ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    display_list = [
        (img, "Original"),
        (img_diff_h, "Horizontal Diff (Vertical Edges)"),
        (img_diff_v, "Vertical Diff (Horizontal Edges)"),
        (img_edge, "Gradient Magnitude (Full Edges)")
    ]'''
    
    # 使用 np.abs() 处理
    plt.imsave(os.path.join(results_dir, f"{file_base}_Edge_H_Abs.png"), np.abs(img_diff_h), cmap='gray')
    plt.imsave(os.path.join(results_dir, f"{file_base}_Edge_V_Abs.png"), np.abs(img_diff_v), cmap='gray')
    plt.imsave(os.path.join(results_dir, f"{file_base}_Edge_Full.png"), img_edge, cmap='gray')

    # --- 3. 可视化对比 (修改处：传入 np.abs 版本) ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    display_list = [
        (img, "Original"),
        (np.abs(img_diff_h), "Abs Horizontal Diff"),
        (np.abs(img_diff_v), "Abs Vertical Diff"),
        (img_edge, "Gradient Magnitude")
    ]

    for i, (data, title) in enumerate(display_list):
        axes[i].imshow(data, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{file_base}_Edge_Summary.png"))
    print(f"边缘提取完成！结果存入: {results_dir}")
    plt.show()

if __name__ == "__main__":
    img_path = "../data/cameraman.png"
    if os.path.exists(img_path):
        run_edge_experiment(img_path)