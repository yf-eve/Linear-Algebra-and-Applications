import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import convolve
import os

def run_blur_experiment(image_path, box_size=7, sigma=3):
    # 1. 加载图片并转为灰度（卷积在单通道上最容易理解）
    img = io.imread(image_path)

    # 如果有 4 个通道 (RGBA)，只取前 3 个 (RGB)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    if img.ndim == 3:
        img = color.rgb2gray(img)
    
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # --- A. 均值模糊 (Box Blur) ---
    # 构建 B 矩阵：所有元素相等且和为 1
    # 对应讲义：Yij = ∑ Xi-k+1,j-l+1 * Bkl
    kernel_box = np.ones((box_size, box_size)) / (box_size**2)
    img_box = convolve(img, kernel_box)
    
    # --- B. 高斯模糊 (Gaussian Blur) ---
    # 构建高斯核：中间高，四周低，模拟光学点扩散函数 (PSF)
    ax = np.linspace(-(box_size // 2), box_size // 2, box_size)
    gauss = np.exp(-0.5 * (ax / sigma)**2)
    kernel_gauss = np.outer(gauss, gauss)
    kernel_gauss /= kernel_gauss.sum() # 归一化，保持亮度能量守恒
    img_gauss = convolve(img, kernel_gauss)

    # --- 2. 存储结果 ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    plt.imsave(os.path.join(results_dir, f"{file_base}_BoxBlur_k{box_size}.png"), img_box, cmap='gray')
    plt.imsave(os.path.join(results_dir, f"{file_base}_GaussBlur_s{sigma}.png"), img_gauss, cmap='gray')

    # --- 3. 可视化对比 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    display_list = [
        (img, "Original"),
        (img_box, f"Box Blur ({box_size}x{box_size})"),
        (img_gauss, f"Gaussian Blur (σ={sigma})")
    ]
    
    for i, (data, title) in enumerate(display_list):
        axes[i].imshow(data, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{file_base}_Blur_Summary.png"))
    print(f"模糊实验完成！卷积核大小: {box_size}")
    plt.show()

if __name__ == "__main__":
    img_path = "../data/cameraman.png"
    if os.path.exists(img_path):
        # 你可以尝试增大 box_size (如 15, 21) 看看模糊程度的变化
        run_blur_experiment(img_path, box_size=15, sigma=5)