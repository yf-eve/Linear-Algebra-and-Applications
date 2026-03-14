import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import convolve
import os

def run_deblurring_experiment(image_path, lambda_reg=0.01):
    # 1. 加载并转为灰度（为了专注理解线性代数逻辑）
    img_raw = io.imread(image_path)
    if img_raw.ndim == 3:
        img = color.rgb2gray(img_raw[:,:,:3])
    else:
        img = img_raw
    img = img.astype(float) / 255.0
    
    # 2. 模拟损坏过程 (y = Ax + v)
    # A: 高斯模糊算子
    psf = np.outer(np.exp(-0.5*np.linspace(-1,1,5)**2), np.exp(-0.5*np.linspace(-1,1,5)**2))
    psf /= psf.sum()
    img_blurred = convolve(img, psf, mode='reflect')
    
    # v: 加入微小的加性高斯噪声
    noise_sigma = 0.001
    img_corrupted = img_blurred + np.random.normal(0, noise_sigma, img.shape)

    # 3. 去模糊处理 (Least-squares de-blurring)
    # 目标：minimize ||Ax - y||^2 + λL(x)
    # 我们使用迭代法（梯度下降）来求解这个最小值
    x_recovered = img_corrupted.copy()
    learning_rate = 0.5
    iterations = 200
    
    for i in range(iterations):
        # 计算第一项的梯度: 2 * A.T * (Ax - y)
        # 这里 A 是卷积，A.T 也是卷积（对于对称核 psf 来说，两者相同）
        ax_minus_y = convolve(x_recovered, psf, mode='reflect') - img_corrupted
        grad_data = convolve(ax_minus_y, psf, mode='reflect')
        
        # 计算第二项 λL(x) 的梯度: 对应拉普拉斯算子
        # 拉普拉斯算子核：[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        grad_reg = convolve(x_recovered, laplacian_kernel, mode='reflect')
        
        # 更新 x: x = x - step * (grad_data + λ * grad_reg)
        x_recovered = x_recovered - learning_rate * (grad_data + lambda_reg * grad_reg)
        
        # 确保像素值在 [0, 1] 范围内
        x_recovered = np.clip(x_recovered, 0, 1)

    # --- 4. 可视化 ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # 将 lambda 格式化为字符串，例如 lambda=0.050
    lambda_str = f"lambda_{lambda_reg:.3f}"
    
    # 修改后的保存文件名
    save_filename = f"{file_base}_Deblur_{lambda_str}_noise_{noise_sigma}.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Original", "Corrupted (Blur + Noise)", f"Recovered (λ={lambda_reg})"]
    datas = [img, img_corrupted, x_recovered]
    
    for i in range(3):
        axes[i].imshow(datas[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, save_filename)) # 使用含 lambda 的文件名
    plt.show()

if __name__ == "__main__":
    img_path = "../data/woman.jpg" 
    if os.path.exists(img_path):
        run_deblurring_experiment(img_path, lambda_reg=0.05)