import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

def run_channel_and_weight_experiment(image_path, custom_w=(0.1, 0.1, 0.8)):
    """
    实现通道剥离与多种权重合成实验
    custom_w: 对应 (R, G, B) 的自定义权重
    """
    # 1. 加载图片并确保是 RGB
    img_raw = io.imread(image_path)
    if img_raw.ndim != 3:
        print("该实验需要彩色 RGB 图片！")
        return
    
    # 归一化到 [0, 1] 方便线性代数计算
    img = img_raw.astype(float) / 255.0
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # 2. 通道剥离 (Matrices)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # 3. 不同权重的合成 (Linear Projection: y = w.T * x)
    # A. 算术平均权重
    w_avg = (1/3, 1/3, 1/3)
    gray_avg = w_avg[0]*R + w_avg[1]*G + w_avg[2]*B
    
    # B. 生理感知权重 (Standard)
    w_std = (0.299, 0.587, 0.114)
    gray_std = w_std[0]*R + w_std[1]*G + w_std[2]*B
    
    # C. 自定义权重 (Special Effects)
    w_cus = custom_w
    gray_cus = np.clip(w_cus[0]*R + w_cus[1]*G + w_cus[2]*B, 0, 1)

    # --- 4. 存储与命名 ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 命名规则：文件名 + 权重数值
    names = [
        f"{file_base}_R_Channel.png",
        f"{file_base}_G_Channel.png",
        f"{file_base}_B_Channel.png",
        f"{file_base}_Gray_Avg_0.33.png",
        f"{file_base}_Gray_Std_0.299_0.587.png",
        f"{file_base}_Gray_Cus_{w_cus[0]}_{w_cus[1]}_{w_cus[2]}.png"
    ]
    imgs = [R, G, B, gray_avg, gray_std, gray_cus]
    
    for name, data in zip(names, imgs):
        plt.imsave(os.path.join(results_dir, name), data, cmap='gray')

    # --- 5. 可视化对比 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    titles = [
        "Red Channel", "Green Channel", "Blue Channel",
        "Average Weight (1/3)", "Standard (Physiological)", f"Custom Weight {w_cus}"
    ]
    
    for i in range(6):
        axes[i].imshow(imgs[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        
    plt.tight_layout()
    summary_name = f"{file_base}_Weights_Compare.png"
    plt.savefig(os.path.join(results_dir, summary_name))
    print(f"实验完成！结果已存入: {results_dir}")
    plt.show()

if __name__ == "__main__":
    # 替换为你自己的图片路径
    img_path = "../data/Mandrill.png" 
    
    if os.path.exists(img_path):
        
        run_channel_and_weight_experiment(img_path, custom_w=(5, 5, 5))