import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, transform
import os

def preprocess_to_gray(img_raw):
    """
    判断图片是否为 RGB，并将其转换为 [0, 1] 的灰度浮点矩阵
    """
    if img_raw.ndim == 3:
        # 使用讲义推荐的亮度权重：0.299R + 0.587G + 0.114B
        # 这体现了人眼对绿色最敏感，蓝色最不敏感的物理特性
        img_gray = 0.299 * img_raw[:,:,0] + 0.587 * img_raw[:,:,1] + 0.114 * img_raw[:,:,2]
    else:
        img_gray = img_raw
    
    # 归一化到 [0, 1]
    return img_gray.astype(float) / 255.0

def run_comprehensive_experiment(path_x, path_y=None, a=1.5, b=0.1, gamma=0.4,d=0.5,e=0.5):
    # 1. 提取基础文件名
    file_base = os.path.splitext(os.path.basename(path_x))[0]
    
    # 2. 读取并预处理
    img_x_raw = io.imread(path_x)
    x = preprocess_to_gray(img_x_raw)
    
    # --- 执行变换 ---
    # A. Negative (1 - x)
    x_neg = 1.0 - x
    
    # B. Contrast & Brightness (ax + c)
    avg_x = np.mean(x)
    c = (1 - a) * avg_x + b
    x_contrast = np.clip(a * x + c, 0, 1)
    
    # C. Gamma (x^gamma)
    x_gamma = np.power(x, gamma)

    # D. Composition (0.5x + 0.5y) - 图像相加
    x_plus_y = None
    if path_y and os.path.exists(path_y):
        if not np.isclose(d + e, 1.0):# 允许极小的浮点误差
            print(f"警告: d + e = {d+e} 不等于 1，已自动重置为默认值 0.5/0.5")
            d, e = 0.5, 0.5

        y_raw = io.imread(path_y)
        y = preprocess_to_gray(y_raw)
        # 线性代数加法要求维度一致，缩放 y 匹配 x
        y_resized = transform.resize(y, x.shape)
        x_plus_y = d * x + e * y_resized
        file_y_base = os.path.splitext(os.path.basename(path_y))[0]

    # --- 3. 动态构建存储路径与命名 ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 命名逻辑：Negative 不需要参数，Contrast 和 Gamma 随参数改变
    # 如果参数没变，文件名就相同，系统会自动覆盖；如果参数变了，文件名就不同，会新建文件。
    neg_name = f"{file_base}_Negative.png"
    contrast_name = f"{file_base}_Contrast_a{a}_b{b}.png"
    gamma_name = f"{file_base}_Gamma_g{gamma}.png"
    
    # 保存单项结果
    plt.imsave(os.path.join(results_dir, neg_name), x_neg, cmap='gray')
    plt.imsave(os.path.join(results_dir, contrast_name), x_contrast, cmap='gray')
    plt.imsave(os.path.join(results_dir, gamma_name), x_gamma, cmap='gray')
    
    if x_plus_y is not None:
        comp_name = f"Combined_{file_base}_{file_y_base}_d{d}_e{e}.png"
        plt.imsave(os.path.join(results_dir, comp_name), x_plus_y, cmap='gray')

    # --- 4. 可视化展示 ---
    num_plots = 5 if x_plus_y is not None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    display_data = [
        (x, "Original"),
        (x_neg, "Negative"),
        (x_contrast, f"Contrast(a={a},b={b})"),
        (x_gamma, f"Gamma(g={gamma})")
    ]
    if x_plus_y is not None:
        display_data.append((x_plus_y, f"Composition({d}x+{e}y)")) #注意f

    for i, (img, title) in enumerate(display_data):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # 保存汇总对比图
    summary_name = f"{file_base}_Summary_a{a}_b{b}_g{gamma}.png"
    plt.savefig(os.path.join(results_dir, summary_name))
    
    print(f"实验完成！结果已存入: {results_dir}")
    plt.show()

if __name__ == "__main__":
    # 替换为你实际的路径
    img_main = "../data/cameraman.png"
    img_second = "../data/grassland.jpg" 
    
    if os.path.exists(img_main):
        # 你可以尝试修改 a, b, gamma 的值
        #run_comprehensive_experiment(img_main, a=1.8, b=0.05, gamma=1.4)
        run_comprehensive_experiment(img_main, path_y=img_second, a=1.8, b=0.05, gamma=0.4,d=0.5,e=0.5)
    else:
        print(f"找不到主图片: {img_main}")