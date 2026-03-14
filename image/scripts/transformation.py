import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import os

def run_geometry_experiment(image_path, shift_x=50, shift_y=30, scale=0.5, angle=45):
    # 1. 读取
    img = io.imread(image_path, as_gray=True)
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # --- A. Flipping (保持不变，因为镜像没有连续参数) ---
    img_flip_h = img[:, ::-1]
    img_flip_v = img[::-1, :]
    
    # --- B. Shifting (变量: shift_x, shift_y) ---
    # translation=(列位移, 行位移)
    tform_shift = transform.SimilarityTransform(translation=(shift_x, shift_y))
    img_shifted = transform.warp(img, tform_shift.inverse)
    
    # --- C. Zooming (变量: scale) ---
    img_zoomed = transform.rescale(img, scale, anti_aliasing=True)
    
    # --- D. Rotation (变量: angle) ---
    img_rotated = transform.rotate(img, angle, resize=False)

    # --- 2. 动态构建存储路径与参数化命名 ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 将参数融入文件名后缀
    shift_suffix = f"SX{shift_x}_SY{shift_y}"
    zoom_suffix = f"Z{scale}"
    rot_suffix = f"R{angle}"
    
    save_list = [
        (img_flip_h, f"{file_base}_FlipH.png"),
        (img_flip_v, f"{file_base}_FlipV.png"),
        (img_shifted, f"{file_base}_Shift_{shift_suffix}.png"),
        (img_zoomed, f"{file_base}_Zoom_{zoom_suffix}.png"),
        (img_rotated, f"{file_base}_Rot_{rot_suffix}.png")
    ]
    
    for data, filename in save_list:
        plt.imsave(os.path.join(results_dir, filename), data, cmap='gray')

    # --- 3. 可视化展示 ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    display_list = [
        (img, "Original"),
        (img_flip_h, "Horizontal Flip"),
        (img_flip_v, "Vertical Flip"),
        (img_shifted, f"Shift ({shift_x}, {shift_y})"),
        (img_zoomed, f"Zoom ({scale}x)"),
        (img_rotated, f"Rotation ({angle}°)")
    ]
    
    for i, (data, title) in enumerate(display_list):
        axes[i].imshow(data, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # 汇总图命名也包含所有关键变量
    summary_name = f"{file_base}_TRANS_S{shift_x}_{shift_y}_Z{scale}_R{angle}.png"
    plt.savefig(os.path.join(results_dir, summary_name))
    print(f"实验完成！结果文件已带参数存入: {results_dir}")
    plt.show()

if __name__ == "__main__":
    img_path = "../data/cameraman.png"
    if os.path.exists(img_path):
        # 你可以在这里随意修改参数进行多次实验
        run_geometry_experiment(img_path, shift_x=80, shift_y=40, scale=0.7, angle=30)