import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

def run_inpainting_pro(image_path, mask_size=(80, 80), iterations=1000):
    # 1. 加载图片
    img_raw = io.imread(image_path)
    
    # 统一剥离 RGBA 的 A 通道
    if img_raw.ndim == 3 and img_raw.shape[-1] == 4:
        img_raw = img_raw[:, :, :3]
    
    # 标准化到 [0, 1]
    img = img_raw.astype(float) / 255.0 if img_raw.max() > 1 else img_raw.astype(float)
    h, w = img.shape[:2]
    is_color = (img.ndim == 3)

    # 2. 制造损坏 (Damage)
    # 计算中心位置
    start_h, start_w = h//2 - mask_size[0]//2, w//2 - mask_size[1]//2
    end_h, end_w = start_h + mask_size[0], start_w + mask_size[1]
    
    img_damaged = img.copy()
    img_damaged[start_h:end_h, start_w:end_w] = 0 # 挖坑
    
    # 3. 修复过程
    img_recovered = img_damaged.copy()
    
    # 如果是彩色，我们需要对 3 个通道分别进行迭代
    # 如果是灰色，channels 为 1
    channels = 3 if is_color else 1
    
    print(f"正在修复 {'彩色' if is_color else '灰色'} 图片: {os.path.basename(image_path)}")
    
    for i in range(iterations):
        prev_recovered = img_recovered.copy()
        
        if is_color:
            for c in range(channels):
                # 核心逻辑：损坏区域像素 = 邻居平均值
                # 这里使用切片加速，代替嵌套循环（工程优化）
                # 这种方法计算每个像素的上下左右均值
                for r in range(start_h, end_h):
                    for col in range(start_w, end_w):
                        neighbors = [
                            prev_recovered[r-1, col, c], prev_recovered[r+1, col, c],
                            prev_recovered[r, col-1, c], prev_recovered[r, col+1, c]
                        ]
                        img_recovered[r, col, c] = np.mean(neighbors)
        else:
            # 灰度图逻辑
            for r in range(start_h, end_h):
                for col in range(start_w, end_w):
                    neighbors = [
                        prev_recovered[r-1, col], prev_recovered[r+1, col],
                        prev_recovered[r, col-1], prev_recovered[r, col+1]
                    ]
                    img_recovered[r, col] = np.mean(neighbors)
        
        if (i+1) % 200 == 0:
            print(f"进度: {i+1}/{iterations}...")

        # 在循环里加入这个判断
        if i % 1000 == 0:
            plt.imshow(img_recovered, cmap='gray')
            plt.title(f"Iteration {i}")
            plt.show()

    # --- 4. 展示结果 ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    file_base = os.path.splitext(os.path.basename(image_path))[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    titles = ["Original", "Damaged", "In-painted"]
    datas = [img, img_damaged, img_recovered]
    
    for i in range(3):
        # imshow 能够自动识别 (H,W,3) 为彩色，(H,W) 为灰色
        axes[i].imshow(datas[i], cmap='gray' if not is_color else None)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{file_base}_Inpainting_Final.png"))
    print(f"修复完成！图片已存入 results 文件夹。")
    plt.show()

if __name__ == "__main__":
  
    img_path = "../data/woman.jpg" 
    if os.path.exists(img_path):
        # 彩色图迭代次数建议多一点，或者 mask 小一点，因为信息扩散比较慢
        run_inpainting_pro(img_path, mask_size=(50, 50), iterations=10000)