import numpy as np
from scipy.io import wavfile
import os

def load_and_standardize(path):
    sample_rate, data = wavfile.read(path)
    # 转换为浮点数并归一化到 -1 到 1 之间，方便处理
    data = data.astype(np.float32) / 32768.0
    # 如果是双声道，取其中一个声道简化处理
    if len(data.shape) > 1:
        data = data[:, 0]
    return sample_rate, data

# 1. 读取两个音轨
current_dir = os.path.dirname(os.path.abspath(__file__))
rate1, x1 = wavfile.read(os.path.join(current_dir, "..", "data", "钢琴背景音.wav"))
rate2, x2 = wavfile.read(os.path.join(current_dir, "..", "data", "人声.wav"))

if rate1 != rate2:
    raise ValueError(f"警告：采样率不匹配！钢琴是{rate1}Hz，人声是{rate2}Hz。请先重采样。")

# 2. 确保长度一致 (取较短的那一个，对应讲义中 same length 的要求)
n = min(len(x1), len(x2))
x1, x2 = x1[:n], x2[:n]

inner_prod = np.dot(x1, x2)
similarity = inner_prod / (np.linalg.norm(x1) * np.linalg.norm(x2))
print(f"两种声音的数学相关性 (Similarity): {similarity:.4f}")

# 3. 混合实验 (对应讲义第 7 页的 Mix 1, 2, 3)

# 方案 A: 均匀混合 (a1=0.5, a2=0.5)
y_balanced = 0.5 * x1 + 0.5 * x2

# 方案 B: 突出音轨2 (例如人声为主，背景音很小)
y_vocal_focus = 0.05 * x1 + 0.95 * x2

# 方案 C: 讲义中的负权重尝试 (虽然 |a| 决定音量，但可以看看相位抵消)
y_diff = 0.5 * x1 - 0.5 * x2

def safe_normalize(data, threshold=1.0):
    """
    判断合成函数的幅值，如果超过阈值则进行整体线性缩放。
    """
    # 找到当前信号的峰值（绝对值的最大值）
    peak = np.max(np.abs(data))
    
    # 如果峰值超过了阈值，进行缩放
    if peak > threshold:
        print(f"检测到幅值溢出 (Peak={peak:.4f})，正在进行线性缩放...")
        # 核心逻辑：整个向量除以峰值，将其压回到 [-1, 1] 空间
        # 这保持了 x1 和 x2 之间的相对比例（线性组合关系不变）
        data = data / peak
    else:
        print(f"信号幅值安全 (Peak={peak:.4f})。")
        
    return data

y_balanced = safe_normalize(y_balanced)
y_vocal_focus = safe_normalize(y_vocal_focus)
y_diff = safe_normalize(y_diff)

# 4. 导出结果
def save_mix(name, data, rate):
    # 重新映射回 int16 范围
    data = np.clip(data * 32768.0, -32768, 32767)
    wavfile.write(name, rate, data.astype(np.int16))
#
save_mix("mix_balanced.wav", y_balanced, rate1)
# save_mix("mix_vocal_focus.wav", y_vocal_focus, rate1)
#save_mix("mix_diff.wav", y_diff, rate1)
#save_mix("mix_diff_double_piano.wav", y_diff, rate1)

print("实验二文件已生成：")
print("1. mix_balanced.wav (均衡混音)")
print("2. mix_vocal_focus.wav (突出第二音轨)")

print("3. mix_diff.wav (减法混音)")

