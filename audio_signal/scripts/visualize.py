import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# 1. 尝试使用 Linux 下最通用的开源中文字体名称
linux_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']

# 2. 自动寻找可用的字体
plt.rcParams['font.sans-serif'] = linux_fonts + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 必须加上，否则负号显示为方块

# 1. 定义观察时间窗口 (0.1s)
fs = 44100
t_window = 0.1  # 0.1s
num_samples = int(fs * t_window)
t = np.linspace(0, t_window, num_samples, endpoint=False)

# 2. 读取并裁剪信号
current_dir = os.path.dirname(os.path.abspath(__file__))
rate1, x_full = wavfile.read(os.path.join(current_dir, "..", "data", "钢琴背景音.wav"))

# 如果是双声道，取左声道
if len(x_full.shape) > 1:
    x_full = x_full[:, 0]

# 【关键点】只取前 num_samples 个点，确保与 t 长度一致
x = x_full[:num_samples].astype(np.float32)

# 3. 实验数据处理
y_louder = 4.0 * x  # 缩放系数 a=4 [cite: 41]
y_inverse = -1.0 * x  # 相位反转 -x [cite: 44]

# 4. 实验三音色合成 (纯正弦波用于对比) [cite: 88, 89]
y_pure = np.sin(2 * np.pi * 440 * t)
c_rich = [0.7, 0.6, 0.3, 0.04]
y_rich = sum(ck * np.sin(2 * np.pi * 440 * (k+1) * t) for k, ck in enumerate(c_rich))

# 5. 绘图
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 子图1: 缩放对比
axs[0, 0].plot(t*1000, x, label='原始 x', color='blue')
axs[0, 0].plot(t*1000, y_louder, '--', label='放大 4x', color='red')
axs[0, 0].set_title("图1：信号缩放对比 (Scaling)")
axs[0, 0].set_ylabel("幅值")
axs[0, 0].legend()

# 子图2: 相位对比
axs[0, 1].plot(t*1000, x, label='原始 x', color='blue')
axs[0, 1].plot(t*1000, y_inverse, ':', label='反相 -x', color='green')
axs[0, 1].set_title("图2：相位反转对比 (Phase Inverse)")
axs[0, 1].legend()

# 子图3: 纯音波形
axs[1, 0].plot(t*1000, y_pure, color='purple')
axs[1, 0].set_title("图3：纯正弦波形 (c=1)")
axs[1, 0].set_ylabel("幅值")

# 子图4: 复合音色波形
axs[1, 1].plot(t*1000, y_rich, color='orange')
axs[1, 1].set_title("图4：复合谐波音色波形 (Rich Timbre)")

for ax in axs.flat:
    ax.set_xlabel("时间 (ms)")

plt.tight_layout()

plt.show()

