import numpy as np
from scipy.io import wavfile

# 基本设置
fs = 44100  # 采样率 (1/h) [cite: 19, 20]
duration = 2.0  # 持续2秒
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

def generate_tone(f, harmonics, time_array):
    """
    根据讲义公式生成信号: p(t) = sum(ck * sin(2 * pi * f * k * t))
    """
    signal = np.zeros_like(time_array)
    for k, ck in enumerate(harmonics):
        # f_k = f * (k + 1) 是第 k+1 个谐波
        signal += ck * np.sin(2 * np.pi * f * (k + 1) * time_array)
    return signal

# --- 任务 A: 音色对比 (讲义第12-13页) ---
f_A = 440.0  # 中音 A

# 1. 纯正弦波 (枯燥的音色) c = (1, 0, ...)
c_pure = [1.0]
tone_pure = generate_tone(f_A, c_pure, t)

# 2. 丰富音色 c = (0.7, 0.6, 0.3, 0.04) [cite: 92]
c_rich = [0.7, 0.6, 0.3, 0.04]
tone_rich = generate_tone(f_A, c_rich, t)

# --- 任务 B: 音程计算 (讲义第9-10页) ---
# 计算中音 C (比 A 高 3 个半音)
f_C = (2**(3/12)) * 440.0  # 约 523.2 Hz [cite: 81]
tone_C = generate_tone(f_C, c_rich, t)

# 保存文件
def export_wav(filename, signal):
    # 归一化处理
    signal = signal / np.max(np.abs(signal)) * 0.5
    wavfile.write(filename, fs, (signal * 32767).astype(np.int16))

export_wav("3_tone_pure.wav", tone_pure)
export_wav("3_tone_rich.wav", tone_rich)
export_wav("3_tone_C_major.wav", tone_C)

print("实验三文件已生成：")
print("- 3_tone_pure.wav: 纯正弦波 (A440)")
print("- 3_tone_rich.wav: 丰富音色 (A440)")
print("- 3_tone_C_major.wav: 使用丰富音色演奏的中音 C")