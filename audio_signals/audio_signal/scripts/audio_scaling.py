import numpy as np
from scipy.io import wavfile
import os

# 1. 读取音频数据
# sample_rate 是采样率 (1/h)，data 是音频向量 x
current_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(current_dir, "..", "data", "钢琴背景音.wav")
sample_rate, x = wavfile.read(audio_path)

# 音频是一个向量 x，每个元素 x_i 是一个采样点 [cite: 13, 17]
x = x.astype(np.float32)/ 32768.0  # 归一化到 [-1, 1] 范围

# 2. 进行缩放实验
# a > 1 听起来应该明显变大
a1 = 4.0
y_louder = a1 * x

# a < 1 听起来应该明显变小
a2 = 0.1
y_quieter = a2 * x

# a = -1: 听起来应该和原声一模一样
a3 = -1.0
y_inverse = a3 * x

def calculate_spl(x_norm):
    # 计算均方根 RMS 
    rms_p = np.sqrt(np.mean(x_norm**2))
    # 定义参考声压 
    p_ref = 2e-5 
    # 计算分贝 SPL 
    spl = 20 * np.log10(rms_p / p_ref)
    return rms_p, spl
   
rms_p0, spl0 = calculate_spl(x)
rms_p1, spl1 = calculate_spl(y_louder)
rms_p2, spl2 = calculate_spl(y_quieter)
rms_p3, spl3 = calculate_spl(y_inverse)


# 3. 保存并导出结果
# 注意：写回文件前需要转回整数格式 (int16)
def save_wav(name, data, rate):
    # 限制范围防止爆音
    data = np.clip(data* 32768.0, -32768, 32767)
    wavfile.write(name, rate, data.astype(np.int16))

save_wav("output_louder.wav", y_louder, sample_rate)
save_wav("output_quieter.wav", y_quieter, sample_rate)
save_wav("output_inverse.wav", y_inverse, sample_rate)

print("实验文件已生成：")
print("1. output_louder.wav 4")
print("2. output_quieter.wav (0.1倍音量)")
print("3. output_inverse.wav (反相音频)")

print(f"原声平均有效声压 (rms(p)): {rms_p0:.6f} N/m^2") # 
print(f"原声计算出的响度 (SPL): {spl0:.2f} dB") #

print(f"变大音频平均有效声压 (rms(p)): {rms_p1:.6f} N/m^2") # 
print(f"变大音频计算出的响度 (SPL): {spl1:.2f} dB") #

print(f"变小音频平均有效声压 (rms(p)): {rms_p2:.6f} N/m^2") # 
print(f"变小音频计算出的响度 (SPL): {spl2:.2f} dB") #

print(f"反相音频平均有效声压 (rms(p)): {rms_p3:.6f} N/m^2") # 

print(f"反相音频计算出的响度 (SPL): {spl3:.2f} dB") #
