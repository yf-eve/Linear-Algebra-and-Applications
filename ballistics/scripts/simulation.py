import numpy as np
import matplotlib.pyplot as plt
import os  # 引入 os 模块处理路径

# --- 物理仿真函数保持不变 ---
def simulate_ballistics(v0_mag, theta_deg, w_vec, m=5, T=100, h=0.1, eta=0.05):
    theta_rad = np.radians(theta_deg)
    v0 = np.array([v0_mag * np.cos(theta_rad), v0_mag * np.sin(theta_rad)])
    p0 = np.array([0, 0])
    x = np.concatenate([p0, v0]) 
    
    decay = 1 - (h * eta / m)
    A = np.array([
        [1, 0, h, 0],
        [0, 1, 0, h],
        [0, 0, decay, 0],
        [0, 0, 0, decay]
    ]) 
    
    g_vec = np.array([0, -9.8])
    w_vec = np.array(w_vec)
    bv = h * g_vec + (h * eta * w_vec / m) 
    b = np.array([0, 0, bv[0], bv[1]])
    
    positions = [p0]
    for _ in range(T):
        x = A @ x + b
        positions.append(x[:2])
    return np.array(positions)

# --- 路径处理逻辑 ---
# 1. 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 定位到上一级目录
parent_dir = os.path.dirname(current_dir)
# 3. 定义 results 文件夹路径
save_dir = os.path.join(parent_dir, 'results')

# 4. 检查文件夹是否存在，不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"创建文件夹: {save_dir}")

# --- 实验全局参数 ---
common_params = {'m': 5, 'T': 100, 'h': 0.1, 'eta': 0.05}

# ---------------------------------------------------------
# Exp 1: Wind Comparison 
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
t1 = simulate_ballistics(50, 45, [0, 0], **common_params) 
t2 = simulate_ballistics(50, 45, [-10, 0], **common_params) 
plt.plot(t1[:, 0], t1[:, 1], 'b-', label='No Wind w=(0,0)')
plt.plot(t2[:, 0], t2[:, 1], 'g-', label='With Wind w=(-10,0)')
plt.title("Exp 1: Wind Comparison (v0=50, theta=45°)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axhline(0, color='black', lw=1)
plt.grid(True, ls=':')
plt.legend()
plt.savefig(os.path.join(save_dir, "exp1_wind_comparison.png"))
plt.show() # 显示当前图表并清空画布

# ---------------------------------------------------------
# Exp 2: Varying Elevation [cite: 88]
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
angles = [30, 45, 80] 
for ang in angles:
    traj = simulate_ballistics(50, ang, [-10, 0], **common_params)
    plt.plot(traj[:, 0], traj[:, 1], label=f'theta={ang}°')
plt.title("Exp 2: Varying Elevation (v0=50, w=[-10,0])")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axhline(0, color='black', lw=1)
plt.grid(True, ls=':')
plt.legend()
plt.savefig(os.path.join(save_dir, "exp2_varying_elevation.png"))
plt.show()

# ---------------------------------------------------------
# Exp 3: Varying Speed [cite: 106]
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
speeds = [50, 75, 100] 
for spd in speeds:
    traj = simulate_ballistics(spd, 50, [-10, 0], **common_params)
    plt.plot(traj[:, 0], traj[:, 1], label=f'v0={spd}')
plt.title("Exp 3: Varying Speed (theta=50°, w=[-10,0])")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axhline(0, color='black', lw=1)
plt.grid(True, ls=':')
plt.legend()
plt.savefig(os.path.join(save_dir, "exp3_varying_speed.png"))
plt.show()