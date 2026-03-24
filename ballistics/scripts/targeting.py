import numpy as np
import matplotlib.pyplot as plt
import os

class BallisticTargeter:
    def __init__(self, m=5, h=0.1, eta=0.05):
        self.m = m
        self.h = h
        self.eta = eta
        self.g = np.array([0, -9.8])

    def get_dynamics_params(self, w_vec):
        decay = 1 - (self.h * self.eta / self.m)
        A = np.array([
            [1, 0, self.h, 0],
            [0, 1, 0, self.h],
            [0, 0, decay, 0],
            [0, 0, 0, decay]
        ])
        bv = self.h * self.g + (self.h * self.eta * np.array(w_vec) / self.m)
        b = np.array([0, 0, bv[0], bv[1]])
        return A, b

    def solve_v0(self, p_target, p0, w_vec, T):
        """核心逻辑：融入 T 的迭代计算"""
        A, b = self.get_dynamics_params(w_vec)
        
        F = np.eye(4)
        j = np.zeros(4)
        for _ in range(int(T)):
            j = A @ j + b
            F = A @ F
            
        # 提取子块 (Page 15)
        F11 = F[0:2, 0:2]
        F12 = F[0:2, 2:4]
        j1 = j[0:2]
        
        C = F12
        d = F11 @ p0 + j1
        
        try:
            # 求解 v0 = C^-1 * (p_target - d)
            v0 = np.linalg.inv(C) @ (p_target - d)
            return v0
        except np.linalg.LinAlgError:
            return None

    def run_simulation(self, v0, p0, w_vec, T):
        x = np.concatenate([p0, v0])
        A, b = self.get_dynamics_params(w_vec)
        path = [p0.copy()]
        for _ in range(int(T)):
            x = A @ x + b
            path.append(x[:2].copy())
        return np.array(path)

def main():
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(parent_dir, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    targeter = BallisticTargeter()
    p_origin = np.array([0, 0])
    p_goal = np.array([120, 20])   # 目标位置
    wind = np.array([-5, 0])       # 微风
    
    # --- 融入迭代数量 T 的对比 ---
    # 我们测试三组不同的总飞行时间步数 T
    T_list = [50, 100, 150] 
    colors = ['r', 'g', 'b']

    plt.figure(figsize=(10, 6))

    for T, col in zip(T_list, colors):
        # 1. 对当前的 T 求解 v0
        v0_solved = targeter.solve_v0(p_goal, p_origin, wind, T=T)
        
        if v0_solved is not None:
            # 2. 仿真验证
            traj = targeter.run_simulation(v0_solved, p_origin, wind, T=T)
            
            # 计算初速度大小
            v_mag = np.linalg.norm(v0_solved)
            
            # 3. 绘图：每个 T 对应一条轨迹
            plt.plot(traj[:, 0], traj[:, 1], color=col, label=f'T={T} (Speed={v_mag:.1f})')

    # 绘制目标点
    plt.scatter(p_goal[0], p_goal[1], color='black', marker='X', s=150, zorder=5, label='Target')
    
    plt.title(f"Targeting with Varying Iterations (T)\nTarget: {p_goal}, Wind: {wind}")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.axhline(0, color='black', lw=1)
    plt.grid(True, ls=':')
    plt.legend()

    save_path = os.path.join(save_dir, "exp4_varying_T_targeting.png")
    plt.savefig(save_path)
    print(f"实验完成！不同 T 的轨迹已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()