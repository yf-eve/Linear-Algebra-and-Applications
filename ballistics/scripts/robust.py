import numpy as np
import matplotlib.pyplot as plt
import os

class RobustTargeter:
    def __init__(self, m=5, h=0.1, eta=0.05):
        self.m = m
        self.h = h
        self.eta = eta
        self.g = np.array([0, -9.8])

    def get_params_for_wind(self, w_vec, T, p0):
        """计算特定风速下的映射矩阵 C 和 偏移向量 d"""
        decay = 1 - (self.h * self.eta / self.m)
        A = np.array([
            [1, 0, self.h, 0],
            [0, 1, 0, self.h],
            [0, 0, decay, 0],
            [0, 0, 0, decay]
        ])
        bv = self.h * self.g + (self.h * self.eta * np.array(w_vec) / self.m)
        b = np.array([0, 0, bv[0], bv[1]])
        
        F, j = np.eye(4), np.zeros(4)
        for _ in range(int(T)):
            j = A @ j + b
            F = A @ F
        
        C = F[0:2, 2:4]
        d = F[0:2, 0:2] @ p0 + j[0:2]
        return C, d

    def solve_robust_v0(self, p_target, p0, wind_scenarios, T):
        """
        稳健求解：最小化所有场景下的总误差
        构建超定方程组: A_big * v0 = B_big
        """
        A_list, B_list = [], []
        for w in wind_scenarios:
            C, d = self.get_params_for_wind(w, T, p0)
            A_list.append(C)
            B_list.append(p_target - d)
        
        A_big = np.vstack(A_list)
        B_big = np.concatenate(B_list) # 注意这里用 concatenate 拉成一列
        
        # 最小二乘法求解
        v0_robust, _, _, _ = np.linalg.lstsq(A_big, B_big, rcond=None)
        return v0_robust

def main():
    # --- 1. 自动化路径处理 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(parent_dir, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 2. 实验参数 ---
    targeter = RobustTargeter()
    p_origin = np.array([0, 0])
    p_goal = np.array([120, 20])
    T_steps = 100
    
    # 定义风力不确定性范围：从强逆风到弱顺风
    wind_scenarios = [np.array([w, 0]) for w in np.linspace(-15, 5, 12)]
    
    # --- 3. 求解稳健 v0 ---
    v0_robust = targeter.solve_robust_v0(p_goal, p_origin, wind_scenarios, T_steps)

    # --- 4. 绘图与验证 ---
    plt.figure(figsize=(12, 7))
    
    for i, w in enumerate(wind_scenarios):
        # 使用相同的 v0 在不同风力下跑仿真
        A, b = targeter.get_params_for_wind(w, 1, p_origin) # 这里只取一步的 A,b 用于仿真循环
        
        x = np.concatenate([p_origin, v0_robust])
        path = [p_origin.copy()]
        # 重新运行物理步进以获得完整轨迹
        decay = 1 - (targeter.h * targeter.eta / targeter.m)
        A_step = np.array([[1,0,0.1,0],[0,1,0,0.1],[0,0,decay,0],[0,0,0,decay]])
        bv = 0.1*targeter.g + (0.1*targeter.eta*w/targeter.m)
        b_step = np.array([0,0,bv[0],bv[1]])
        
        for _ in range(T_steps):
            x = A_step @ x + b_step
            path.append(x[:2].copy())
        
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], alpha=0.4, label=f"Wind: {w[0]:.1f}")

    plt.scatter(p_goal[0], p_goal[1], color='red', marker='X', s=250, zorder=10, label='TARGET')
    plt.title(f"Robust Ballistics Experiment\nOptimized v0 for Wind Range [-15, 5]")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid(True, ls=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # --- 5. 保存图片 ---
    save_path = os.path.join(save_dir, "exp5_robust_targeting.png")
    plt.savefig(save_path)
    print(f"稳健性实验完成！\n图片已保存至: {save_path}")
    print(f"计算出的稳健初速度向量为: {v0_robust}")
    plt.show()

if __name__ == "__main__":
    main()