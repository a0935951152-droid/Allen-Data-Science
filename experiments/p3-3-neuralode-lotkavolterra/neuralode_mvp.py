"""
neuralode_mvp.py — 單元 3-3 Demo MVP：Neural ODE——把 RHS 參數化並微分穿過解算器（純 numpy）

命題：Neural ODE 把「一層」看成 ODE 的一步，直接交給 ODE solver，並對整條軌跡的誤差
      反傳來學 RHS（Chen 2018）。這支 MVP 用最小版：把 Lotka–Volterra 的 RHS 參數化，
      最小化「rollout 軌跡 vs 資料」的誤差來還原參數。
做法：真值 LV 生成軌跡當資料；以 4 個未知參數的 RHS 積分出預測軌跡，
      用 Adam + 有限差分梯度下降，看參數收斂回真值、軌跡吻合。
執行：python3 neuralode_mvp.py   需求：numpy
"""
import numpy as np

def lv_rhs(s, p):
    x, y = s
    a, b, d, g = p
    return np.array([a * x - b * x * y, d * x * y - g * y])

def rollout(p, s0, dt, n):
    s = s0.copy(); traj = np.empty((n, 2))
    for i in range(n):
        traj[i] = s
        k1 = lv_rhs(s, p); k2 = lv_rhs(s + 0.5*dt*k1, p)
        k3 = lv_rhs(s + 0.5*dt*k2, p); k4 = lv_rhs(s + dt*k3, p)
        s = s + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return traj

def main():
    true_p = np.array([1.1, 0.4, 0.1, 0.4])
    s0, dt, n = np.array([10.0, 5.0]), 0.05, 200
    data = rollout(true_p, s0, dt, n)

    def loss(p):
        return np.mean((rollout(p, s0, dt, n) - data) ** 2)

    p = np.array([0.8, 0.6, 0.2, 0.6])          # 初始猜測（偏離真值）
    m = np.zeros(4); v = np.zeros(4); lr = 0.02
    for it in range(1, 4001):
        g = np.empty(4); h = 1e-4; l0 = loss(p)
        for k in range(4):
            pp = p.copy(); pp[k] += h
            g[k] = (loss(pp) - l0) / h
        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*g**2
        p = p - lr * (m/(1-0.9**it)) / (np.sqrt(v/(1-0.999**it)) + 1e-8)
        p = np.clip(p, 1e-3, None)

    print("Neural ODE（最小版）：從軌跡還原 Lotka–Volterra 的 RHS 參數")
    names = ["α", "β", "δ", "γ"]
    for nm, t, e in zip(names, true_p, p):
        print(f"  {nm}: 真值 {t:.3f}   還原 {e:.3f}")
    print(f"  最終軌跡 MSE = {loss(p):.2e}")
    print("\n→ 對『積分整條軌跡』的誤差做梯度下降，還原出連續動力學的 RHS：Neural ODE 的精神。")

if __name__ == "__main__":
    main()
