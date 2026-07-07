"""
latentsde_mvp.py — 單元 4-3 Demo MVP：Neural SDE 的核心——同時學 drift 與 diffusion（純 numpy）

命題：把 Neural ODE 推廣到隨機，同時學漂移(drift)與擴散(diffusion)（Li 2020）。
      這支 MVP 隔離出「從一條隨機軌跡同時辨識 drift 與 diffusion」——latent SDE 的推論骨架。
做法：模擬 Ornstein–Uhlenbeck 過程 dx = -θx dt + σ dW（動物在家域中的隨機遊走原型）。
      用增量回歸還原：Δx ≈ -θ·x·Δt + σ·√Δt·ξ。斜率給 drift θ，殘差方差給 diffusion σ。
執行：python3 latentsde_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def main():
    theta_t, sigma_t, dt, N = 1.0, 0.5, 0.01, 200000
    x = np.empty(N); x[0] = 2.0
    for i in range(1, N):                              # 模擬 OU 路徑
        x[i] = x[i-1] - theta_t * x[i-1] * dt + sigma_t * np.sqrt(dt) * rng.normal()

    dx = np.diff(x)
    xm = x[:-1]
    b = (xm @ dx) / (xm @ xm)                          # Δx = b·x + noise，b = -θ·dt
    theta_hat = -b / dt
    resid = dx - b * xm
    sigma_hat = np.sqrt(resid.var() / dt)             # 殘差方差 = σ²·dt

    print("Latent SDE 核心：從一條 OU 隨機軌跡同時還原 drift 與 diffusion")
    print(f"  drift     θ: 真值 {theta_t:.3f}   還原 {theta_hat:.3f}")
    print(f"  diffusion σ: 真值 {sigma_t:.3f}   還原 {sigma_hat:.3f}")
    print("\n→ 漂移(決定性回拉)與擴散(隨機強度)被分開辨識：這是神經 SDE 學的兩樣東西。")

if __name__ == "__main__":
    main()
