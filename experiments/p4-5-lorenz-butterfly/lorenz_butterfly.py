"""
lorenz_butterfly.py  —  單元 4-5 的模擬示範（純 numpy，零第三方依賴）

目的：親眼看到「發散」的原型——混沌系統對初始條件的敏感依賴（蝴蝶效應）。
      這是整份文獻回顧裡「支柱四：發散」的源頭（Lorenz 1963）。
      與 p4-1 的 sindy_lorenz.py（收斂/還原方程）互補：一個展示發散，一個展示還原。

做什麼：
  1. 用 RK4 積分兩條 Lorenz 軌跡，初值只差 1e-9，畫（文字版）距離如何指數放大再飽和。
  2. 用 Benettin 重正規化法估計「最大 Lyapunov 指數」λ。
     — 為何不直接對單一軌跡對的 log 距離做線性擬合？因為沿吸引子的局部拉伸率會漲落，
       單一軌跡對的斜率對擬合窗很敏感（換個窗就從 0.83 跳到 0.89），那是在噪聲裡挑數字。
     — Benettin 法每步把擾動拉回一個固定小量 d0（保持在線性區），累加 log(成長比)再對時間平均，
       等於對「局部拉伸率」沿軌跡取長時間平均 → 穩定收斂到真值。
     Lorenz 標準參數 (σ=10, ρ=28, β=8/3) 的理論值 λ ≈ 0.9056（自然對數版；bit/time ≈ 1.3）。
  3. 由 λ 反推「可預測時界」：初始 1e-9 的誤差放大到 O(1) 需要多久。

執行：  python3 experiments/p4-5-lorenz-butterfly/lorenz_butterfly.py
需求：  只需 numpy。若主機沒有 numpy，請在本資料夾 .venv 內 `pip install -r requirements.txt`（勿裝進主機）。
"""

import numpy as np


def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])


def rk4_step(s, dt):
    k1 = lorenz_rhs(s)
    k2 = lorenz_rhs(s + 0.5 * dt * k1)
    k3 = lorenz_rhs(s + 0.5 * dt * k2)
    k4 = lorenz_rhs(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_pair(x0, delta0, dt, steps):
    """同時積分兩條僅差 delta0 的軌跡，回傳每步的距離（供視覺化：看指數放大→飽和）。"""
    a = np.array(x0, dtype=float)
    b = a + np.array([delta0, 0.0, 0.0])
    dist = np.empty(steps)
    for i in range(steps):
        dist[i] = np.linalg.norm(a - b)
        a = rk4_step(a, dt)
        b = rk4_step(b, dt)
    return dist


def benettin_lyapunov(x0, dt, steps, d0=1e-8, transient=2000):
    """Benettin 重正規化法估最大 Lyapunov 指數：
    每步演化參考軌跡與一個相距 d0 的擾動軌跡，量測新距離 d1，累加 log(d1/d0)，
    再把擾動沿當前分離方向拉回距離 d0（保持在線性區）。λ = Σ log(d1/d0) / (steps·dt)。"""
    ref = np.array(x0, dtype=float)
    for _ in range(transient):          # 先讓軌跡落到吸引子上
        ref = rk4_step(ref, dt)
    pert = ref + np.array([d0, 0.0, 0.0])
    log_sum = 0.0
    for _ in range(steps):
        ref = rk4_step(ref, dt)
        pert = rk4_step(pert, dt)
        diff = pert - ref
        d1 = np.linalg.norm(diff)
        log_sum += np.log(d1 / d0)
        pert = ref + (diff / d1) * d0   # 重正規化：方向保留，長度拉回 d0
    return log_sum / (steps * dt)


def ascii_curve(t, dist, rows=18, cols=60):
    """把 log10(dist) vs t 畫成文字圖，讓終端也能看到指數爬升與飽和。"""
    y = np.log10(dist)
    idx = np.linspace(0, len(t) - 1, cols).astype(int)
    ys, ts = y[idx], t[idx]
    ymin, ymax = ys.min(), ys.max()
    grid = [[" "] * cols for _ in range(rows)]
    for c, val in enumerate(ys):
        r = int((val - ymin) / (ymax - ymin + 1e-12) * (rows - 1))
        grid[rows - 1 - r][c] = "*"
    print(f"\n  log10(距離)  上界 {ymax:+.1f}")
    for row in grid:
        print("  |" + "".join(row))
    print("  +" + "-" * cols)
    print(f"   下界 {ymin:+.1f}   t = 0 … {ts[-1]:.1f}")


def main():
    dt = 0.01
    delta0 = 1e-9

    # (1) 視覺化：單一軌跡對的距離如何指數放大再飽和
    vis_steps = 4000
    t = np.arange(vis_steps) * dt
    dist = integrate_pair([-8.0, 8.0, 27.0], delta0, dt, vis_steps)

    # (2) 準確估計：Benettin 重正規化（長時間平均，不受擬合窗影響）
    lam = benettin_lyapunov([-8.0, 8.0, 27.0], dt, steps=200000)

    print("=" * 62)
    print("Lorenz 蝴蝶效應：兩條初值只差 1e-9 的軌跡如何分離")
    print("=" * 62)
    ascii_curve(t, dist)

    print(f"\n初始間距 δ0             = {delta0:.1e}")
    print(f"Benettin 最大 Lyapunov λ = {lam:.3f}  (理論 ≈ 0.906)")
    horizon = np.log(1.0 / delta0) / lam
    print(f"可預測時界 (δ→O(1))     ≈ {horizon:.1f}  時間單位")
    print("\n→ 誤差以 e^(λt) 放大：再精確的初值，混沌也會在有限時間內把它抹平。")
    print("  這就是『發散』——世界模型無法靠記憶硬扛，只能靠守恆量與方差去約束。")


if __name__ == "__main__":
    main()
