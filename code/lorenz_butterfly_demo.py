"""
lorenz_butterfly_demo.py  —  單元 4-5 的模擬示範（純 numpy，零第三方依賴）

目的：親眼看到「發散」的原型——混沌系統對初始條件的敏感依賴（蝴蝶效應）。
      這是整份文獻回顧裡「支柱四：發散」的源頭（Lorenz 1963）。
      與 lorenz_sindy_demo.py（收斂/還原方程）互補：一個展示發散，一個展示還原。

做什麼：
  1. 用 RK4 積分兩條 Lorenz 軌跡，初值只差 1e-9。
  2. 追蹤兩者的距離 δ(t)，畫（文字版）它如何指數放大。
  3. 從 log δ(t) 的斜率估計「最大 Lyapunov 指數」λ。
     Lorenz 標準參數下理論值 λ ≈ 0.906（bit/time ≈ 1.3），這裡估的是自然對數版。
  4. 由 λ 反推「可預測時界」：初始 1e-9 的誤差放大到 O(1) 需要多久。

執行：  python3 code/lorenz_butterfly_demo.py
需求：  只需 numpy。若主機沒有 numpy，請在專案 .venv 內 `pip install numpy`（勿裝進主機）。
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
    """同時積分兩條僅差 delta0 的軌跡，回傳每步的距離。"""
    a = np.array(x0, dtype=float)
    b = a + np.array([delta0, 0.0, 0.0])
    dist = np.empty(steps)
    for i in range(steps):
        dist[i] = np.linalg.norm(a - b)
        a = rk4_step(a, dt)
        b = rk4_step(b, dt)
    return dist


def estimate_lyapunov(t, dist, fit_lo, fit_hi):
    """在距離仍呈指數成長（未飽和）的區間，對 log(dist) 做線性擬合取斜率。"""
    mask = (t >= fit_lo) & (t <= fit_hi)
    slope, intercept = np.polyfit(t[mask], np.log(dist[mask]), 1)
    return slope, intercept


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
    dt, steps = 0.01, 4000
    delta0 = 1e-9
    t = np.arange(steps) * dt
    dist = integrate_pair([-8.0, 8.0, 27.0], delta0, dt, steps)

    # 只在指數成長段（避開初始瞬態與後期飽和 ~O(吸引子尺度)）擬合
    lam, b = estimate_lyapunov(t, dist, fit_lo=5.0, fit_hi=25.0)

    print("=" * 62)
    print("Lorenz 蝴蝶效應：兩條初值只差 1e-9 的軌跡如何分離")
    print("=" * 62)
    ascii_curve(t, dist)

    print(f"\n初始間距 δ0        = {delta0:.1e}")
    print(f"估計最大 Lyapunov λ = {lam:.3f}  (理論 ≈ 0.906)")
    horizon = np.log(1.0 / delta0) / lam
    print(f"可預測時界 (δ→O(1)) ≈ {horizon:.1f}  時間單位")
    print("\n→ 誤差以 e^(λt) 放大：再精確的初值，混沌也會在有限時間內把它抹平。")
    print("  這就是『發散』——世界模型無法靠記憶硬扛，只能靠守恆量與方差去約束。")


if __name__ == "__main__":
    main()
