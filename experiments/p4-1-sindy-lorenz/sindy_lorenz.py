"""
sindy_lorenz.py  —  單元 4-1 的模擬示範 / 專案 P1 的旗艦（純 numpy，零第三方依賴外掛）

目的：親眼看到「機器從一段混沌時間序列裡，挖出產生它的加減乘除微分方程」。
      這驗證了整場推導的核心命題之一：真正的世界模型是邏輯/方程，NN 只是逼近它。

方法：SINDy（Sparse Identification of Nonlinear Dynamics, Brunton+ 2016）的最小實作。
  1. 用 RK4 積分 Lorenz 系統得到軌跡 X(t)（這是「真值/資料」）。
  2. 用有限差分估計導數 dX/dt。
  3. 建一個候選函數庫 Theta(X) = [1, x, y, z, x^2, xy, ... ] （多項式）。
  4. 解 dX/dt ≈ Theta(X) @ Xi，用「序列閾值最小二乘」逼出稀疏係數 Xi。
  5. 印出被還原的方程，跟真實 Lorenz 方程比對。

真實 Lorenz (sigma=10, rho=28, beta=8/3):
    dx/dt = 10(y - x)          = -10 x + 10 y
    dy/dt = x(28 - z) - y      = 28 x - y - x z
    dz/dt = x y - (8/3) z      = x y - 2.667 z

執行：  python3 experiments/p4-1-sindy-lorenz/sindy_lorenz.py
需求：  只需 numpy。若主機沒有 numpy，請在專案 .venv 內 `pip install numpy`（勿裝進主機）。
"""

import numpy as np


# ---------- 1. 產生 Lorenz 軌跡（RK4，手刻，不依賴 scipy） ----------
def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])


def integrate(x0, dt, steps):
    traj = np.empty((steps, 3))
    s = np.array(x0, dtype=float)
    for i in range(steps):
        traj[i] = s
        k1 = lorenz_rhs(s)
        k2 = lorenz_rhs(s + 0.5 * dt * k1)
        k3 = lorenz_rhs(s + 0.5 * dt * k2)
        k4 = lorenz_rhs(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return traj


# ---------- 2. 候選函數庫：二階多項式 ----------
def poly_library(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    feats = [np.ones_like(x), x, y, z,
             x * x, x * y, x * z, y * y, y * z, z * z]
    names = ["1", "x", "y", "z",
             "x^2", "x y", "x z", "y^2", "y z", "z^2"]
    return np.stack(feats, axis=1), names


# ---------- 3. 序列閾值最小二乘（SINDy 的核心優化） ----------
def stlsq(Theta, dXdt, threshold=0.1, n_iter=10):
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]      # 初始最小二乘
    for _ in range(n_iter):
        small = np.abs(Xi) < threshold                    # 把小係數砍成 0（稀疏化）
        Xi[small] = 0
        for col in range(dXdt.shape[1]):                  # 對每個保留項重新回歸
            big = ~small[:, col]
            if big.any():
                Xi[big, col] = np.linalg.lstsq(
                    Theta[:, big], dXdt[:, col], rcond=None)[0]
    return Xi


def print_equation(Xi, names, target):
    terms = []
    for i, c in enumerate(Xi):
        if abs(c) > 1e-6:
            terms.append(f"{c:+.3f} {names[i]}")
    print(f"  d{target}/dt = " + "  ".join(terms) if terms else f"  d{target}/dt = 0")


# ---------- 主流程 ----------
def main():
    dt, steps = 0.002, 5000
    traj = integrate([-8.0, 8.0, 27.0], dt, steps)

    # 中央差分估計導數，去掉邊界
    dXdt = (traj[2:] - traj[:-2]) / (2 * dt)
    X = traj[1:-1]

    Theta, names = poly_library(X)
    Xi = stlsq(Theta, dXdt, threshold=0.1)

    print("=" * 60)
    print("SINDy 從 Lorenz 混沌時間序列還原出的動力學方程：")
    print("=" * 60)
    for j, tgt in enumerate("xyz"):
        print_equation(Xi[:, j], names, tgt)

    print("\n對照真實 Lorenz 方程：")
    print("  dx/dt = -10.000 x  +10.000 y")
    print("  dy/dt = +28.000 x   -1.000 y   -1.000 x z")
    print("  dz/dt =  -2.667 z   +1.000 x y")
    print("\n→ 若兩者吻合，你就親眼見到『機器從資料裡挖出加減乘除的律』。")


if __name__ == "__main__":
    main()
