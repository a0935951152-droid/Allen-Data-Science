"""
pinn_mvp.py — 單元 3-4 Demo MVP：PINN——把 PDE 殘差寫進損失（物理牆）（純 numpy）

命題：PINN 把 PDE 殘差直接寫進損失，逼網路的解自動滿足物理律（Raissi 2019）。
      這支 MVP 解 1D Poisson u''(x) = -f(x)，u(0)=u(1)=0，用滿足邊界的正弦基底，
      最小化「PDE 殘差」求係數——正弦基底下殘差最小化是線性系統，能對到解析解。
做法：解 u'' = -f，取 f 使解析解 u*(x)=sin(πx)+0.5 sin(3πx)。基底 φ_k=sin(kπx) 天然滿足 BC，
      u=Σ a_k φ_k，殘差 = u''+f = Σ a_k(-(kπ)²)φ_k + f，於配點上最小化 → 解出 a_k。
執行：python3 pinn_mvp.py   需求：numpy
"""
import numpy as np

def main():
    x = np.linspace(0, 1, 200)
    u_true = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    f = (np.pi**2) * np.sin(np.pi*x) + 0.5*(3*np.pi)**2 * np.sin(3*np.pi*x)   # = -u''

    K = 8
    ks = np.arange(1, K + 1)
    Phi = np.sin(np.outer(x, ks) * np.pi)              # (N,K) 基底，滿足 u(0)=u(1)=0
    Phi_xx = -(ks * np.pi) ** 2 * Phi                  # 二階導
    # PDE 殘差 r = u'' + f = Phi_xx @ a + f，最小化 ||r||² → 線性最小二乘 Phi_xx a = -f
    a, *_ = np.linalg.lstsq(Phi_xx, -f, rcond=None)
    u = Phi @ a

    print("PINN（線性版）解 1D Poisson u'' = -f，BC u(0)=u(1)=0")
    print(f"  還原係數 a[1..3] = {a[:3].round(3)}  (真值 [1.0, 0.0, 0.5])")
    print(f"  與解析解最大誤差 = {np.abs(u - u_true).max():.2e}")
    print(f"  PDE 殘差 RMS     = {np.sqrt(np.mean((Phi_xx @ a + f)**2)):.2e}")
    print("\n→ 不給任何 u 的標籤，只最小化 PDE 殘差(物理牆)，就逼出滿足方程的解。")

if __name__ == "__main__":
    main()
