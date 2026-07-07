"""
sindy_ae_mvp.py — 單元 4-4 Demo MVP：SINDy-Autoencoder——同時找座標與方程（純 numpy）

命題：同時學「座標」(autoencoder) 與「該座標下的稀疏動力學」(SINDy)，讓機器自己找到
      最簡潔的變數與方程（Champion 2019）。
做法（最小版）：真動力學是 2D 線性焦點 dz/dt = A z（旋轉+收縮）。把 z 用隨機正交映射
      抬升到 8 維觀測 y=W z（模擬高維量測，如反應擴散影像）。流程：
      (1) 用 PCA 從 8 維觀測還原 2 維座標（線性 autoencoder）；
      (2) 在還原座標上用最小二乘/SINDy 還原線性動力學矩陣。
      驗證：還原動力學的特徵值 ≈ 真值 −0.1±1i（特徵值在線性座標變換下不變，是乾淨的檢驗）。
執行：python3 sindy_ae_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def main():
    A = np.array([[-0.1, -1.0], [1.0, -0.1]])         # 真動力學：旋轉 + 緩慢收縮
    dt, n = 0.02, 3000
    z = np.empty((n, 2)); z[0] = [1.0, 0.0]
    for i in range(1, n):
        z[i] = z[i-1] + dt * (A @ z[i-1])

    W = np.linalg.qr(rng.normal(size=(8, 2)))[0]       # 8×2 正交抬升
    Y = z @ W.T                                        # 高維觀測 (n,8)

    # (1) 線性 autoencoder = PCA：從 8 維還原 2 維座標
    Yc = Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    zhat = Yc @ Vt[:2].T                               # 還原座標（真 z 的某個線性變換）

    # (2) 在還原座標上 SINDy（線性庫）：dzhat/dt ≈ zhat @ M
    dz = (zhat[2:] - zhat[:-2]) / (2 * dt)
    M, *_ = np.linalg.lstsq(zhat[1:-1], dz, rcond=None)

    ev_true = np.linalg.eigvals(A)
    ev_hat = np.linalg.eigvals(M.T)
    print("SINDy-AE（最小版）：從 8 維觀測還原 2 維座標 + 線性動力學")
    print(f"  真動力學特徵值   = {np.sort_complex(ev_true).round(3)}")
    print(f"  還原動力學特徵值 = {np.sort_complex(ev_hat).round(3)}")
    print("\n→ 即使只看到高維觀測，也能自動壓回低維座標並還原其動力學(特徵值吻合)：")
    print("  這是 Champion「同時學座標與方程」的最小骨架。")

if __name__ == "__main__":
    main()
