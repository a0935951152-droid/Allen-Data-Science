"""
koopman_mvp.py — 單元 4-2 Demo MVP：Koopman/EDMD——把非線性動力學線性化（純 numpy）

命題：找一組觀測座標，讓非線性動力學在該座標下變成線性（Lusch/Koopman）。
      這支 MVP 用 EDMD（擴展動態模態分解）：把狀態抬升到多項式觀測空間，在那裡用
      一個線性算子 K 逼近時間演化——「抽掉時間的表象，讓相空間裡演化變線性」。
做法：單擺 (q'=p, p'=-sin q)。字典 ψ=多項式(q,p)。解 ψ_{t+1} ≈ ψ_t·K。
      驗證：用線性 K 多步預測 q(t)，並從 K 的特徵值相位還原振盪頻率。
執行：python3 koopman_mvp.py   需求：numpy
"""
import numpy as np

def integrate(q, p, dt, n):
    Q = np.empty((n, 2))
    for i in range(n):
        Q[i] = [q, p]
        p = p - dt * np.sin(q); q = q + dt * p       # 辛積分產生乾淨資料
    return Q

def dictionary(Q):
    q, p = Q[:, 0], Q[:, 1]
    return np.stack([np.ones_like(q), q, p, q*q, q*p, p*p, q**3, q*q*p, q*p*p, p**3], 1)

def main():
    dt = 0.05
    Q = integrate(0.8, 0.0, dt, 1200)
    Psi = dictionary(Q)
    X, Y = Psi[:-1], Psi[1:]
    K, *_ = np.linalg.lstsq(X, Y, rcond=None)         # 線性 Koopman 算子近似

    # 多步預測：只用 K 在觀測空間線性演化
    psi = Psi[0].copy(); pred = []
    for _ in range(300):
        pred.append(psi[1])                            # index 1 = q
        psi = psi @ K
    pred = np.array(pred)
    err = np.abs(pred - Q[:300, 0]).max()

    # 多項式觀測會生出諧波(ω,2ω,3ω…)，基頻 = 振盪特徵值裡「最小正頻率」那個
    ev = np.linalg.eigvals(K)
    ang = np.abs(np.angle(ev))
    cand = ev[(ang > 0.02) & (np.abs(ev) > 0.9)]       # 排除近實數(≈0 頻)與強衰減模態
    freq = np.min(np.abs(np.angle(cand))) / dt

    print("Koopman/EDMD 把單擺線性化")
    print(f"  線性 K 300 步預測 q(t) 的最大誤差 = {err:.3f}")
    print(f"  由 K 特徵值還原的振盪頻率 ω = {freq:.3f}  (小幅擺理論 ω≈1.0)")
    print("\n→ 在多項式觀測空間裡，非線性擺的演化被一個線性算子精準捕捉：Koopman 線性化。")

if __name__ == "__main__":
    main()
