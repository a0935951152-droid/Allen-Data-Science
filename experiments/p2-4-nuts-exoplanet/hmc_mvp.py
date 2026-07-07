"""
hmc_mvp.py — 單元 2-4 Demo MVP：哈密頓蒙地卡羅(HMC) 高效採樣高維後驗（純 numpy）

命題：HMC 借哈密頓動力學，讓 MCMC 沿等能量面高效移動、遠距離提案仍高接受率
      （NUTS 是其自動調步版，Hoffman & Gelman 2014）。
做法：對一個強相關的 2D 高斯後驗做 HMC（leapfrog 積分 + Metropolis 接受），
      比對「估計的均值/共變異數」與真值，並印出接受率。
執行：python3 hmc_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

Sigma = np.array([[1.0, 0.9], [0.9, 1.0]])       # 強相關目標
Sinv = np.linalg.inv(Sigma)

def U(q):      return 0.5 * q @ Sinv @ q          # 位能 = -log p
def gradU(q):  return Sinv @ q

def leapfrog(q, p, eps, L):
    p = p - 0.5 * eps * gradU(q)                    # 半步動量
    for _ in range(L - 1):
        q = q + eps * p                            # 整步位置
        p = p - eps * gradU(q)                     # 整步動量
    q = q + eps * p
    p = p - 0.5 * eps * gradU(q)                    # 收尾半步動量（標準 leapfrog）
    return q, p

def main():
    q = np.zeros(2)
    eps, L, n = 0.25, 20, 8000
    samples = np.empty((n, 2))
    acc = 0
    for i in range(n):
        p0 = rng.normal(size=2)
        qn, pn = leapfrog(q, p0.copy(), eps, L)
        dH = (U(qn) + 0.5 * pn @ pn) - (U(q) + 0.5 * p0 @ p0)
        if np.log(rng.uniform()) < -dH:
            q = qn; acc += 1
        samples[i] = q

    burn = samples[1000:]
    print("HMC 採樣強相關 2D 高斯後驗")
    print(f"  接受率           = {acc / n:.2f}")
    print(f"  估計均值         = {burn.mean(0).round(3)}   (真值 [0 0])")
    print(f"  估計共變異數     =\n{np.cov(burn.T).round(2)}")
    print(f"  真實共變異數     =\n{Sigma}")
    print("\n→ 遠距 leapfrog 提案仍高接受率，正確重建相關結構：HMC 在高維後驗高效移動。")

if __name__ == "__main__":
    main()
