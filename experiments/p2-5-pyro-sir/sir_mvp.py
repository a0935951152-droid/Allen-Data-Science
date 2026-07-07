"""
sir_mvp.py — 單元 2-5 Demo MVP：從疫情曲線貝氏反推 SIR 參數（純 numpy）

命題：Pyro 之類的機率程式平台讓「機率模型 + 深度學習」接起來；核心工作是從觀測反推
      機制參數（Bingham 2019）。這支 MVP 用 Metropolis 從合成疫情曲線反推 β, γ 與 R₀。
做法：積分 SIR 得到感染曲線 I(t)，加觀測噪聲當「資料」；對 (β,γ) 做 Metropolis 採樣，
      看後驗均值是否還原真值與基本再生數 R₀=β/γ。
執行：python3 sir_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def sir(beta, gamma, T=60, dt=1.0, I0=0.01):
    S, I, R = 1 - I0, I0, 0.0
    out = []
    for _ in range(T):
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        S += dS * dt; I += dI * dt; R += gamma * I * dt
        out.append(I)
    return np.array(out)

def main():
    beta_t, gamma_t = 0.6, 0.2
    obs = sir(beta_t, gamma_t) + 0.01 * rng.normal(size=60)

    def logpost(th):
        b, g = th
        if not (0 < b < 2 and 0 < g < 1):
            return -np.inf
        return -0.5 * np.sum((sir(b, g) - obs) ** 2) / 0.01 ** 2

    th = np.array([0.3, 0.3])
    lp = logpost(th)
    chain = []
    for _ in range(20000):
        prop = th + 0.02 * rng.normal(size=2)
        lp2 = logpost(prop)
        if np.log(rng.uniform()) < lp2 - lp:
            th, lp = prop, lp2
        chain.append(th.copy())
    chain = np.array(chain[2000:])
    b, g = chain.mean(0)
    print("從合成疫情曲線貝氏反推 SIR")
    print(f"  真值   β={beta_t:.3f} γ={gamma_t:.3f}  R₀={beta_t/gamma_t:.2f}")
    print(f"  後驗   β={b:.3f} γ={g:.3f}  R₀={b/g:.2f}")
    print(f"  後驗標準差 β±{chain[:,0].std():.3f}  γ±{chain[:,1].std():.3f}")
    print("\n→ 從一條帶噪聲的感染曲線，還原出傳染/康復率與 R₀：機制參數的貝氏反演。")

if __name__ == "__main__":
    main()
