"""
ddpm_mvp.py — 單元 3-1 Demo MVP：DDPM 的「加噪發散 → 去噪收斂」（純 numpy）

命題：前向過程不斷加噪把資料發散成高斯；訓練一個網路逐步去噪，把它收斂回資料流形
      （Ho 2020）。這支 MVP 用雙峰目標，跑完整 DDPM 前向排程 + 反向 ancestral 取樣。
做法：目標 = 0.5·N(-3,0.3²)+0.5·N(+3,0.3²)。前向 x_t=√ᾱ·x₀+√(1-ᾱ)·ε。
      反向用該雙峰在時刻 t 的解析 score（去噪方向）做 DDPM 標準更新，從純噪聲生成樣本。
      驗證：生成樣本是否回到兩個模態、比例約 50/50。
執行：python3 ddpm_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)
A, S0 = 3.0, 0.3                      # 模態位置 ±A，各自寬度 S0

def score(x, abar):
    """雙峰在時刻 t 的 ∇log p_t(x)（去噪要用的『收斂方向』）。"""
    var = abar * S0 ** 2 + (1 - abar)
    mu = np.sqrt(abar) * A
    n_pos = np.exp(-0.5 * (x - mu) ** 2 / var)
    n_neg = np.exp(-0.5 * (x + mu) ** 2 / var)
    w = n_pos / (n_pos + n_neg)
    mean = w * mu + (1 - w) * (-mu)
    return (mean - x) / var

def main():
    T = 200
    beta = np.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    abar = np.cumprod(alpha)

    n = 4000
    x = rng.normal(size=n)            # 從純高斯噪聲起步
    for t in range(T - 1, -1, -1):
        eps_hat = -np.sqrt(1 - abar[t]) * score(x, abar[t])
        mean = (x - beta[t] / np.sqrt(1 - abar[t]) * eps_hat) / np.sqrt(alpha[t])
        x = mean + (np.sqrt(beta[t]) * rng.normal(size=n) if t > 0 else 0.0)

    frac_pos = np.mean(x > 0)
    print("DDPM 從純噪聲生成雙峰分佈")
    print(f"  正模態附近均值 = {x[x>0].mean():+.2f}  (目標 +{A})")
    print(f"  負模態附近均值 = {x[x<0].mean():+.2f}  (目標 -{A})")
    print(f"  正:負 比例      = {frac_pos:.2f} : {1-frac_pos:.2f}  (目標 0.50:0.50)")
    hist, edges = np.histogram(x, bins=30, range=(-5, 5))
    for h, e in zip(hist, edges):
        if h: print(f"  {e:+5.1f} |" + "#" * int(40 * h / hist.max()))
    print("→ 純噪聲經反向去噪，收斂回兩個資料模態：發散→收斂的生成回流。")

if __name__ == "__main__":
    main()
