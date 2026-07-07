"""
vae_mvp.py — 單元 2-1 Demo MVP：VAE 的核心——重參數化技巧（純 numpy）

命題：VAE 用「重參數化」z=μ+σ·ε 讓『採樣』變得可微，才能把貝氏後驗推論塞進梯度下降
      （Kingma & Welling 2013）。這支 MVP 隔離出這個關鍵，量化證明它為何有用。
做法：估計 ∇_θ E_{z~N(θ,σ)}[z²]（真值 = 2θ）。比較兩種梯度估計子的變異數：
      (1) 重參數化：z=θ+σε → ∇=2z；   (2) score-function/REINFORCE：((z-θ)/σ²)·z²。
      兩者都無偏，但重參數化的變異數低得多 → 這正是 VAE 能穩定訓練的原因。
執行：python3 vae_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def main():
    theta, sigma, N = 1.5, 1.0, 20000
    eps = rng.normal(size=N)
    z = theta + sigma * eps

    reparam = 2 * z                                   # ∇_θ (θ+σε)²
    score = ((z - theta) / sigma ** 2) * z ** 2       # REINFORCE

    true = 2 * theta
    print("估計 ∇_θ E[z²],  真值 = 2θ = %.3f\n" % true)
    for name, est in [("重參數化 (reparam)", reparam), ("score-function", score)]:
        print(f"  {name:>18}: 均值 {est.mean():+.3f}   變異數 {est.var():.3f}")
    print(f"\n變異數比 (score / reparam) = {score.var() / reparam.var():.1f} 倍")
    print("→ 兩者都無偏，但重參數化變異數低得多；梯度乾淨，才撐得起 VAE 的深度後驗推論。")

if __name__ == "__main__":
    main()
