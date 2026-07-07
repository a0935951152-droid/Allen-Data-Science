"""
dpgmm_mvp.py — 單元 5-2 Demo MVP：Dirichlet Process 混合——資料自己決定群數（純 numpy）

命題：讓混合成分數趨於無窮（DP），資料自己決定要用幾群，不用事先設定 K
      （Rasmussen 2000）。這支 MVP 用 CRP + 高斯共軛的塌縮 Gibbs，對未標註資料自動定群數。
做法：合成 3 群 2D 資料（不告訴模型有幾群）。已知似然方差 var、群均值先驗 N(0,pvar)，
      邊際掉群均值後，點 i 進既有群 c 的機率 ∝ n_c·N(x_i; 後驗預測)，開新群 ∝ α·N(x_i; 先驗預測)。
      跑幾輪 Gibbs 看使用中的群數收斂到真值附近。
執行：python3 dpgmm_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(1)

def lognorm(x, mean, var):
    return -0.5 * np.sum((x - mean) ** 2) / var - np.log(2 * np.pi * var)

def main():
    centers = np.array([[0, 0], [6, 6], [6, -6]])
    X = np.vstack([c + rng.normal(0, 0.7, (60, 2)) for c in centers])
    n = len(X)
    alpha, var, pvar = 0.3, 0.7 ** 2, 25.0

    z = np.zeros(n, int)
    for sweep in range(40):
        for i in range(n):
            z[i] = -1                                  # 先移除 i
            labels = [c for c in np.unique(z) if c >= 0]
            logp, cand = [], []
            for c in labels:
                m = X[z == c]
                nc = len(m)
                post_prec = 1.0 / pvar + nc / var       # 邊際掉群均值的後驗
                post_mean = (m.sum(0) / var) / post_prec
                pred_var = var + 1.0 / post_prec
                logp.append(np.log(nc) + lognorm(X[i], post_mean, pred_var))
                cand.append(c)
            logp.append(np.log(alpha) + lognorm(X[i], np.zeros(2), var + pvar))
            cand.append(max(labels) + 1 if labels else 0)
            logp = np.array(logp) - max(logp)
            p = np.exp(logp); p /= p.sum()
            z[i] = cand[rng.choice(len(cand), p=p)]
        _, z = np.unique(z, return_inverse=True)

    k = len(np.unique(z))
    print("DP 混合：不指定群數，讓資料自己決定")
    print(f"  真實群數 = 3")
    print(f"  推論群數 = {k}")
    for c in np.unique(z):
        print(f"    群 {c}: {np.sum(z==c):>3} 點, 中心 {X[z==c].mean(0).round(1)}")
    print("\n→ CRP 先驗 + 共軛似然自動平衡『開新群 vs 併入舊群』，收斂到資料真正的群數。")

if __name__ == "__main__":
    main()
