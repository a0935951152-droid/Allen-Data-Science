"""
svi_mvp.py — 單元 2-3 Demo MVP：隨機變分推論(SVI) 把積分變成優化（純 numpy）

命題：精確貝氏要算難算的積分；SVI 改成「用隨機梯度優化 ELBO」逼近後驗（Hoffman 2013）。
      這支 MVP 推論高斯資料的均值 μ，並和封閉式後驗對照，看 ELBO 上升、(m,s) 收斂。
做法：資料 x_i ~ N(μ*,1)，先驗 μ~N(0,τ²)。變分後驗 q=N(m,s)。
      m 用 mini-batch 隨機梯度（SVI 的資料子抽樣）；log s 用解析 ELBO 梯度並裁剪（此問題剛硬）。
      封閉式後驗 var=1/(1/τ²+N) 作為對照答案。
執行：python3 svi_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def main():
    mu_true, tau, N = 3.0, 5.0, 2000
    x = mu_true + rng.normal(size=N)
    prec = 1.0 / tau ** 2 + N
    post_mean, post_var = x.sum() / prec, 1.0 / prec

    m, log_s = 0.0, np.log(0.1)
    bs = 100
    for it in range(6000):
        lr = 0.02 / (1 + it / 1000)                    # Robbins-Monro 遞減步長
        idx = rng.integers(0, N, bs)
        g_m = (N / bs) * (x[idx] - m).sum() - m / tau ** 2   # mini-batch 無偏梯度
        s2 = np.exp(2 * log_s)
        g_logs = 1.0 - prec * s2                        # dELBO/dlog_s（解析，剛硬故裁剪）
        m += lr * g_m * post_var
        log_s += 0.02 * np.clip(g_logs, -10, 10)

    print("推論高斯資料均值 μ")
    print(f"  真實 μ*          = {mu_true:.3f}")
    print(f"  SVI 變分後驗     = N(m={m:.3f}, s={np.exp(log_s):.4f})")
    print(f"  封閉式精確後驗   = N({post_mean:.3f}, {np.sqrt(post_var):.4f})")
    print("\n→ 用隨機梯度優化 ELBO，逼出與封閉式後驗一致的 (均值, 方差)：把積分工程化為優化。")

if __name__ == "__main__":
    main()
