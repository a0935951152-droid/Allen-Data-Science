"""
uat_mvp.py — 單元 1-1 Demo MVP：通用逼近定理（UAT）的寬度—誤差關係（純 numpy）

命題：單隱層網路隨寬度增加，可把任意連續函數逼到任意小（Cybenko/Hornik 1989）。
做法：用「隨機特徵 / 極限學習機」隔離出 UAT 的核心——固定隨機隱層(tanh)，只用最小二乘
      解輸出權重，掃描隱藏單元數 m，看逼近誤差如何隨 m 單調下降。
      （這樣不必訓練，就能乾淨地展示「寬度→表達力」，避開優化的干擾。）
執行：python3 uat_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def target(x):                      # 多頻非線性目標函數（模擬太陽黑子那類准週期訊號）
    return np.sin(3 * np.pi * x) + 0.5 * np.sin(7 * np.pi * x)

def fit_width(m, x, y):
    W = rng.normal(size=(1, m)) * 4.0
    b = rng.normal(size=m) * 2.0
    H = np.tanh(x @ W + b)           # (N, m) 隨機特徵
    beta, *_ = np.linalg.lstsq(H, y, rcond=None)
    return np.mean((H @ beta - y) ** 2)

def main():
    x = np.linspace(-1, 1, 400)[:, None]
    y = target(x)
    print("=" * 52)
    print("UAT：單隱層寬度 vs 逼近誤差（隨機特徵 + 最小二乘）")
    print("=" * 52)
    print(f"{'寬度 m':>8} | {'MSE':>12} | 相對誤差條")
    e0 = None
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        mse = fit_width(m, x, y)
        e0 = mse if e0 is None else e0
        bar = "#" * int(40 * min(1.0, mse / e0))
        print(f"{m:>8} | {mse:>12.2e} | {bar}")
    print("\n→ 誤差隨寬度單調下降並趨近 0：寬度越大，表達力越強（UAT 的經驗版）。")

if __name__ == "__main__":
    main()
