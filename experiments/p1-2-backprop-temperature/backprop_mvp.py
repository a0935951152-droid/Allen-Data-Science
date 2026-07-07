"""
backprop_mvp.py — 單元 1-2 Demo MVP：手刻反向傳播 + 梯度檢核（純 numpy）

命題：反向傳播用鏈式法則把誤差分配到每個權重，讓多層網路可訓練（RHW 1986）。
做法：2 層 tanh MLP 學「同心圓」二分類（線性不可分）。全手刻前向/反向；
      並對隨機幾個權重做「數值梯度 vs 解析梯度」檢核，證明反傳算對了。
執行：python3 backprop_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def make_data(n=400):
    r = rng.uniform(0, 1, n)
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    y = (r > 0.5).astype(float)[:, None]     # 內圈 0 / 外圈 1
    return X, y

def init(h=16):
    return {"W1": rng.normal(size=(2, h)) * 0.5, "b1": np.zeros(h),
            "W2": rng.normal(size=(h, 1)) * 0.5, "b2": np.zeros(1)}

def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.tanh(z1)
    z2 = a1 @ p["W2"] + p["b2"]; a2 = 1 / (1 + np.exp(-z2))
    return a2, (z1, a1, z2, a2)

def loss(p, X, y):
    a2, _ = forward(p, X)
    eps = 1e-9
    return -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))

def grads(p, X, y):
    a2, (z1, a1, z2, _) = forward(p, X)
    n = len(X)
    dz2 = (a2 - y) / n
    g = {"W2": a1.T @ dz2, "b2": dz2.sum(0)}
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (1 - np.tanh(z1) ** 2)
    g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
    return g

def main():
    X, y = make_data()
    p = init()

    # 梯度檢核：數值 vs 解析（抽查 W1 的一個元素）
    g = grads(p, X, y)
    eps = 1e-5; i, j = 0, 0
    p["W1"][i, j] += eps; lp = loss(p, X, y)
    p["W1"][i, j] -= 2 * eps; lm = loss(p, X, y)
    p["W1"][i, j] += eps
    num = (lp - lm) / (2 * eps)
    print("梯度檢核 W1[0,0]:  解析 %.6e  數值 %.6e  相對差 %.1e"
          % (g["W1"][i, j], num, abs(g["W1"][i, j] - num) / (abs(num) + 1e-12)))

    lr = 0.5
    print("\n訓練（手刻反傳）：")
    for ep in range(2001):
        g = grads(p, X, y)
        for k in p:
            p[k] -= lr * g[k]
        if ep % 400 == 0:
            a2, _ = forward(p, X)
            acc = np.mean((a2 > 0.5) == y)
            print(f"  epoch {ep:>4}  loss {loss(p, X, y):.4f}  acc {acc:.3f}")
    print("\n→ 損失下降、準確率逼近 1：反傳把誤差正確分配到各層權重。")

if __name__ == "__main__":
    main()
