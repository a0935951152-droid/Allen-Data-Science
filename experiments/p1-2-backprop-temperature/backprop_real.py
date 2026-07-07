"""
backprop_real.py — 單元 1-2 B 層（真實資料）：手刻反傳學真實氣溫的季節循環

資料：Melbourne 每日最低氣溫（1981–1990，自動下載到 data/）。
做法：以「一年中的第幾天」的正弦/餘弦特徵為輸入，手刻 2 層 MLP + 反傳迴歸日溫，
      在held-out 上報 R²，看它是否學到季節循環。
執行：python3 backprop_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
rng = np.random.default_rng(0)

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def main():
    temps, doys = [], []
    with open(fetch(URL, "melb_temps.csv")) as f:
        next(f)
        for line in f:
            d, t = line.strip().replace('"', "").split(",")
            mo, da = int(d[5:7]), int(d[8:10])
            doys.append((mo - 1) * 30.4 + da)
            temps.append(float(t))
    doy = np.array(doys); y = np.array(temps)[:, None]
    ph = 2 * np.pi * doy / 365.25
    X = np.stack([np.sin(ph), np.cos(ph), np.sin(2*ph), np.cos(2*ph)], 1)
    ym, ys = y.mean(), y.std(); yn = (y - ym) / ys

    n = len(X); idx = rng.permutation(n); tr, te = idx[:n*4//5], idx[n*4//5:]
    h = 12
    W1 = rng.normal(size=(4, h))*0.5; b1 = np.zeros(h)
    W2 = rng.normal(size=(h, 1))*0.5; b2 = np.zeros(1)
    lr = 0.02
    for ep in range(3000):
        a1 = np.tanh(X[tr] @ W1 + b1); o = a1 @ W2 + b2
        d2 = (o - yn[tr]) / len(tr)
        dW2 = a1.T @ d2; db2 = d2.sum(0)
        d1 = (d2 @ W2.T) * (1 - a1**2)
        dW1 = X[tr].T @ d1; db1 = d1.sum(0)
        W2 -= lr*dW2; b2 -= lr*db2; W1 -= lr*dW1; b1 -= lr*db1

    def pred(Xte):
        return (np.tanh(Xte @ W1 + b1) @ W2 + b2) * ys + ym
    p = pred(X[te])
    r2 = 1 - np.sum((p - y[te])**2) / np.sum((y[te] - y[te].mean())**2)
    print(f"Melbourne 日最低氣溫：{n} 天")
    print(f"  手刻 2 層 MLP + 反傳，held-out R² = {r2:.3f}")
    print(f"  夏季(1月)預測≈{pred(np.array([[np.sin(2*np.pi*15/365),np.cos(2*np.pi*15/365),np.sin(4*np.pi*15/365),np.cos(4*np.pi*15/365)]]))[0,0]:.1f}°C, "
          f"冬季(7月)≈{pred(np.array([[np.sin(2*np.pi*195/365),np.cos(2*np.pi*195/365),np.sin(4*np.pi*195/365),np.cos(4*np.pi*195/365)]]))[0,0]:.1f}°C")
    print("→ 反傳從真實氣溫學到季節循環（南半球 1 月熱、7 月冷）。")

if __name__ == "__main__":
    main()
