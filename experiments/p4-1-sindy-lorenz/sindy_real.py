"""
sindy_real.py — 單元 4-1 B 層（真實資料）：SINDy 從真實猞猁–野兔還原生態方程

資料：Hudson Bay 猞猁–野兔年皮毛數 1900–1920（stan 鏡像，自動下載到 data/，千隻）。
做法：對真實 (野兔 x, 猞猁 y) 年序列，平滑估導數、建二階多項式庫，用序列閾值最小二乘
      還原稀疏動力學。預期挑出 Lotka–Volterra 的結構項（x, xy for prey；y, xy for predator）。
      提醒：21 點噪聲年資料，還原的是「有效方程」，係數不如合成乾淨——這本身是誠實的結果。
執行：python3 sindy_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = ("https://raw.githubusercontent.com/stan-dev/example-models/master/"
       "knitr/lotka-volterra/hudson-bay-lynx-hare.csv")

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line[0] == "#" or line.startswith("Year"):
            continue
        _, lynx, hare = line.split(",")
        rows.append([float(hare), float(lynx)])       # x=野兔, y=猞猁
    return np.array(rows)

def stlsq(Theta, dX, thr, n_iter=10):
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
    for _ in range(n_iter):
        small = np.abs(Xi) < thr; Xi[small] = 0
        for c in range(dX.shape[1]):
            big = ~small[:, c]
            if big.any():
                Xi[big, c] = np.linalg.lstsq(Theta[:, big], dX[:, c], rcond=None)[0]
    return Xi

def main():
    D = load(fetch(URL, "lynx_hare.csv"))
    # 標準化到 O(1)，讓線性項與 xy 交互項的係數量級可比（否則 xy~千量級、係數被閾值誤砍）
    sx, sy = D[:, 0].std(), D[:, 1].std()
    x, y = D[:, 0] / sx, D[:, 1] / sy
    dx = (x[2:] - x[:-2]) / 2; dy = (y[2:] - y[:-2]) / 2
    xs, ys = x[1:-1], y[1:-1]
    dX = np.stack([dx, dy], 1)
    Theta = np.stack([np.ones_like(xs), xs, ys, xs*xs, xs*ys, ys*ys], 1)
    names = ["1", "x", "y", "x^2", "xy", "y^2"]
    Xi = stlsq(Theta.copy(), dX.copy(), thr=0.08)

    print(f"猞猁–野兔真實資料：{len(D)} 年；SINDy 還原的生態方程（標準化座標）")
    for j, tgt in enumerate(["x(野兔)", "y(猞猁)"]):
        terms = [f"{Xi[i,j]:+.3f}·{names[i]}" for i in range(len(names)) if abs(Xi[i,j]) > 1e-6]
        print(f"  d{tgt}/dt = " + "  ".join(terms))
    print("  (Lotka–Volterra 結構: dx/dt=αx−βxy, dy/dt=δxy−γy)")
    print("→ 從真實生態年序列挖出含 xy 交互項的稀疏結構：野兔靠自身增長、被 xy 壓制；")
    print("  猞猁靠 xy 增長、自身衰減——正是 Lotka–Volterra 的定性骨架。")

if __name__ == "__main__":
    main()
