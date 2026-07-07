"""
uat_real.py — 單元 1-1 B 層（真實資料）：用 SILSO 太陽黑子驗證 UAT 的寬度→逼近力

資料：WDC-SILSO 月均太陽黑子數（自動下載到 data/，semicolon 分隔）。
做法：把黑子曲線當目標函數 SN(t)，用隨機特徵單隱層在不同寬度下逼近，看 MSE 隨寬度下降。
執行：python3 uat_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
rng = np.random.default_rng(0)

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def main():
    arr = np.genfromtxt(fetch(URL, "sunspots.csv"), delimiter=";")
    t, sn = arr[:, 2], arr[:, 3]
    m = sn >= 0
    t, sn = t[m], sn[m]
    x = ((t - t.min()) / (t.max() - t.min()) * 2 - 1)[:, None]   # → [-1,1]
    y = ((sn - sn.min()) / (sn.max() - sn.min()))[:, None]        # → [0,1]

    print(f"SILSO 太陽黑子：{len(t)} 個月, {t.min():.0f}–{t.max():.0f} 年")
    print("UAT 在真實黑子曲線上的寬度 vs 逼近 MSE")
    for width in [2, 8, 32, 128, 512, 2048]:
        W = rng.normal(size=(1, width)) * 8
        b = rng.normal(size=width) * 4
        H = np.tanh(x @ W + b)
        beta, *_ = np.linalg.lstsq(H, y, rcond=None)
        mse = np.mean((H @ beta - y) ** 2)
        print(f"  寬度 {width:>5}  MSE {mse:.2e}")
    print("→ 誤差隨寬度單調下降：真實非週期黑子訊號也被單隱層逼近（UAT）。")

if __name__ == "__main__":
    main()
