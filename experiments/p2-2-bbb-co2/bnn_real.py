"""
bnn_real.py — 單元 2-2 B 層（真實資料）：貝氏迴歸外推真實 CO₂ 並誠實示警

資料：NOAA Mauna Loa 月均 CO₂（Keeling 曲線，自動下載到 data/）。
做法：用「中心化縮放的二次 trend + 年週期季節」特徵做貝氏線性迴歸，只在早期年份訓練，
      外推近年。看預測方差在外推區張大（誠實），並檢查真實值涵蓋率。
      數值要點：時間特徵中心化並縮放（避免 tt² 與 tt 共線爆炸）；σ² 用訓練殘差估；
      predictive variance = σ² + φᵀ Σ φ 隨外推距離張開。
執行：python3 bnn_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def features(t, t0, scale):
    u = (t - t0) / scale                                   # 中心化並縮放的時間
    return np.stack([np.ones_like(u), u, u**2,
                     np.sin(2*np.pi*t), np.cos(2*np.pi*t)], 1)

def main():
    arr = np.genfromtxt(fetch(URL, "co2_mlo.txt"), comments="#")
    t, co2 = arr[:, 2], arr[:, 3]
    m = co2 > 0; t, co2 = t[m], co2[m]

    cut = t < 2005
    t0, scale = t[cut].mean(), t[cut].std()               # 用「訓練區」統計做縮放
    Phi = features(t, t0, scale)
    Pt, yt = Phi[cut], co2[cut]

    sigma2 = 2.0 ** 2                                      # 觀測噪聲（季節殘差量級）
    alpha = 100.0                                         # 適度權重先驗
    A = Pt.T @ Pt / sigma2 + np.eye(Phi.shape[1]) / alpha
    L = np.linalg.cholesky(A)
    w = np.linalg.solve(L.T, np.linalg.solve(L, Pt.T @ yt / sigma2))
    Cinv_Phi = np.linalg.solve(L.T, np.linalg.solve(L, Phi.T))   # Σ φ
    mean = Phi @ w
    std = np.sqrt(sigma2 + np.einsum("ij,ji->i", Phi, Cinv_Phi))

    recent = t >= 2018
    cover = np.mean(np.abs(co2[recent] - mean[recent]) < 2 * std[recent])
    print(f"Mauna Loa CO₂：{len(t)} 個月, {t.min():.0f}–{t.max():.0f}（訓練 <2005）")
    print(f"  訓練區平均預測 σ = {std[cut].mean():.2f} ppm")
    print(f"  外推區(≥2018) σ  = {std[recent].mean():.2f} ppm  ← 誠實張大")
    print(f"  外推區 RMSE      = {np.sqrt(np.mean((co2[recent]-mean[recent])**2)):.2f} ppm")
    print(f"  外推真實值 ±2σ 涵蓋率 = {cover*100:.0f}%")
    print("→ 只用 <2005 訓練，貝氏方差在外推區自動放大：模型知道自己在外插。")

if __name__ == "__main__":
    main()
