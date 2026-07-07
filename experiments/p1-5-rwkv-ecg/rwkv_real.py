"""
rwkv_real.py — 單元 1-5 B 層（真實資料）：時間衰減特徵做真實 ECG 心拍分類

資料：ECG5000 心拍資料（TF 鏡像，自動下載到 data/；140 點/心拍，末欄 0/1 標籤）。
做法：用 RWKV 的時間衰減核(WKV)對每個心拍做因果平滑當特徵，最近質心分類正常/異常，
      比較「原始 vs 時間衰減特徵」的準確率。
執行：python3 rwkv_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def wkv(V, w):
    """對每列(心拍)做 RWKV 時間衰減核的因果平滑。"""
    decay = np.exp(-w); out = np.empty_like(V); n = 0.0; d = 0.0
    for t in range(V.shape[1]):
        n = decay * n + V[:, t]; d = decay * d + 1.0
        out[:, t] = n / d
    return out

def main():
    arr = np.genfromtxt(fetch(URL, "ecg5000.csv"), delimiter=",")
    X, y = arr[:, :140], arr[:, 140].astype(int)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); tr, te = idx[:4000], idx[4000:]

    def ncc(feat):                                  # 最近質心分類準確率
        c0 = feat[tr][y[tr] == 0].mean(0); c1 = feat[tr][y[tr] == 1].mean(0)
        d0 = ((feat[te]-c0)**2).sum(1); d1 = ((feat[te]-c1)**2).sum(1)
        return np.mean((d1 < d0).astype(int) == y[te])

    acc_raw = ncc(X)
    acc_wkv = ncc(wkv(X, 0.3))
    print(f"ECG5000：{len(X)} 個心拍 (正常/異常 = {np.sum(y==1)}/{np.sum(y==0)})")
    print(f"  原始訊號    最近質心準確率 = {acc_raw:.3f}")
    print(f"  時間衰減特徵 最近質心準確率 = {acc_wkv:.3f}")
    print("→ RWKV 式時間衰減核對真實 ECG 提供有效特徵，區分正常/異常心拍。")

if __name__ == "__main__":
    main()
