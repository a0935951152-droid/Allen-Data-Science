"""
rwkv_mvp.py — 單元 1-5 Demo MVP：RWKV 的時間衰減記憶（純 numpy）

命題：RWKV 把 softmax attention 拆成可遞迴的線性形式，用時間衰減 e^{-w} 當記憶核，
      得到「RNN 的身體、Transformer 的訓練」（Peng 2023）。
做法：實作 WKV 的時間衰減加權平均 y_t = Σ_{i≤t} e^{-w(t-i)} v_i / Σ_{i≤t} e^{-w(t-i)}。
      在稀疏脈衝序列上，比較大 w（短記憶）與小 w（長記憶）：脈衝後 y 以時間常數 1/w 衰減。
執行：python3 rwkv_mvp.py   需求：numpy
"""
import numpy as np

def wkv(v, w):
    """時間衰減線性 attention 的遞迴形式（數值穩定版）。"""
    T = len(v)
    y = np.empty(T)
    num = den = 0.0
    decay = np.exp(-w)
    for t in range(T):
        num = decay * num + v[t]
        den = decay * den + 1.0
        y[t] = num / den
    return y

def ascii_row(y, label):
    lo, hi = y.min(), y.max()
    cells = " .:-=+*#@"
    s = "".join(cells[int((v - lo) / (hi - lo + 1e-12) * (len(cells) - 1))] for v in y)
    print(f"  {label:>16} |{s}|")

def main():
    T = 60
    v = np.zeros(T)
    v[[5, 25, 45]] = 1.0               # 稀疏脈衝

    print("時間衰減記憶：脈衝在 t=5,25,45；比較不同衰減率 w")
    ascii_row(v, "輸入脈衝 v")
    for w in [0.05, 0.2, 0.6]:
        ascii_row(wkv(v, w), f"w={w} (τ≈{1/w:.0f})")
    print("\n→ w 小 = 長記憶（脈衝後拖尾長）；w 大 = 短記憶（很快遺忘）。")
    print("  每個通道自帶一個記憶時間常數 τ=1/w——ECG 上可對應不同節律尺度。")

if __name__ == "__main__":
    main()
