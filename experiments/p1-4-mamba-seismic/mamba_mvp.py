"""
mamba_mvp.py — 單元 1-4 Demo MVP：選擇性狀態空間的「快轉常態、鎖定事件」（純 numpy）

命題：Mamba 讓狀態空間的參數變成「輸入的函數」(selectivity)，於是模型能對常態噪聲
      幾乎不更新狀態、對關鍵事件強烈寫入（Gu & Dao 2023）。
做法：離散化遞迴 h_t = a_t h_{t-1} + b_t x_t，其中步長 Δ_t 由輸入大小決定：
      噪聲 → Δ≈0 → a≈1,b≈0（狀態幾乎不動，忽略噪聲）；
      事件(大 |x|) → Δ大 → a≈0,b≈1（狀態latch住事件值）。
      在「小噪聲 + 稀疏大事件」序列上，看狀態 h 精準追蹤最後一次事件、無視噪聲。
執行：python3 mamba_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(2)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def main():
    T = 40
    x = rng.normal(0, 0.3, T)          # 常態噪聲
    events = {8: 5.0, 23: -4.0, 34: 6.0}
    for i, v in events.items():
        x[i] = v                        # 稀疏關鍵事件

    delta = softplus(4.0 * np.abs(x) - 4.0)   # 只有大 |x| 才給大步長
    a = np.exp(-delta)                  # ≈1 忽略、≈0 latch
    b = 1 - a
    h = 0.0
    H = np.empty(T)
    for t in range(T):
        h = a[t] * h + b[t] * x[t]
        H[t] = h

    print(f"{'t':>3} {'x_t':>7} {'Δ_t':>6} {'gate a':>7} {'狀態 h':>8}  事件")
    for t in range(T):
        mk = "  <== 事件" if t in events else ""
        if t in events or t in (9, 10, 24, 35) or t < 2:
            print(f"{t:>3} {x[t]:>7.2f} {delta[t]:>6.2f} {a[t]:>7.3f} {H[t]:>8.3f}{mk}")
    print("\n→ 噪聲處 a≈1、狀態幾乎不動；事件處 a≈0、狀態瞬間 latch 到事件值。")
    print("  這就是『在已知模式裡篩選干擾』：快轉常態、只記關鍵事件。")

if __name__ == "__main__":
    main()
