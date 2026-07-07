"""
ibp_mvp.py — 單元 5-1 Demo MVP：印度自助餐過程(IBP)——維度自發增生（純 numpy）

命題：IBP 定義一個「無限多潛在特徵」的先驗，但只觀測到有限子集，讓模型維度可以自發增長
      （Griffiths & Ghahramani 2011）——「無限膨脹」的數學核心。
做法：模擬 IBP：第 i 位顧客以機率 m_k/i 取已有菜 k，另點 Poisson(α/i) 道新菜。
      看使用中的特徵數 K 如何隨資料量 N 成長，並對照理論期望 α·H_N（H_N=調和數）。
執行：python3 ibp_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def simulate_ibp(N, alpha):
    dishes = []                       # 每道菜目前的取用人數 m_k
    Ks = []
    for i in range(1, N + 1):
        for k in range(len(dishes)):
            if rng.random() < dishes[k] / i:
                dishes[k] += 1
        for _ in range(rng.poisson(alpha / i)):   # 點新菜
            dishes.append(1)
        Ks.append(len(dishes))
    return np.array(Ks)

def main():
    alpha, N, runs = 3.0, 300, 400
    Ks = np.array([simulate_ibp(N, alpha) for _ in range(runs)]).mean(0)   # 多次平均
    H = np.cumsum(1.0 / np.arange(1, N + 1))          # 調和數
    print(f"IBP：使用中的潛在特徵數 K 隨資料量 N 的成長（{runs} 次模擬平均）")
    print(f"{'N':>5} | {'平均 K':>8} | {'理論 α·H_N':>10}")
    for n in [10, 50, 100, 200, 300]:
        print(f"{n:>5} | {Ks[n-1]:>8.1f} | {alpha*H[n-1]:>10.1f}")
    print("\n→ 特徵數不預設、隨資料自發增生，平均緊貼理論 α·H_N ~ α·ln N：無限先驗、有限使用。")

if __name__ == "__main__":
    main()
