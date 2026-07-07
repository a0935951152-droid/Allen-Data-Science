"""
hnn_mvp.py — 單元 3-5 Demo MVP：哈密頓結構 → 能量守恆（守恆牆）（純 numpy）

命題：HNN 不直接學動力學，而是學哈密頓量 H，再用 ∂H 導出運動方程，於是能量守恆天生被滿足
      （Greydanus 2019）。這支 MVP 隔離出「為何守恆」：比較忽略哈密頓結構的樸素積分(Euler)
      與尊重辛結構的積分(symplectic)，看能量是否漂移。
做法：單擺 H=½p²+(1-cosq)。樸素 Euler vs 辛 leapfrog，各積分同樣時長，追蹤能量。
      Euler 能量指數漂移；辛積分能量長期有界——這正是 HNN 用 ∂H 帶來的結構性好處。
執行：python3 hnn_mvp.py   需求：numpy
"""
import numpy as np

def H(q, p):        return 0.5 * p ** 2 + (1 - np.cos(q))
def dHdq(q):        return np.sin(q)

def euler(q, p, dt, n):
    E = []
    for _ in range(n):
        q, p = q + dt * p, p - dt * dHdq(q)      # 樸素前向 Euler（不尊重辛結構）
        E.append(H(q, p))
    return np.array(E)

def symplectic(q, p, dt, n):
    E = []
    for _ in range(n):
        p = p - dt * dHdq(q)                      # 辛 leapfrog（尊重 ∂H 結構）
        q = q + dt * p
        E.append(H(q, p))
    return np.array(E)

def main():
    q0, p0, dt, n = 0.0, 1.2, 0.05, 4000
    E0 = H(q0, p0)
    Ee = euler(q0, p0, dt, n)
    Es = symplectic(q0, p0, dt, n)
    print("單擺能量守恆：樸素 Euler vs 辛積分（初始能量 E0 = %.4f）" % E0)
    print(f"  Euler     末端能量 = {Ee[-1]:.4f}   漂移 {abs(Ee[-1]-E0)/E0*100:6.1f}%")
    print(f"  Symplectic 末端能量 = {Es[-1]:.4f}   漂移 {abs(Es[-1]-E0)/E0*100:6.1f}%")
    print(f"  Euler 能量標準差 = {Ee.std():.4f}   辛積分 = {Es.std():.4f}")
    print("\n→ 尊重哈密頓(辛)結構，能量長期有界；忽略它則能量漂移。")
    print("  HNN 學 H 再用 ∂H 導出動力學，就是把這個『守恆牆』內建進模型。")

if __name__ == "__main__":
    main()
