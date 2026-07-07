"""
score_mvp.py — 單元 3-2 Demo MVP：score = ∇ₓ log p(x) 就是「發散的方向」（純 numpy）

命題：score-based 模型學的不是資料本身，而是「機率質量往哪流」的向量場 ∇log p
      （Song 2021）。反向沿此場移動即生成。這支 MVP 把 score 場畫出來，並用 Langevin 生成。
做法：目標 = 兩個 2D 高斯（(±2,0)）。解析算 score 場、用箭頭畫出（指向高密度）；
      再跑 Langevin 動力學 x←x+½ε²·score+ε·z，看樣本被 score 場帶回兩個模態。
執行：python3 score_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)
MU = np.array([[-2.0, 0.0], [2.0, 0.0]])
VAR = 0.6

def score(x):                        # x:(...,2) → ∇log p
    d = x[..., None, :] - MU         # (...,2modes,2)
    w = np.exp(-0.5 * (d ** 2).sum(-1) / VAR)      # (...,2)
    w = w / w.sum(-1, keepdims=True)
    mean = (w[..., None] * MU).sum(-2)
    return (mean - x) / VAR

ARROWS = np.array(list("→↗↑↖←↙↓↘"))
def main():
    print("score 向量場 ∇log p（箭頭 = 機率質量流向，指向兩個模態 ●）")
    ys = np.linspace(2, -2, 9)
    xs = np.linspace(-4, 4, 25)
    for y in ys:
        row = ""
        for xx in xs:
            s = score(np.array([xx, y]))
            if np.hypot(*s) < 0.3:
                row += "·"
            else:
                ang = (np.arctan2(s[1], s[0]) + np.pi/8) % (2*np.pi)
                row += ARROWS[int(ang / (np.pi/4)) % 8]
        print("  " + row)

    x = rng.normal(size=(3000, 2)) * 3      # Langevin 生成
    for _ in range(300):
        eps = 0.1
        x = x + 0.5 * eps ** 2 * score(x) + eps * rng.normal(size=x.shape)
    left = np.mean(x[:, 0] < 0)
    print(f"\nLangevin 生成後：左模態 {left:.2f} / 右模態 {1-left:.2f}  (目標 0.5/0.5)")
    print("→ 只要知道 score（發散方向），沿它走就能把噪聲收斂回資料分佈。")

if __name__ == "__main__":
    main()
