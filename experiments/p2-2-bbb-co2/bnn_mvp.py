"""
bnn_mvp.py — 單元 2-2 Demo MVP：貝氏迴歸在 OOD 會「誠實示警」（純 numpy）

命題：確定性網路在分佈外(OOD)會給「自信的胡言亂語」；貝氏方法讓預測方差在遠離資料處張開
      （Blundell 2015 的動機）。這支 MVP 用高斯過程（封閉式貝氏後驗）乾淨地展示這個喇叭口。
做法：只在 x∈[-3,-1]∪[1,3] 有資料（中間與兩端皆為 OOD）。用 RBF 核 GP 做後驗，
      比較「資料區」與「OOD 區」的預測標準差——後者顯著放大。
執行：python3 bnn_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def rbf(a, b, l=0.7):
    d = a[:, None] - b[None, :]
    return np.exp(-0.5 * (d / l) ** 2)

def main():
    xtr = np.concatenate([rng.uniform(-3, -1, 12), rng.uniform(1, 3, 12)])
    ytr = np.sin(2 * xtr) + 0.05 * rng.normal(size=len(xtr))
    noise = 0.05 ** 2

    K = rbf(xtr, xtr) + noise * np.eye(len(xtr))
    Kinv = np.linalg.inv(K)
    xs = np.linspace(-5, 5, 200)
    Ks = rbf(xs, xtr)
    var = 1.0 - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks)
    std = np.sqrt(np.clip(var, 0, None))

    def region(lo, hi):
        m = (xs >= lo) & (xs <= hi)
        return std[m].mean()

    print("GP 預測標準差（越大 = 越不確定 = 誠實示警）")
    print(f"  資料區  x∈[-3,-1]∪[1,3] : {np.mean([region(-3,-1), region(1,3)]):.3f}")
    print(f"  內插 OOD x∈[-1, 1]      : {region(-1, 1):.3f}")
    print(f"  外推 OOD |x|>3          : {np.mean([region(-5,-3), region(3,5)]):.3f}")
    print("\n→ 方差在無資料處自動張成喇叭口：貝氏模型『知道自己不知道』。")

if __name__ == "__main__":
    main()
