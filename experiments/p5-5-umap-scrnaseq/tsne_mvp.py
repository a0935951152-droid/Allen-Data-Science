"""
tsne_mvp.py — 單元 5-5 Demo MVP：t-SNE——保留局部鄰居把高維攤到 2D（純 numpy）

命題：t-SNE/UMAP 保留局部鄰居的拓撲結構，把高維攤到低維可視化（van der Maaten 2008）。
      這支 MVP 實作最小 t-SNE，把 50 維的 3 群資料攤到 2D，並量化「鄰居保真度」。
做法：50 維空間 3 個高斯群。用高斯相似度 P（高維）與 Student-t 相似度 Q（2D），
      對 KL(P‖Q) 做梯度下降。驗證：2D 上用 kNN 標籤純度衡量群是否被分開。
執行：python3 tsne_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def main():
    n_per, D = 60, 50
    labels = np.repeat([0, 1, 2], n_per)
    C = rng.normal(size=(3, D)) * 6
    X = np.vstack([C[c] + rng.normal(size=(n_per, D)) for c in range(3)])
    n = len(X)

    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    P = np.exp(-D2 / (2 * np.median(D2)))              # 高維相似度（固定尺度版）
    np.fill_diagonal(P, 0); P /= P.sum(); P = (P + P.T) / 2

    Y = rng.normal(size=(n, 2)) * 0.01                 # 低維座標
    vel = np.zeros_like(Y)
    for it in range(500):
        d2 = ((Y[:, None] - Y[None]) ** 2).sum(-1)
        num = 1.0 / (1.0 + d2); np.fill_diagonal(num, 0)
        Q = num / num.sum()
        g = np.zeros_like(Y)
        PQ = (P - Q) * num
        for i in range(n):
            g[i] = 4 * ((PQ[i][:, None]) * (Y[i] - Y)).sum(0)
        vel = 0.8 * vel - 200 * g
        Y += vel

    # 鄰居保真度：2D 上每點最近 5 鄰居與自己同群的比例
    d2 = ((Y[:, None] - Y[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, 1)[:, :5]
    purity = np.mean(labels[nn] == labels[:, None])
    cent = np.array([Y[labels == c].mean(0) for c in range(3)])
    gaps = [np.linalg.norm(cent[a] - cent[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    print("最小 t-SNE：50 維 3 群 → 2D")
    print(f"  2D 上 5-近鄰同群純度 = {purity:.2f}  (完美分離 = 1.00)")
    print(f"  三群 2D 中心間距     = {np.round(gaps, 1)}")
    print("\n→ 只靠『保留局部鄰居』，高維群結構被忠實攤平到 2D：t-SNE/UMAP 的核心。")

if __name__ == "__main__":
    main()
