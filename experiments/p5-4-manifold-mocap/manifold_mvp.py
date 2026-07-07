"""
manifold_mvp.py — 單元 5-4 Demo MVP：流形假設——高維資料其實在低維流形上（純 numpy）

命題：高維資料常集中在低維流形上（Fefferman 2016），這解釋了為何 NN 的降維/摺疊有效。
      這支 MVP 造一個嵌在 3D 的 2D「瑞士捲」，用 Levina–Bickel 最大似然估計其內在維度。
做法：瑞士捲本質是 2D 曲面卷進 3D。線性 PCA 會說「要 3 維」；但內在維度估計（靠 kNN
      距離的標度）會回答 ≈ 2——區分「環境維度」與「內在維度」。
執行：python3 manifold_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def main():
    n = 1500
    t = rng.uniform(1.5*np.pi, 4.5*np.pi, n)
    h = rng.uniform(0, 20, n)
    X = np.stack([t*np.cos(t), h, t*np.sin(t)], 1)     # 2D 流形嵌入 3D

    # 線性 PCA：需要幾個主成分抓住 99% 變異？
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False) ** 2
    pca_dim = np.searchsorted(np.cumsum(sv)/sv.sum(), 0.99) + 1

    # Levina–Bickel MLE 內在維度（靠 kNN 距離標度）
    k = 12
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    D.sort(1)
    Tk = D[:, 1:k+1]                                   # 前 k 個鄰居距離
    m = (k - 1) / (np.log(Tk[:, k-1:k]) - np.log(Tk[:, :k-1])).sum(1)
    intrinsic = m.mean()

    print("流形假設：環境維度 vs 內在維度（瑞士捲，真內在維 = 2）")
    print(f"  線性 PCA 需要維度 (99%變異) = {pca_dim}   (環境維度 3)")
    print(f"  Levina–Bickel 內在維度估計  = {intrinsic:.2f}   (真值 2)")
    print("\n→ 線性看是 3 維，但內在只有 2 維：資料躺在捲曲的低維流形上。")

if __name__ == "__main__":
    main()
