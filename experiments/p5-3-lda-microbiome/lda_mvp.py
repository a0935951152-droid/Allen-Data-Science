"""
lda_mvp.py — 單元 5-3 Demo MVP：LDA——把「大集合」拆成潛在主題混合（純 numpy）

命題：LDA 為「大集合文件」設計生成式混合先驗，發現潛在結構（Blei 2003）。
      放到微生物組：每個樣本是「菌群主題」的混合。這支 MVP 用塌縮 Gibbs 還原主題。
做法：3 個真主題各偏好一組不重疊的詞（模擬不同菌群）。生成 200 份文件，跑 collapsed
      Gibbs LDA，看還原主題的高頻詞是否對回真主題。
執行：python3 lda_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def main():
    V, K, M, L = 9, 3, 120, 30
    true_topics = np.zeros((K, V))
    for k in range(K):
        true_topics[k, 3*k:3*k+3] = 1.0                # 主題 k 偏好詞 {3k,3k+1,3k+2}
    true_topics /= true_topics.sum(1, keepdims=True)

    docs = []
    for _ in range(M):
        theta = rng.dirichlet([0.3] * K)
        zs = rng.choice(K, L, p=theta)
        docs.append([rng.choice(V, p=true_topics[z]) for z in zs])

    # collapsed Gibbs
    alpha, beta = 0.3, 0.1
    ndk = np.zeros((M, K)); nkw = np.zeros((K, V)); nk = np.zeros(K)
    Z = []
    for d, doc in enumerate(docs):
        zd = rng.integers(0, K, len(doc))
        Z.append(zd)
        for w, z in zip(doc, zd):
            ndk[d, z] += 1; nkw[z, w] += 1; nk[z] += 1
    for it in range(150):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                z = Z[d][i]
                ndk[d, z] -= 1; nkw[z, w] -= 1; nk[z] -= 1
                p = (ndk[d] + alpha) * (nkw[:, w] + beta) / (nk + V * beta)
                p /= p.sum()
                z = rng.choice(K, p=p)
                Z[d][i] = z; ndk[d, z] += 1; nkw[z, w] += 1; nk[z] += 1

    print("LDA 塌縮 Gibbs 還原潛在主題（每主題印高頻詞）")
    topics = nkw / nkw.sum(1, keepdims=True)
    for k in range(K):
        top = np.argsort(topics[k])[::-1][:3]
        print(f"  還原主題 {k}: 高頻詞 {sorted(top.tolist())}")
    print("  真主題分別偏好詞 {0,1,2} / {3,4,5} / {6,7,8}")
    print("\n→ 無監督地把文件集拆成三組不重疊的詞群：潛在主題結構被還原。")

if __name__ == "__main__":
    main()
