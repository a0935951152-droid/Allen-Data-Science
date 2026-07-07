"""
transformer_mvp.py — 單元 1-3 Demo MVP：注意力的「induction head」機制（純 numpy）

命題：注意力做的是「全域兩兩對齊」，能在序列裡找到相關位置（Vaswani 2017）。
      Transformer 能複製/外推，靠 induction head：看到 token X，就去「上一次 X 之後
      出現什麼」，據此預測下一個 token。
做法：手工構造 induction head 電路（不訓練，直接展示機制）——
      Q=當前 token 的 one-hot，K=每個位置「前一個 token」的 one-hot；
      因果遮罩下 attention[t,s] 在 (s 的前一 token == 當前 token) 時點亮，
      再用 V=該位置 token 預測「下一個」。
      任務：前半是一段各不相同的 token，後半整段重複前半 → 後半每一步都能被 induction 命中。
執行：python3 transformer_mvp.py   需求：numpy
"""
import numpy as np

rng = np.random.default_rng(1)

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def main():
    V, L = 10, 6
    half = rng.permutation(V)[:L]        # 前半：L 個互不相同的 token（模擬一段獨特 motif）
    seq = np.concatenate([half, half])   # 後半整段重複前半
    T = len(seq)
    onehot = np.eye(V)[seq]

    prev = np.zeros((T, V))              # 每個位置「前一個 token」的 one-hot
    prev[1:] = onehot[:-1]

    scores = (onehot @ prev.T) * 8.0    # induction 電路：當前 token vs 各位置的前一 token
    scores += np.triu(np.full((T, T), -1e9), k=1)   # 因果遮罩：只能看過去
    A = softmax(scores)
    pred_next = (A @ onehot).argmax(-1)

    print("序列         :", seq.tolist(), "  (後半重複前半)")
    print("預測下一 token :", pred_next.tolist())
    print("真實下一 token :", seq[1:].tolist() + ["-"])
    ok = pred_next[L:T - 1] == seq[L + 1:]           # 只評估「重複段」
    print(f"\n重複段預測正確率 = {ok.mean():.2f}  ({ok.sum()}/{len(ok)})")
    print("→ 注意力在『前一 token 相同』處點亮，複製出接下來的 token：induction head 機制。")

if __name__ == "__main__":
    main()
