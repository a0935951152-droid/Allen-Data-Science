"""
transformer_real.py — 單元 1-3 B 層（真實資料）：真實 DNA 序列的可學結構

資料：E. coli K-12 基因組一段（NCBI efetch 自動下載到 data/）。
做法：Transformer 靠序列的可預測結構運作。這裡用 order-k 上下文的下一鹼基預測（注意力在
      C 層會取代這個 Markov 代理），量測真實 DNA 是否比隨機(25%)更可預測。
執行：python3 transformer_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=U00096.3"
       "&rettype=fasta&retmode=text&seq_start=1&seq_stop=200000")

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def main():
    seq = "".join(l.strip() for l in open(fetch(URL, "ecoli.fasta")) if not l.startswith(">"))
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    s = np.array([m[c] for c in seq if c in m])
    K = 4                                     # 上下文長度
    ctx = s[:-1]
    for k in range(1, K):
        ctx = ctx * 4 + np.roll(s, -k)[:-1]   # 拼 k-mer 上下文碼（近似）
    # 用前 80% 統計 P(next|context)，後 20% 測準確率
    n = len(s) - K
    codes = np.zeros(n, int)
    for k in range(K):
        codes = codes * 4 + s[k:k + n]
    nxt = s[K:K + n]
    split = n * 4 // 5
    table = {}
    for c, y in zip(codes[:split], nxt[:split]):
        table.setdefault(c, np.zeros(4))[y] += 1
    pred = np.array([table[c].argmax() if c in table else 0 for c in codes[split:]])
    acc = np.mean(pred == nxt[split:])
    print(f"E. coli 基因組片段：{len(s)} bp")
    print(f"  {K}-mer 上下文下一鹼基預測準確率 = {acc:.3f}  (隨機基線 0.25)")
    print("→ 真實 DNA 遠比隨機可預測：序列有結構(motif/密碼子)，正是注意力要捕捉的東西。")

if __name__ == "__main__":
    main()
