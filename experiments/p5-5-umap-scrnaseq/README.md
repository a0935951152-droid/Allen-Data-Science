# 5-5 UMAP/t-SNE × 單細胞

- **自然現象**：細胞類型
- **論文**：van der Maaten & Hinton 2008 — t-SNE / McInnes 2018 — UMAP
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`tsne_mvp.py` — 最小 t-SNE 把 50 維 3 群攤到 2D，5-近鄰同群純度 1.00

- **MVP 使用的資料集**：合成 50 維 3 群(程式內生成)

```bash
python3 tsne_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：10x Genomics PBMC scRNA-seq
- **來源**：https://www.10xgenomics.com/datasets
- **準備步驟**：
  1. 下載 PBMC 表達矩陣(10x .h5)
  2. QC 過濾、正規化、log1p、選高變基因
  3. PCA 預降維到 ~50 維

## ③ 規劃最大實驗
- **目標**：鄰域保真/全域結構失真的定量評估(trustworthiness/continuity)
- **設定**：對 PBMC 跑 t-SNE 與 UMAP，量化鄰域保真
- **度量**：trustworthiness、continuity、已知細胞型分離度
- **算力**：GPU 0/CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：量化比較 t-SNE vs UMAP 的局部/全域保真取捨

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `tsne_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
