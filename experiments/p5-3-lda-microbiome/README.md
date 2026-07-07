# 5-3 LDA × 微生物組

- **自然現象**：群落組成
- **論文**：Blei et al. 2003 — Latent Dirichlet Allocation
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`lda_mvp.py` — 塌縮 Gibbs 還原 3 個不重疊詞群(對回真主題)

- **MVP 使用的資料集**：合成『群落=主題混合』文件(程式內生成)

```bash
python3 lda_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：HMP / QIITA 16S 微生物組
- **來源**：https://qiita.ucsd.edu/
- **準備步驟**：
  1. 下載 OTU/ASV 豐度表
  2. 過濾稀有 OTU、rarefy 或轉計數
  3. 以樣本為『文件』、OTU 為『詞』

## ③ 規劃最大實驗
- **目標**：LDA 主題 vs 宿主表型(健康/疾病)的關聯
- **設定**：對真實微生物組跑 LDA，關聯主題比例與表型
- **度量**：主題-表型關聯顯著性、預測力
- **算力**：CPU/GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：找到與表型顯著關聯的菌群『主題』

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `lda_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
