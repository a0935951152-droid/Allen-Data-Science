# 1-3 Transformer × 生物序列

- **自然現象**：DNA/蛋白長程相依與 motif
- **論文**：Vaswani et al. 2017 — Attention Is All You Need
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`transformer_mvp.py` — 手構 induction head 電路，重複段下一 token 預測 100%

- **MVP 使用的資料集**：合成『前半唯一、後半重複』token 序列（程式內生成）

```bash
python3 transformer_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃

> **✅ 已接真實資料並實測**：`python3 transformer_real.py`（E. coli 基因組 200kbp(NCBI 自動下載)，首次執行自動下載到 `data/`）
> — 4-mer 下一鹼基預測 0.34 > 隨機 0.25，真實 DNA 有可學結構。

- **資料集**：UniProt 蛋白序列子集 / NCBI E. coli 基因組
- **來源**：https://www.uniprot.org/
- **準備步驟**：
  1. 下載 FASTA（E. coli 基因組或一組同源蛋白）
  2. 字元級 tokenize（A/C/G/T 或 20 胺基酸）
  3. 切成固定長度窗口，保留含真實 motif 的區段(啟動子/TFBS)

## ③ 規劃最大實驗
- **目標**：在生物序列上訓練小 Transformer，剖析 induction/motif 注意力頭的湧現
- **設定**：2~4 層字元級 Transformer，下一鹼基/胺基酸預測 + 注意力頭功能分析
- **度量**：perplexity、注意力頭對 motif 的對齊率
- **算力**：GPU 0 (11GB) 訓小模型（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：出現可解釋的 motif-對齊注意力頭

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `transformer_mvp.py`  ① 可跑 MVP
- `transformer_real.py`  ② 真實資料實測（可跑）
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
