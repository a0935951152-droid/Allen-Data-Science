# 2-2 Bayes-by-Backprop × CO₂

- **自然現象**：大氣 CO₂ 上升 (Keeling)
- **論文**：Blundell et al. 2015 — Weight Uncertainty
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`bnn_mvp.py` — GP 預測標準差在資料區 0.04、內插 OOD 0.40、外推 OOD 0.75（誠實喇叭口）

- **MVP 使用的資料集**：合成『兩段有資料、中間與兩端 OOD』1D 迴歸（程式內生成）

```bash
python3 bnn_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：NOAA Mauna Loa 月均 CO₂
- **來源**：https://gml.noaa.gov/ccgg/trends/data.html
- **準備步驟**：
  1. 下載 co2_mm_mlo.txt（月均序列，純文字）
  2. 去季節/建 trend+season 特徵
  3. 訓練在早期年份、外推近年當 OOD 測試

## ③ 規劃最大實驗
- **目標**：確定性 MLP / MC-Dropout / BBB / GP 四法在 CO₂ 外推的不確定性校準對比
- **設定**：四模型同資料，畫預測帶 + 可靠度圖 + 覆蓋率
- **度量**：NLL、校準誤差(ECE)、外推區覆蓋率
- **算力**：GPU 0/CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：貝氏法在 OOD 提供校準良好的方差，點估計則過度自信

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `bnn_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
