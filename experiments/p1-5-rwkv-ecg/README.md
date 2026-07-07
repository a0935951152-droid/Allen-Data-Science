# 1-5 RWKV × 心電圖

- **自然現象**：ECG 節律
- **論文**：Peng et al. 2023 — RWKV
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`rwkv_mvp.py` — 時間衰減線性 attention(WKV)，記憶時間常數 τ=1/w 可調

- **MVP 使用的資料集**：合成稀疏脈衝序列（程式內生成）

```bash
python3 rwkv_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：PhysioNet MIT-BIH Arrhythmia
- **來源**：https://physionet.org/content/mitdb/
- **準備步驟**：
  1. 用 wfdb 讀取單導程訊號與心拍標註
  2. R 峰分段、重取樣、正規化
  3. 切心拍窗口 + 類別標籤(N/V/…)

## ③ 規劃最大實驗
- **目標**：RWKV 式線性遞迴做心律分類，並解讀每通道時間衰減核 w 的生理意義
- **設定**：多通道時間衰減遞迴 + 分類頭；分析 w 分佈 vs 病理
- **度量**：分類 F1、w↔節律尺度的對應
- **算力**：GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：達到基線分類水準且 w 提供可解釋的記憶尺度

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `rwkv_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
