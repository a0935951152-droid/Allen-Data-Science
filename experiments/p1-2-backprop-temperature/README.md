# 1-2 反向傳播 × 地表氣溫

- **自然現象**：季節週期 + 長期趨勢
- **論文**：Rumelhart, Hinton & Williams 1986 — Backpropagation
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`backprop_mvp.py` — 手刻 2 層反傳學同心圓分類，acc≈0.98；含數值/解析梯度檢核

- **MVP 使用的資料集**：合成同心圓二分類（程式內生成）

```bash
python3 backprop_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃

> **✅ 已接真實資料並實測**：`python3 backprop_real.py`（Melbourne 日最低氣溫(自動下載)，首次執行自動下載到 `data/`）
> — 手刻反傳 held-out R²=0.54，學到南半球季節(1月熱7月冷)。

- **資料集**：NOAA GHCN-Daily 單站日均溫
- **來源**：https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- **準備步驟**：
  1. 選一測站下載 .dly 檔（逐日 TMAX/TMIN）
  2. 轉日均溫、補缺、建 lag 特徵(day-of-year + 前 k 日)
  3. 標準化後切訓練/測試

## ③ 規劃最大實驗
- **目標**：梯度流健康度診斷：消失/爆炸如何隨深度變化
- **設定**：2~20 層 MLP 迴歸日溫，記錄各層梯度範數、啟用分佈
- **度量**：各層梯度範數、收斂速度、測試 RMSE
- **算力**：CPU/GPU 0 皆可（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：重現深度增加→梯度消失曲線，並用殘差/正規化緩解

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `backprop_mvp.py`  ① 可跑 MVP
- `backprop_real.py`  ② 真實資料實測（可跑）
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
