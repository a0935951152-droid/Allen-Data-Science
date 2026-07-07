# 2-5 Pyro × 傳染病

- **自然現象**：SIR 動力
- **論文**：Bingham et al. 2019 — Pyro
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`sir_mvp.py` — Metropolis 從合成疫情曲線反推 β,γ,R₀（後驗 R₀=2.95 對真值 3.0）

- **MVP 使用的資料集**：合成 SIR + 觀測噪聲（程式內生成）

```bash
python3 sir_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃

> **✅ 已接真實資料並實測**：`python3 sir_real.py`（JHU CSSE COVID 每日確診(自動下載)，首次執行自動下載到 `data/`）
> — 義/德/韓早期病例估 R₀ ≈ 2.2–2.6，與文獻量級相符。

- **資料集**：JHU CSSE COVID-19 每日病例
- **來源**：https://github.com/CSSEGISandData/COVID-19
- **準備步驟**：
  1. clone 或抓某地區每日確診序列
  2. 平滑週末效應、對齊人口
  3. 設觀測模型(NegBinom)

## ③ 規劃最大實驗
- **目標**：時變再生數 R_t 的貝氏 state-space 推論
- **設定**：Pyro 隨機變分/NUTS 擬合含時變 β(t) 的 SIR/SEIR
- **度量**：R_t 後驗帶、與官方估計對照
- **算力**：GPU 0/CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：R_t 軌跡與干預時點吻合且不確定性合理

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `sir_mvp.py`  ① 可跑 MVP
- `sir_real.py`  ② 真實資料實測（可跑）
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
