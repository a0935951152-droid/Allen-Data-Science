# 3-5 HNN × 行星軌道

- **自然現象**：二體/三體軌道
- **論文**：Greydanus et al. 2019 — Hamiltonian NN
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`hnn_mvp.py` — 辛積分(尊重∂H)能量漂移 1.8% vs 樸素 Euler 696%

- **MVP 使用的資料集**：單擺哈密頓量(程式內生成)

```bash
python3 hnn_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃

> **✅ 已接真實資料並實測**：`python3 hnn_real.py`（JPL Horizons 地球星曆(API 自動下載)，首次執行自動下載到 `data/`）
> — 真實行星軌道比能量守恆到 6e-4：HNN 內建的不變量。

- **資料集**：JPL Horizons 星曆
- **來源**：https://ssd.jpl.nasa.gov/horizons/
- **準備步驟**：
  1. 查詢行星位置/速度星曆(向量表)
  2. 無量綱化(關鍵！)成正則座標 (q,p)
  3. 切訓練片段

## ③ 規劃最大實驗
- **目標**：HNN vs Neural ODE 的長期能量守恆標度律
- **設定**：HNN 學 H、對比 Neural ODE 學 f，長時間 rollout
- **度量**：能量漂移 vs 時間/步長、軌道穩定性
- **算力**：GPU 0/CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：HNN 長期能量有界，優於無結構模型

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `hnn_mvp.py`  ① 可跑 MVP
- `hnn_real.py`  ② 真實資料實測（可跑）
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
