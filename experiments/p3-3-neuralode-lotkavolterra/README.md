# 3-3 Neural ODE × 捕食振盪

- **自然現象**：Lotka–Volterra
- **論文**：Chen et al. 2018 — Neural ODE
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`neuralode_mvp.py` — 微分穿過 ODE 解算器，從軌跡還原 LV 四參數(誤差<3%)

- **MVP 使用的資料集**：合成 Lotka–Volterra 軌跡（程式內生成）

```bash
python3 neuralode_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃

> **✅ 已接真實資料並實測**：`python3 neuralode_real.py`（Hudson Bay 猞猁–野兔(自動下載)，首次執行自動下載到 `data/`）
> — Neural ODE 擬合 LV 有效參數，RMSE 4.3 千隻(真實資料非乾淨 LV)。

- **資料集**：Hudson Bay 猞猁–野兔年皮毛數
- **來源**：https://www.math.tamu.edu/~phoward/m442/lynxhare.dat (經典資料，多套件內建)
- **準備步驟**：
  1. 取猞猁-野兔年序列(~90 點)
  2. 對數變換/標準化
  3. 注意：真實資料不吻合乾淨 LV，預期還原扭曲版

## ③ 規劃最大實驗
- **目標**：Neural ODE(學 RHS 為 NN) vs SINDy(稀疏方程) 在同資料的可解釋性/外推
- **設定**：torchdiffeq 訓 Neural ODE，對比 SINDy 還原
- **度量**：外推 RMSE、可解釋性、對噪聲穩健度
- **算力**：GPU 0/CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：釐清『黑箱連續模型 vs 顯式方程』的取捨

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `neuralode_mvp.py`  ① 可跑 MVP
- `neuralode_real.py`  ② 真實資料實測（可跑）
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
