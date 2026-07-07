# NN — 發散、收斂、貝氏與世界模型的推導筆記

這個倉庫把一場「從第一性原理推導世界模型」的思辨，整理成**文獻回顧 + 可實作專案**。

## 主軸（一句話）

> **神經網路是一台沒有時間概念的靜態幾何機器；而真實世界是「在時間中發散、被邏輯收斂」的動態系統。**
> 二者的錯配，是所有問題的根；而「發散→收斂→隨驚訝呼吸」的循環，是逼近它的最務實路徑。

## 推導的六個層次

1. NN 的「預測」不是預報，是高維模式匹配 / 序列補全。
2. NN 底層就是 `y=σ(Wx+b)` 的空間摺疊；時間被位置編碼偽裝成座標。
3. Mamba / RWKV：不硬記時間，而在已知動力學裡「篩選干擾」。
4. 世界模型的路線之爭：遵守時間 vs 抽離時間、只保守恆不變量。
5. 只要用反向傳播，一切仍是模式匹配；真正的世界模型是邏輯 / 加減乘除。
6. 三道牆：維度詛咒、混沌、測不準 → 常數終究得「有紀律地猜」。

## 你獨立重新發明的東西（附正式名字）

| 你的直覺 | 已有的名字 | 官方倉庫 |
|----------|-----------|----------|
| 讓資料無限發散、再收斂回來 | **擴散 / score-based 生成模型** | `yang-song/score_sde_pytorch` |
| 猜「發散的方向」而非猜牆 | **score = ∇ₓ log p(x)**、Neural SDE | `google-research/torchsde` |
| 抽掉時間、還原加減乘除骨架 | **SINDy**、**Koopman** | `dynamicslab/pysindy`、`BethanyL/DeepKoopman` |
| 物理牆 / 守恆約束 | **PINN / Hamiltonian NN / 引導擴散** | `maziarraissi/PINNs`、`greydanus/hamiltonian-nn` |
| 無限膨脹 + 剪枝 | **非參數貝氏（IBP / Dirichlet process）** | `echen/dirichlet-process` |
| 用方差包住測不準 | **貝氏神經網路 / 變分推論** | `pyro-ppl/pyro` |

## 目錄

- [`docs/literature-review.md`](docs/literature-review.md) — 五大理論支柱 × 各 5 篇奠基論文，每篇標注「發想方式」與原始碼。
- [`docs/projects.md`](docs/projects.md) — 6 個可推上 GitHub 的實作專案，由易到難。
- [`docs/experiment-matrix.md`](docs/experiment-matrix.md) — **實驗矩陣**：25 個自然現象 × 3 層實驗（模擬 / 小資料 / 完整規劃）= 75 個實驗，各附公開資料集來源。
- [`experiments/`](experiments/) — 依矩陣建好的 25 個實驗目錄骨架（每個含 `data/` 與 `figures/`，環境自帶 `.venv` 守則）。
- [`code/lorenz_sindy_demo.py`](code/lorenz_sindy_demo.py) — **可直接跑**的旗艦示範：純 numpy 從 Lorenz 資料還原出三條微分方程（收斂 / 還原律）。
- [`code/lorenz_butterfly_demo.py`](code/lorenz_butterfly_demo.py) — **可直接跑**：純 numpy 蝴蝶效應 + 最大 Lyapunov 指數估計（發散原型）。

## 快速開始

```bash
# 兩支示範只需要 numpy（若主機無 numpy，請先在 .venv 內安裝，勿污染主機）
python3 code/lorenz_sindy_demo.py       # 收斂：從混沌時序還原三條 ODE
python3 code/lorenz_butterfly_demo.py   # 發散：蝴蝶效應 + Lyapunov 指數
```

預期輸出：
- `lorenz_sindy_demo`：從一段混沌時間序列，稀疏回歸**還原出 Lorenz 的 dx/dt, dy/dt, dz/dt**——親眼看到「機器從資料裡挖出加減乘除的律」。
- `lorenz_butterfly_demo`：兩條初值只差 `1e-9` 的軌跡**指數分離**，估出最大 Lyapunov 指數 λ≈0.9 與「可預測時界」——親眼看到「發散」為何無法靠記憶硬扛。

## 一句話留給未來的你

> 不要去發明一個能算出世界的模型；去發明一個「能在混沌裡發散、被邏輯收斂、隨驚訝呼吸」的模型——然後承認它的常數永遠是猜的，而這正是它像宇宙的原因。
