# NN — 發散、收斂、貝氏與世界模型的推導筆記

這個倉庫把一場「從第一性原理推導世界模型」的思辨，整理成**文獻回顧 + 可實作專案**。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a0935951152-droid/Allen-Data-Science/blob/main/notebooks/colab_launcher.ipynb)
— 一鍵在 Colab 跑全部 25 個 MVP 與 11 個真實資料實驗；需 GPU 的 C 層可用 Colab 的 T4(16GB)，比本機 1080 Ti 更寬裕。

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
- [`experiments/`](experiments/) — 25 個實驗目錄，**每個都有一支純 numpy、可直接跑的 A 層 Demo MVP**，隔離出該論文的核心機制。
  每個資料夾 README 含三段：**① Demo MVP + 使用資料、② 文獻資料集準備規劃、③ 規劃最大實驗**。

## 快速開始

全部 25 支 MVP 只需要 numpy（若主機無 numpy，請先在 .venv 內安裝，勿污染主機）。任選一支：

```bash
python3 experiments/p4-1-sindy-lorenz/sindy_lorenz.py         # 發散→收斂：從混沌時序還原三條 ODE
python3 experiments/p3-2-score-particles/score_mvp.py         # 收斂：畫出 score 向量場(發散方向)
python3 experiments/p3-5-hnn-orbits/hnn_mvp.py                # 物理牆：辛積分守恆 vs Euler 漂移
python3 experiments/p2-2-bbb-co2/bnn_mvp.py                   # 貝氏：OOD 方差誠實張開
python3 experiments/p5-5-umap-scrnaseq/tsne_mvp.py            # 高維：t-SNE 攤平 3 群
```

每支都印出可驗證的結果（還原的方程、估計的指數、方差喇叭口、群純度…），對應 `docs/literature-review.md` 的一篇論文。
一鍵全跑：`for f in experiments/*/*_mvp.py experiments/*/sindy_lorenz.py experiments/*/lorenz_butterfly.py; do python3 "$f"; done`

## 一句話留給未來的你

> 不要去發明一個能算出世界的模型；去發明一個「能在混沌裡發散、被邏輯收斂、隨驚訝呼吸」的模型——然後承認它的常數永遠是猜的，而這正是它像宇宙的原因。
