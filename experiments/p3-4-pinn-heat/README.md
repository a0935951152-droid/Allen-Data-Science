# 3-4 PINN × 熱擴散

- **自然現象**：熱擴散/波動
- **論文**：Raissi et al. 2019 — PINN
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`pinn_mvp.py` — 只最小化 PDE 殘差(物理牆)解 1D Poisson，與解析解誤差 1e-15

- **MVP 使用的資料集**：解析 f 使解為 sin(πx)+0.5sin(3πx)（程式內生成）

```bash
python3 pinn_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：解析解(無需下載) / 可選真實熱像
- **來源**：https://maziarraissi.github.io/PINNs/
- **準備步驟**：
  1. 以解析 PDE 生成配點與(可選)稀疏觀測
  2. 定義 PDE 殘差 + BC/IC 損失
  3. (真實情境)用熱像儀資料當稀疏觀測

## ③ 規劃最大實驗
- **目標**：反問題：用 PINN 從稀疏含噪觀測反推未知擴散係數場
- **設定**：全連接 PINN + autodiff 求 u_t,u_xx，聯合估 κ(x)
- **度量**：κ 還原誤差、對噪聲/取樣密度的穩健度
- **算力**：GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：在稀疏觀測下仍還原擴散係數場

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `pinn_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
