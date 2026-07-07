# 4-3 Latent SDE × 動物遷移

- **自然現象**：隨機遷移/布朗
- **論文**：Li et al. 2020 — Latent SDE (torchsde)
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`latentsde_mvp.py` — 從 OU 路徑同時還原 drift θ 與 diffusion σ(誤差<2%)

- **MVP 使用的資料集**：合成 Ornstein–Uhlenbeck 路徑(程式內生成)

```bash
python3 latentsde_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：Movebank 動物 GPS 軌跡
- **來源**：https://www.movebank.org/
- **準備步驟**：
  1. 下載某物種 GPS 軌跡(需帳號)
  2. 重取樣到近等時距、投影座標
  3. 處理不規則取樣/缺測

## ③ 規劃最大實驗
- **目標**：latent SDE 的 drift/diffusion 分離可辨識性研究
- **設定**：torchsde 訓 latent SDE，變不同觀測密度/噪聲
- **度量**：drift/diffusion 還原誤差、可辨識邊界
- **算力**：GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：釐清何種取樣密度下 drift 與 diffusion 可分離辨識

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `latentsde_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
