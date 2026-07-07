# 2-1 VAE × 花形

- **自然現象**：葉形/花形形態連續變異
- **論文**：Kingma & Welling 2013 — VAE
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`vae_mvp.py` — 重參數化梯度變異數比 score-function 低約 12 倍

- **MVP 使用的資料集**：解析目標 ∇E[z²]（程式內生成）

```bash
python3 vae_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：Oxford 102 Flowers
- **來源**：https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- **準備步驟**：
  1. 下載 102flowers.tgz 影像
  2. 中心裁切、縮到 64×64、正規化
  3. (可選)以類別做條件

## ③ 規劃最大實驗
- **目標**：β-VAE 在自然花形上的 latent 維度 vs 重建/KL 權衡與解耦
- **設定**：卷積 VAE，掃 β 與 latent 維度，latent 內插看形態變形
- **度量**：重建誤差、KL、(dSprites 對照的)解耦指標
- **算力**：GPU 0 (11GB)（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：找到重建-解耦的 Pareto 前緣，內插連續且語意合理

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `vae_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
