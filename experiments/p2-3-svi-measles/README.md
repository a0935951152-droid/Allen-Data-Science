# 2-3 SVI × 麻疹

- **自然現象**：傳染病流行曲線
- **論文**：Hoffman et al. 2013 — Stochastic Variational Inference
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`svi_mvp.py` — mini-batch 隨機梯度優化 ELBO，變分後驗 (m,s) 對上封閉式後驗

- **MVP 使用的資料集**：合成高斯資料（程式內生成）

```bash
python3 svi_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：Project Tycho 美國法定傳染病週報
- **來源**：https://www.tycho.pitt.edu/
- **準備步驟**：
  1. 下載麻疹州級週病例(CSV)
  2. 彙整為週序列、對齊人口
  3. 設 Poisson/SIR 觀測模型

## ③ 規劃最大實驗
- **目標**：SVI vs 完整批次 VI 在資料量標度下的速度/準確度取捨
- **設定**：Poisson 迴歸/SIR，mini-batch 大小與步長排程掃描
- **度量**：ELBO 收斂步數、每步時間、後驗準確度
- **算力**：CPU/GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：SVI 在大資料下以更少 wall-time 達到同等後驗品質

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `svi_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
