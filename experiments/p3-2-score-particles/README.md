# 3-2 Score SDE × 粒子擴散

- **自然現象**：布朗/膠體粒子
- **論文**：Song et al. 2021 — Score-Based SDE
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`score_mvp.py` — 解析 score 向量場(指向模態) + Langevin 生成回 50/50 雙模態

- **MVP 使用的資料集**：合成 2D 雙高斯（程式內生成）

```bash
python3 score_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：公開膠體/單分子粒子追蹤軌跡
- **來源**：https://www.movebank.org/ (或 soft-matter/trackpy 範例資料)
- **準備步驟**：
  1. 取粒子追蹤位移資料(或用 trackpy 產生)
  2. 算位移分佈/擴散係數
  3. 標準化成 score 訓練樣本

## ③ 規劃最大實驗
- **目標**：學 score 網路後，反向 SDE 生成 vs 概率流 ODE 的取樣品質對比
- **設定**：訓 score MLP，比較 reverse-SDE 與 PF-ODE 取樣
- **度量**：分佈距離(MMD)、生成的擴散係數保真
- **算力**：GPU 0/CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：兩種反向取樣皆還原正確擴散統計，並比較效率

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `score_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
