# 2-4 NUTS/HMC × 系外行星

- **自然現象**：恆星徑向速度擺動
- **論文**：Hoffman & Gelman 2014 — NUTS
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`hmc_mvp.py` — HMC(leapfrog) 採樣強相關 2D 高斯，共變異數還原 0.88/0.97

- **MVP 使用的資料集**：解析強相關 2D 高斯後驗（程式內生成）

```bash
python3 hmc_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：NASA Exoplanet Archive 徑向速度
- **來源**：https://exoplanetarchive.ipac.caltech.edu/
- **準備步驟**：
  1. 下載一顆行星的 RV 時間序列(如 51 Peg)
  2. 扣除系統速度、標準化時間
  3. 設 Keplerian 軌道似然

## ③ 規劃最大實驗
- **目標**：多行星模型選擇：高維後驗診斷與 WAIC/邊際似然比較
- **設定**：NUTS 擬合 1~3 行星 Keplerian，R̂/ESS 診斷
- **度量**：WAIC、後驗預測、參數 (K,P,e) 覆蓋
- **算力**：CPU/GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：正確辨識行星數並給出校準的軌道參數後驗

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `hmc_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
