# 1-4 Mamba × 地震波形

- **自然現象**：長噪聲中稀疏 P 波到時
- **論文**：Gu & Dao 2023 — Mamba (Selective SSM)
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`mamba_mvp.py` — 選擇性 SSM 對噪聲 a≈1 不動、對事件 a≈0 latch，快轉常態鎖定事件

- **MVP 使用的資料集**：合成『小噪聲+稀疏大事件』序列（程式內生成）

```bash
python3 mamba_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：STEAD (STanford EArthquake Dataset)
- **來源**：https://github.com/smousavi05/STEAD
- **準備步驟**：
  1. 取 STEAD 子集(數百~數千條三分量波形，勿抓全 70GB)
  2. 帶通濾波、正規化、標注 P/S 到時
  3. 切長窗口(含長前導噪聲)

## ③ 規劃最大實驗
- **目標**：選擇性 SSM vs Transformer 在超長地震序列的 P 波到時精度/記憶體標度
- **設定**：自寫 selective-scan SSM，序列長 10³~10⁵，對比 attention
- **度量**：到時誤差、峰值記憶體、吞吐
- **算力**：GPU 0 (11GB)（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：SSM 在長序列以 O(N) 記憶體達到 attention 級精度

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `mamba_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
