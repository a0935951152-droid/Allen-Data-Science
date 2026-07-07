# 1-1 UAT × 太陽黑子

- **自然現象**：太陽黑子 ~11 年准週期
- **論文**：Cybenko/Hornik 1989 — Universal Approximation
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`uat_mvp.py` — 隨機特徵單隱層寬度掃描，MSE 隨寬度→0

- **MVP 使用的資料集**：合成多頻正弦 f(x)=sin(3πx)+0.5sin(7πx)（程式內生成，無需下載）

```bash
python3 uat_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：WDC-SILSO 月均太陽黑子數
- **來源**：https://www.sidc.be/SILSO/datafiles
- **準備步驟**：
  1. 下載 SN_m_tot_V2.0.txt（月均序列，<1MB）
  2. 解析年月與黑子數欄，正規化到 [0,1]
  3. 切訓練/驗證：以時間 t 為輸入做純函數逼近（非預報）

## ③ 規劃最大實驗
- **目標**：量測『達到 ε 逼近誤差所需寬度』vs 目標函數頻譜複雜度的經驗標度律
- **設定**：在多個自然 1D 訊號(黑子、氣溫、潮汐)上，掃寬度 8~4096 + 正式訓練(SGD)
- **度量**：MSE、達標寬度、與訊號功率譜的相關
- **算力**：CPU 即可（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：得到 width∝f(頻寬) 的單調標度曲線，跨訊號一致

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `uat_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
