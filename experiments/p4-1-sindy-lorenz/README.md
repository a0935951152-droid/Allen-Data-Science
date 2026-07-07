# 4-1 SINDy × Lorenz/生態

- **自然現象**：混沌與生態震盪
- **論文**：Brunton et al. 2016 — SINDy
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`sindy_lorenz.py` — 純 numpy 從 Lorenz 混沌時序稀疏回歸還原三條 ODE(逐項吻合)

- **MVP 使用的資料集**：合成 Lorenz 軌跡(程式內生成)

```bash
python3 sindy_lorenz.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃

> **✅ 已接真實資料並實測**：`python3 sindy_real.py`（Hudson Bay 猞猁–野兔(自動下載)，首次執行自動下載到 `data/`）
> — SINDy 還原含 xy 交互項的 LV 定性骨架(野兔 +x−xy、猞猁 +xy−y)。

- **資料集**：Hudson Bay 猞猁–野兔
- **來源**：https://www.math.tamu.edu/~phoward/m442/lynxhare.dat
- **準備步驟**：
  1. 取猞猁-野兔年序列
  2. 平滑估導數、建多項式庫
  3. 注意真實資料非乾淨 LV

## ③ 規劃最大實驗
- **目標**：噪聲/取樣率對 SINDy 還原成功率的影響曲線，並與 pysindy 對照
- **設定**：掃噪聲強度×取樣率，統計正確項辨識率
- **度量**：係數誤差、支撐集正確率、與 pysindy 一致性
- **算力**：CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：畫出還原成功的 (噪聲, 取樣率) 相圖

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `sindy_lorenz.py`  ① 可跑 MVP
- `sindy_real.py`  ② 真實資料實測（可跑）
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
