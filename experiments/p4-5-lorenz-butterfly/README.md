# 4-5 Lorenz 1963 × 對流混沌

- **自然現象**：蝴蝶效應/敏感依賴
- **論文**：Lorenz 1963 — Deterministic Nonperiodic Flow
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`lorenz_butterfly.py` — 蝴蝶效應指數分離 + Benettin 法估最大 Lyapunov λ=0.908(理論 0.906)

- **MVP 使用的資料集**：合成 Lorenz(三行 ODE，程式內生成)

```bash
python3 lorenz_butterfly.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃

> **✅ 已接真實資料並實測**：`python3 butterfly_real.py`（Melbourne 氣溫 Takens 嵌入(自動下載)，首次執行自動下載到 `data/`）
> — 去季節殘差發散曲線第 1 步即飽和 → 真實天氣近高維隨機、非低維混沌(對照合成 Lorenz)。

- **資料集**：ERA5 再分析 / 真實氣溫時序
- **來源**：https://cds.climate.copernicus.eu/ (ERA5)
- **準備步驟**：
  1. 取單點氣溫/位勢高度時序
  2. 去季節、標準化
  3. 相空間重構(Takens 嵌入)

## ③ 規劃最大實驗
- **目標**：可預測時界(predictability horizon)的資料驅動估計
- **設定**：對真實氣象時序做 Takens 嵌入 + 最大 Lyapunov 估計
- **度量**：λ、可預測時界、與數值天氣預報極限對照
- **算力**：CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：從資料估出與已知大氣可預測性一致的時界

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `lorenz_butterfly.py`  ① 可跑 MVP
- `butterfly_real.py`  ② 真實資料實測（可跑）
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
