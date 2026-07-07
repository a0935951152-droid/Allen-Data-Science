# 4-5 Lorenz 1963 × 對流混沌

- **自然現象**：蝴蝶效應/敏感依賴
- **資料集**：合成 + 可選 ERA5
- 完整三層實驗設計（模擬 / 小資料 / 完整規劃）見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## 狀態
✅ **A 層（模擬）可跑**：`lorenz_butterfly.py`，純 numpy 展示蝴蝶效應，並用 Benettin 重正規化法估最大 Lyapunov 指數（λ≈0.908，理論 0.906）。
🟡 B 層（真實氣溫時序 Takens 嵌入）、C 層（可預測時界的資料驅動估計）待實作。

## 執行
```bash
python3 lorenz_butterfly.py      # 只需 numpy，零第三方依賴
```
預期：文字圖顯示兩條近初值軌跡指數分離後飽和；印出 λ≈0.91 與可預測時界。

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `lorenz_butterfly.py`  A 層可跑示範
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
