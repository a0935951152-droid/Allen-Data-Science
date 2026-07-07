# 4-1 SINDy × Lorenz/生態

- **自然現象**：混沌與生態震盪
- **資料集**：合成 Lorenz + 猞猁-野兔
- 完整三層實驗設計（模擬 / 小資料 / 完整規劃）見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## 狀態
✅ **A 層（模擬）可跑**：`sindy_lorenz.py`，純 numpy 從 Lorenz 混沌時序稀疏回歸還原三條 ODE。
🟡 B 層（猞猁-野兔還原 Lotka–Volterra）、C 層（噪聲/取樣率對還原率的影響）待實作。

## 執行
```bash
python3 sindy_lorenz.py          # 只需 numpy，零第三方依賴
```
預期：印出還原的 dx/dt, dy/dt, dz/dt，與真實 Lorenz 方程逐項吻合。

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `sindy_lorenz.py`  A 層可跑示範
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
