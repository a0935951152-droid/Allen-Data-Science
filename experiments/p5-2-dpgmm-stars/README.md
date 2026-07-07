# 5-2 Infinite GMM × 恆星族群

- **自然現象**：恆星族群/地震群集
- **資料集**：Gaia 測光 / 地震目錄
- 完整三層實驗設計（模擬 / 小資料 / 完整規劃）見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## 狀態
🟡 骨架已建，待實作。

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
