# 3-1 DDPM × 湍流

- **自然現象**：湍流/雲紋理
- **論文**：Ho et al. 2020 — DDPM
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`ddpm_mvp.py` — 完整前向排程+反向 ancestral 取樣，從純噪聲生成雙峰(±3, 50/50)

- **MVP 使用的資料集**：合成雙峰分佈 + 解析 score（程式內生成）

```bash
python3 ddpm_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：JHU Turbulence DB 切片 / 衛星雲圖
- **來源**：https://turbulence.pha.jhu.edu/
- **準備步驟**：
  1. 用官方 pyJHTDB 取速度場切片(小塊)
  2. 轉灰度/渦量、切 patch、正規化
  3. (GPU 記憶體有限→用 32×32)

## ③ 規劃最大實驗
- **目標**：DDPM 生成湍流 patch，檢驗能譜是否符合 Kolmogorov −5/3
- **設定**：U-Net DDPM 訓 32×32 湍流 patch，比較生成/真實能譜
- **度量**：FID、能譜斜率、守恆量統計
- **算力**：GPU 0 (11GB, 小尺寸)（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：生成樣本能譜在慣性區呈 −5/3

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `ddpm_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
