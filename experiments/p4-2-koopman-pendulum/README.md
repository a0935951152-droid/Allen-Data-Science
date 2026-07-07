# 4-2 Deep Koopman × 擺

- **自然現象**：單擺/雙擺相空間
- **論文**：Lusch et al. 2018 — Deep Koopman
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`koopman_mvp.py` — EDMD 多項式觀測線性化單擺，300 步線性預測誤差≈0、還原 ω=0.96

- **MVP 使用的資料集**：合成單擺軌跡(程式內生成)

```bash
python3 koopman_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：自錄擺影片 / 合成雙擺
- **來源**：(自產：手機錄擺 + tracking，或合成雙擺)
- **準備步驟**：
  1. 錄/合成擺運動，追蹤角度時序
  2. 去噪、估相空間 (q,p)
  3. 建觀測字典

## ③ 規劃最大實驗
- **目標**：Deep Koopman(autoencoder+線性 latent) 學到的模態 vs 物理振盪頻率的對應
- **設定**：autoencoder 學觀測、線性 latent 動力學，分析特徵頻率
- **度量**：多步預測誤差、Koopman 模態↔物理頻率吻合
- **算力**：GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：學到的線性 latent 頻率對應真實振盪模態

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `koopman_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
