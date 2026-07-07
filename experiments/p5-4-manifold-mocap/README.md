# 5-4 流形假設 × 人體動作

- **自然現象**：低維肌骨自由度
- **論文**：Fefferman et al. 2016 — Testing the Manifold Hypothesis
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`manifold_mvp.py` — 瑞士捲 Levina–Bickel 內在維度估計 2.14(真值 2) vs 環境維 3

- **MVP 使用的資料集**：合成瑞士捲(程式內生成)

```bash
python3 manifold_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：CMU Motion Capture
- **來源**：http://mocap.cs.cmu.edu/
- **準備步驟**：
  1. 下載 .asf/.amc 動作序列
  2. 轉關節角/3D 座標、對齊
  3. 攤平成高維姿態向量

## ③ 規劃最大實驗
- **目標**：內在維度 vs 動作複雜度的關係
- **設定**：對多類 MoCap 動作估內在維度，對比關節自由度
- **度量**：內在維度、與動作類別複雜度的相關
- **算力**：CPU（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：內在維度遠小於關節數且隨動作複雜度上升

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `manifold_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
