# 4-4 SINDy-AE × 反應擴散

- **自然現象**：BZ 反應圖樣
- **論文**：Champion et al. 2019 — SINDy Autoencoder
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`sindy_ae_mvp.py` — 從 8 維觀測 PCA 還原 2D 座標 + 線性動力學，特徵值 −0.09±1.0j 對真值

- **MVP 使用的資料集**：合成 2D 線性焦點抬升到 8 維(程式內生成)

```bash
python3 sindy_ae_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：合成反應擴散 PDE / BZ 反應影片
- **來源**：(自產：模擬 Gray-Scott/BZ，或公開 BZ 影片)
- **準備步驟**：
  1. 模擬反應擴散 PDE 生成高維圖樣序列
  2. 降採樣、攤平成觀測向量
  3. (真實)用 BZ 反應影片幀

## ③ 規劃最大實驗
- **目標**：SINDy-AE 學到的低維座標是否對應物理序參量
- **設定**：非線性 autoencoder + latent SINDy 聯合訓練
- **度量**：座標↔序參量相關、方程稀疏度、重建
- **算力**：GPU 0 (訓練不穩，先小尺寸)（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：學到可解釋座標且其動力學稀疏

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `sindy_ae_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
