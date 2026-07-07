# 5-1 IBP × 物種性狀

- **自然現象**：性狀/基因存在-缺失
- **論文**：Griffiths & Ghahramani 2011 — Indian Buffet Process
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`ibp_mvp.py` — 模擬 IBP，平均特徵數 K 緊貼理論 α·H_N

- **MVP 使用的資料集**：IBP 先驗模擬(程式內生成)

```bash
python3 ibp_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：物種-性狀 / 基因存在矩陣
- **來源**：https://www.gbif.org/ (性狀) 或基因存在矩陣
- **準備步驟**：
  1. 取物種×性狀(或菌株×基因)二元矩陣
  2. 清理缺值、二值化
  3. 設 IBP 潛在特徵模型

## ③ 規劃最大實驗
- **目標**：潛在特徵數隨資料增長的『呼吸』曲線(呼應無限膨脹+剪枝)
- **設定**：對真實二元矩陣做 IBP 後驗推論(Gibbs/變分)
- **度量**：潛在特徵數後驗、重建、與已知功能群對照
- **算力**：CPU/GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：特徵數隨資料自適應且對應可解釋功能群

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `ibp_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
