# 5-2 Infinite GMM × 恆星族群

- **自然現象**：恆星族群/地震群集
- **論文**：Rasmussen 2000 — Infinite GMM (DP)
- 高層設計索引見 [`../../docs/experiment-matrix.md`](../../docs/experiment-matrix.md)。

## ① Demo MVP（可跑）
`dpgmm_mvp.py` — CRP 塌縮 Gibbs 不指定群數，自動收斂到真實 3 群

- **MVP 使用的資料集**：合成 3 群 2D 資料(程式內生成)

```bash
python3 dpgmm_mvp.py          # 純 numpy，零第三方依賴
```

## ② 文獻資料集：準備規劃
- **資料集**：Gaia 恆星測光子集
- **來源**：https://gea.esac.esa.int/archive/
- **準備步驟**：
  1. ADQL 查詢一片天區的 (BP-RP, G) 測光
  2. 去紅化、品質過濾
  3. 標準化特徵

## ③ 規劃最大實驗
- **目標**：DP-GMM 群數後驗 vs 真實族群數的模型選擇評估
- **設定**：對恆星色-星等圖做 DP-GMM，比較 K 後驗與已知星團數
- **度量**：群數後驗、與已知成員對照的純度
- **算力**：CPU/GPU 0（依 `~/CLAUDE.md` 只用 GPU 0 / 1080 Ti 11GB）
- **成功判準**：自動辨識的族群數與天文已知一致

## 環境（依 ~/CLAUDE.md）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 勿污染主機
export CUDA_VISIBLE_DEVICES=0           # 需要 GPU 時只用 1080 Ti
```

## 結構
- `dpgmm_mvp.py`  ① 可跑 MVP
- `data/`  資料（gitignore，用到才下載）
- `figures/`  「說服自己的那張圖」
