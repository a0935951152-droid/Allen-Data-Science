# 專案藍圖：把「發散↔收斂」的直覺實作成可跑的程式碼

> 六個專案，難度由低到高。每個都：對應你對話裡的一個想法、標注要借用哪個官方倉庫、給出最小檔案結構。
> 全部用**抽象動力系統**（Lorenz 等），不碰特定應用領域，純資工/數學。
>
> 環境守則（依 `~/CLAUDE.md`）：每個專案自建 `.venv`，用 `requirements.txt` 鎖版本，**不污染主機**。GPU 只用 GPU 0（1080 Ti），`export CUDA_VISIBLE_DEVICES=0`。本機這些實驗多半 CPU/小 GPU 即可。

---

## P1. `diverge-converge-lab`（旗艦｜最小可驗證實驗）

**對應想法**：整場對話的收斂點——「NN 會漂移、邏輯骨架可還原、物理牆能壓住發散」，一次驗證。

**做什麼**：以 Lorenz 吸引子當真值，只餵時間序列投影，跑三個對照：
- (A) 純自迴歸 MLP/RNN — 觀察幾步後開始幻覺漂移。
- (B) SINDy — 看能否直接回歸出那三條微分方程（大概率成功，很震撼）。
- (C) score/SDE 模型 + 能量或守恆懲罰項（物理牆）— 比較外推誤差的發散速度。

**借用**：`dynamicslab/pysindy`（B）、`google-research/torchsde` 或 `yang-song/score_sde_pytorch`（C）。

**檔案結構**
```
diverge-converge-lab/
├── README.md
├── requirements.txt        # numpy scipy matplotlib pysindy torch torchsde
├── data/lorenz.py          # 生成真值軌跡（三條 ODE）
├── models/
│   ├── autoregressive.py   # (A) 純 NN
│   ├── sindy_fit.py        # (B) SINDy 還原方程
│   └── constrained_sde.py  # (C) Neural SDE + 物理牆懲罰
├── experiments/compare_extrapolation.py
└── figures/                # 三種模型的誤差發散曲線
```
**成功判準**：畫出「預測誤差 vs 外推步數」，A 指數爆炸、B 幾乎零誤差、C 介於兩者且誤差有界。

---

## P2. `timestrip`（抽掉時間：SINDy + Koopman）

**對應想法**：「抽掉時間的表象，還原成加減乘除的邏輯硬骨」「相空間裡時間變線性」。

**做什麼**：對同一個非線性系統，比較兩條「去時間」路線——
- SINDy：直接回歸出微分方程（顯式加減乘除）。
- Deep Koopman：學一組座標讓動力學變線性（隱式線性化），比較預測與可解釋性。

**借用**：`dynamicslab/pysindy`、`BethanyL/DeepKoopman`。

```
timestrip/
├── requirements.txt
├── systems/{lorenz.py,pendulum.py,vanderpol.py}
├── sindy/run_sindy.py
├── koopman/deep_koopman.py     # autoencoder + 線性 latent 動力學
└── report/compare.md           # 可解釋性 vs 外推能力對照
```

---

## P3. `physics-wall-diffusion`（物理牆 × 擴散）

**對應想法**：「不猜牆、猜發散方向（score）」+「用守恆律當硬邊界」= 引導擴散。

**做什麼**：在一個玩具分佈（如受守恆約束的粒子系統狀態）上訓練 score 模型，生成時加入**守恆量懲罰/投影**，比較有牆 vs 無牆時樣本違反守恆的比例。

**借用**：`yang-song/score_sde_pytorch`（score 訓練）、`lucidrains/denoising-diffusion-pytorch`（易改的 DDPM）。

```
physics-wall-diffusion/
├── requirements.txt
├── toy_distribution.py         # 帶守恆律的目標分佈
├── score_model.py              # 學 ∇log p
├── guidance.py                 # 生成時的守恆投影（物理牆）
└── eval_conservation.py        # 有牆/無牆的違反率對比
```
**成功判準**：加牆後樣本的守恆量方差顯著下降，且不明顯犧牲樣本多樣性。

---

## P4. `bayes-vs-nn-uncertainty`（不確定性：貝氏在 OOD 會誠實示警）

**對應想法**：「NN 遇到 OOD 給自信的胡言亂語；貝氏會讓方差飆升」。

**做什麼**：在一段區間訓練，故意在**分佈外**測試。比較：
- 確定性 MLP（點預測）
- Bayes-by-Backprop / MC-Dropout BNN（帶方差）
- 高斯過程（可選，完全貝氏基準）

畫出預測帶：BNN 在外推區的信賴區間應如喇叭口張開。

**借用**：`pyro-ppl/pyro`（BNN / SVI）、`AntixK/PyTorch-VAE`（重參數化範例）。

```
bayes-vs-nn-uncertainty/
├── requirements.txt            # torch pyro matplotlib
├── data.py                     # 1D 迴歸，含 OOD 區段
├── deterministic_mlp.py
├── bnn_pyro.py
└── plot_uncertainty.py         # 三模型的預測帶對比
```

---

## P5. `growth-net`（無限膨脹 + 剪枝的雛形）

**對應想法**：「模型隨驚訝度自發增生維度，無效維度萎縮」。

**做什麼**：實作一個會依「驚訝度（重建誤差/邊際似然代理）」動態**增生神經元/維度**、並在無梯度流時**剪枝**的網路，在串流資料上觀察維度隨資料複雜度呼吸。

**借用**：非參數貝氏靈感 `echen/dirichlet-process`、`RobRomijnders/indian_buffet`；剪枝參考 lottery ticket（`facebookresearch/open_lth`）。

```
growth-net/
├── requirements.txt
├── stream_data.py              # 複雜度隨時間變化的資料流
├── growing_layer.py            # 觸發增生的臨界準則
├── pruning.py                  # 無效維度萎縮
└── watch_dimensions.py         # 畫「隱維度數 vs 時間」的呼吸曲線
```
**注意**：這是最研究性、最不穩的一個——增生/剪枝準則與梯度訓練的耦合正是領域公開難題（見 review 文件的「誠實冷水」）。當作探索，不保證收斂。

---

## P6. `selective-ssm-mini`（在已知模式裡找干擾）

**對應想法**：「新一代 RNN 不硬記時間，而是在每步做情報篩選，只留符合因果的擾動」。

**做什麼**：從零實作一個 ~200 行的極簡選擇性狀態空間 / 線性衰減遞迴（Mamba/RWKV 精神的教學版），在合成序列（常態段 + 稀疏關鍵事件）上展示它會「快轉常態、鎖定關鍵事件」。

**借用**：對照官方 `state-spaces/mamba`、`BlinkDL/RWKV-LM`，但自己寫最小版本以理解機制。

```
selective-ssm-mini/
├── requirements.txt            # torch numpy
├── ssm_mini.py                 # h_t = Ā(x)·h_{t-1} + B̄(x)·x_t
├── synthetic_task.py           # 常態噪聲 + 稀疏「需記住」的 token
└── visualize_gate.py           # 畫出門控在關鍵事件時如何開啟
```

---

## 建議推進順序

1. **P1 + P2**（最有把握、最能立刻看到「震撼結果」，SINDy 還原 Lorenz 方程幾乎必成功）。
2. **P4**（不確定性對比，圖很漂亮，概念清楚）。
3. **P3 + P6**（進階但成熟，有官方倉庫可靠）。
4. **P5**（研究性探索，預期會卡——卡住本身就是有價值的觀察）。

每個專案先做出「一張說服自己的圖」再談優化。全部推上 GitHub 時，README 用 `docs/literature-review.md` 對應的論文連結佐證動機。
