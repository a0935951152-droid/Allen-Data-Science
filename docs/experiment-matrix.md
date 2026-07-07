# 實驗矩陣：25 個自然現象 × 3 層實驗 = 75 個實驗

> 把 `literature-review.md` 的 **5 支柱 × 5 篇論文 = 25 個單元**，各自：
> 1. 配一個**自然現象**（讓抽象方法落在真實世界上）。
> 2. 指定一個**公開資料集**（來源 + 規模 + 取得方式；資料「用到才下載」，不預先塞爆硬碟）。
> 3. 設計**三層實驗**：
>    - **A 模擬性質**：合成/模擬資料，最小可跑、零或極少依賴，先驗證機制。
>    - **B 小資料**：真實但小的公開資料，看方法在真世界訊號上成不成立。
>    - **C 完整規劃**：規劃級的完整實驗設計（度量、對照、標度律），不保證本回合跑完。
>
> **環境守則（依 `~/CLAUDE.md`）**：每個實驗在自己的 `.venv`，`requirements.txt` 鎖版本，**不污染主機**；需要 GPU 時只用 GPU 0（1080 Ti），`export CUDA_VISIBLE_DEVICES=0`。多數 A/B 層 CPU 即可。
> **資料下載守則**：長下載走背景並留 log；資料放各實驗的 `data/`（已 gitignore），**不進 git**；大於數百 MB 的資料集只在 C 層規劃、實作時才拉。
> 連結標注：`✅` 已知穩定官方來源；`⧗` 確切下載路徑於實作前再驗證。

命名慣例：`experiments/p{支柱}{序}-{slug}/`，例如 `p1-1-uat-sunspots/`。

---

## 支柱一：神經網路與序列結構

### 1-1 · UAT（通用逼近定理）
- **自然現象**：太陽黑子的 ~11 年准週期（非線性、多頻、非平穩）。
- **資料集**：WDC-SILSO 月均太陽黑子數，Royal Observatory of Belgium ✅（`sidc.be/SILSO`，純文字，< 1 MB）。
- **A 模擬**：合成「多正弦 + 噪聲」目標函數，單隱層 MLP 逼近，掃隱藏單元數 → 畫「寬度 vs 逼近誤差」，實證 UAT 的稠密性。
- **B 小資料**：SILSO 月黑子曲線，用 MLP 做**純函數逼近**（非預報），比較不同寬度的擬合殘差。
- **C 完整**：量測「達到 ε 誤差所需寬度 vs 目標函數頻譜/Lipschitz」的經驗標度律，跨多個自然 1D 訊號。

### 1-2 · 反向傳播（RHW 1986）
- **自然現象**：地表氣溫的季節週期 + 長期趨勢。
- **資料集**：NOAA GHCN-Daily 單站日均溫 ✅（`ncei.noaa.gov`）或 Berkeley Earth 單站序列 ⧗。
- **A 模擬**：手刻 2 層網路 + 反傳，學 XOR / 雙螺旋，畫損失下降與決策面 → 看「誤差如何分配到權重」。
- **B 小資料**：單站日溫，2 層 MLP 迴歸；用數值梯度對照解析梯度做 gradient check。
- **C 完整**：梯度流健康度診斷（各層梯度範數、消失/爆炸）隨深度變化的對照實驗。

### 1-3 · Transformer（Attention Is All You Need）
- **自然現象**：生物序列（DNA / 蛋白質）的長程相依與 motif。
- **資料集**：UniProt 蛋白序列子集 ✅（`uniprot.org`）或 E. coli 基因組 FASTA（NCBI ✅）。
- **A 模擬**：合成「複製 / induction」任務，迷你 Transformer 學會對齊匹配位置，畫注意力熱圖。
- **B 小資料**：短段 DNA，字元級 mini-Transformer 預測下一鹼基，觀察注意力是否鎖住重複 motif。
- **C 完整**：induction head 等注意力頭功能在生物序列上的湧現與剖析。

### 1-4 · Mamba（選擇性狀態空間）
- **自然現象**：地震波形——長噪聲中稀疏的 P 波到時。
- **資料集**：STEAD（STanford EArthquake Dataset）⧗ 或 IRIS 波形（`service.iris.edu`）✅。
- **A 模擬**：「長噪聲 + 稀疏需記住脈衝」合成任務，最小選擇性 SSM 展示「快轉常態、鎖定事件」（與 P6 共用核心）。
- **B 小資料**：數百條 STEAD 波形，SSM 偵測 P 波到時。
- **C 完整**：SSM vs Transformer 在超長地震序列的準確度 / 記憶體標度。

### 1-5 · RWKV（線性遞迴、時間衰減）
- **自然現象**：心電圖（ECG）節律。
- **資料集**：PhysioNet MIT-BIH Arrhythmia ✅（`physionet.org`，需 wfdb 讀取）。
- **A 模擬**：合成帶時間衰減記憶的序列，用線性遞迴（e^-w）擬合。
- **B 小資料**：MIT-BIH 單導程，RWKV 式線性遞迴做心拍分類。
- **C 完整**：時間衰減核 w 的可解釋性（每通道對應的記憶時間常數）與心律病理的關聯。

---

## 支柱二：貝氏與機率推論

### 2-1 · VAE（Auto-Encoding Variational Bayes）
- **自然現象**：葉形 / 花形的形態連續變異。
- **資料集**：Oxford 102 Flowers ✅（`robots.ox.ac.uk/~vgg/data/flowers`）或 Leafsnap ⧗。
- **A 模擬**：2D 雙月 / 環形分佈的 VAE，示範重參數化採樣與 latent 幾何。
- **B 小資料**：花朵縮圖 VAE，latent 內插看形態連續變形。
- **C 完整**：β-VAE 下 latent 維度 vs 重建/KL 權衡，自然形態的解耦程度。

### 2-2 · Bayes by Backprop（權重不確定性）
- **自然現象**：大氣 CO₂ 上升（Keeling 曲線）。
- **資料集**：Mauna Loa 月均 CO₂，NOAA GML / Scripps ✅（`gml.noaa.gov/ccgg/trends`，純文字）。
- **A 模擬**：含 OOD 區段的 1D 迴歸，BNN 信賴帶在外推區張成喇叭口。
- **B 小資料**：Mauna Loa CO₂，訓練在早期、外推近年，檢查 BNN 方差是否「誠實」放大。
- **C 完整**：確定性 MLP / MC-Dropout / BBB / GP 四法的 OOD 校準對比（可靠度圖 + 覆蓋率）。

### 2-3 · SVI（隨機變分推論）
- **自然現象**：麻疹等傳染病的流行曲線。
- **資料集**：Project Tycho（美國法定傳染病週報）✅（`tycho.pitt.edu`）。
- **A 模擬**：合成大量 1D 迴歸，SVI mini-batch 擬合，畫 ELBO 收斂。
- **B 小資料**：麻疹週病例，Poisson / SIR 迴歸的 SVI。
- **C 完整**：SVI vs 完整 VI 在資料量標度下的速度 / 準確度取捨。

### 2-4 · NUTS / HMC（哈密頓蒙地卡羅）
- **自然現象**：系外行星造成的恆星徑向速度擺動。
- **資料集**：NASA Exoplanet Archive 徑向速度 ✅（`exoplanetarchive.ipac.caltech.edu`）。
- **A 模擬**：合成漏斗 / 雙峰高維後驗，NUTS 採樣看混合與 R̂。
- **B 小資料**：單顆行星 RV 曲線，貝氏擬合軌道參數（K, P, e）後驗。
- **C 完整**：多行星模型選擇（邊際似然 / WAIC），高維後驗診斷。

### 2-5 · Pyro（機率程式平台）
- **自然現象**：傳染病 SIR 動力。
- **資料集**：COVID-19 每日病例，JHU CSSE 存檔 ✅（GitHub `CSSEGISandData/COVID-19`）。
- **A 模擬**：合成 SIR 軌跡 + 觀測噪聲，Pyro 反推 β, γ。
- **B 小資料**：某地區疫情早期，Pyro 貝氏 SIR 擬合 R₀ 後驗。
- **C 完整**：時變 R_t 的貝氏 state-space 推論。

---

## 支柱三：收斂（生成回流 + 物理牆）

### 3-1 · DDPM（去噪擴散）
- **自然現象**：湍流 / 雲的紋理。
- **資料集**：Johns Hopkins Turbulence DB 切片 ⧗（`turbulence.pha.jhu.edu`）或衛星雲圖。
- **A 模擬**：2D 玩具分佈 DDPM，逐步觀察加噪→去噪。
- **B 小資料**：小塊湍流灰度圖 DDPM 生成。
- **C 完整**：生成樣本的能譜是否符合 Kolmogorov −5/3（物理保真度檢驗）。

### 3-2 · Score-based SDE（score = ∇ₓ log p）
- **自然現象**：布朗 / 膠體粒子擴散。
- **資料集**：合成 + 公開粒子追蹤軌跡 ⧗。
- **A 模擬**：2D 分佈學 score 場，**畫出 ∇log p 向量場**——親眼看「機率質量往哪流 = 發散的方向」。
- **B 小資料**：實測粒子位移分佈，score 模型擬合。
- **C 完整**：反向 SDE 生成 vs 概率流 ODE 的取樣品質對比。

### 3-3 · Neural ODE（連續深度）
- **自然現象**：捕食者–被捕食者（Lotka–Volterra）振盪。
- **資料集**：Hudson Bay 猞猁–野兔年皮毛數（經典，多套件內建）✅。
- **A 模擬**：合成 LV 軌跡，Neural ODE 擬合連續動力。
- **B 小資料**：猞猁–野兔年資料，Neural ODE 外推。
- **C 完整**：Neural ODE vs SINDy 在同一資料的可解釋性 / 外推對比。

### 3-4 · PINN（物理資訊神經網路 = 物理牆）
- **自然現象**：熱擴散 / 波動傳播。
- **資料集**：解析解（無需下載）+ 可選真實熱像 ⧗。
- **A 模擬**：1D 熱方程 PINN，把 PDE 殘差寫進損失。
- **B 小資料**：稀疏量測點 + PINN 補全整場。
- **C 完整**：反問題——用 PINN 從稀疏觀測反推擴散係數。

### 3-5 · HNN（哈密頓神經網路 = 守恆牆）
- **自然現象**：行星軌道（二體 / 三體）。
- **資料集**：JPL Horizons 星曆 ✅（`ssd.jpl.nasa.gov/horizons`）。
- **A 模擬**：合成單擺 / 二體，HNN 學 H，能量守恆對比 baseline MLP。
- **B 小資料**：真實行星星曆片段，比較 HNN 與 Neural ODE 的長期能量漂移。
- **C 完整**：HNN vs Neural ODE 的長期能量守恆標度律。

---

## 支柱四：發散（動力系統 + 從資料還原方程）

### 4-1 · SINDy（稀疏動力學辨識）
- **自然現象**：Lorenz 對流混沌 / 生態震盪。
- **資料集**：合成 Lorenz（已有 `experiments/p4-1-sindy-lorenz/sindy_lorenz.py`）✅ + 猞猁–野兔 ✅。
- **A 模擬**：**現有旗艦 demo**，純 numpy 從 Lorenz 時序還原三條 ODE。
- **B 小資料**：猞猁–野兔資料還原 Lotka–Volterra 方程。
- **C 完整**：噪聲 / 取樣率對還原成功率的影響曲線，並與 `pysindy` 對照。

### 4-2 · Deep Koopman（線性化）
- **自然現象**：單擺 / 雙擺的相空間。
- **資料集**：合成 + 可選真實擺影片追蹤 ⧗。
- **A 模擬**：合成擺，autoencoder + 線性 latent 動力學。
- **B 小資料**：實測擺角時序，Koopman 線性化預測。
- **C 完整**：Koopman 模態 vs 物理振盪頻率的對應。

### 4-3 · Latent SDE（神經隨機微分方程）
- **自然現象**：動物的隨機遷移 / 分子布朗運動。
- **資料集**：Movebank 動物 GPS 軌跡子集 ✅（`movebank.org`）。
- **A 模擬**：合成 Ornstein–Uhlenbeck 過程，latent SDE 學 drift + diffusion。
- **B 小資料**：動物 GPS 軌跡，latent SDE 擬合。
- **C 完整**：drift / diffusion 分離的可辨識性研究。

### 4-4 · Champion（SINDy-Autoencoder）
- **自然現象**：反應–擴散圖樣（如 Belousov–Zhabotinsky 反應）。
- **資料集**：合成反應–擴散 PDE + 可選 BZ 反應影片 ⧗。
- **A 模擬**：高維觀測（合成 PDE），SINDy-AE 同時學座標與稀疏方程。
- **B 小資料**：降採樣圖樣序列。
- **C 完整**：學到的座標是否對應物理序參量。

### 4-5 · Lorenz 1963（混沌原型）
- **自然現象**：大氣對流的敏感依賴（蝴蝶效應）。
- **資料集**：合成（三行 ODE，`experiments/p4-5-lorenz-butterfly/lorenz_butterfly.py`）✅ + 可選 ERA5 再分析 ⧗。
- **A 模擬**：**新增 demo**，兩條近初值軌跡指數分離，估最大 Lyapunov 指數。
- **B 小資料**：真實氣溫時序的相空間重構（Takens 嵌入）。
- **C 完整**：可預測時界（predictability horizon）的資料驅動估計。

---

## 支柱五：大集合 / 高維 / 非參數

### 5-1 · Indian Buffet Process（無限特徵先驗）
- **自然現象**：物種的性狀 / 基因存在–缺失。
- **資料集**：合成 IBP + 可選基因存在矩陣 ⧗。
- **A 模擬**：合成 IBP 生成的二元特徵矩陣，推論潛在特徵數。
- **B 小資料**：小型物種–性狀矩陣。
- **C 完整**：潛在特徵數隨資料增長的「呼吸」曲線（呼應 P5 growth-net）。

### 5-2 · Infinite GMM（Dirichlet Process）
- **自然現象**：恆星族群 / 地震群集。
- **資料集**：Gaia 恆星測光子集 ✅（`gea.esac.esa.int/archive`）或地震目錄 ⧗。
- **A 模擬**：合成多群資料，DP-GMM 自動決定群數。
- **B 小資料**：恆星色–星等圖聚類。
- **C 完整**：群數後驗 vs 真實族群數的模型選擇評估。

### 5-3 · LDA（潛在狄利克雷分配）
- **自然現象**：微生物群落組成。
- **資料集**：人類微生物組 16S，HMP / QIITA ✅（`hmpdacc.org` / `qiita.ucsd.edu`）。
- **A 模擬**：合成「群落 = 主題混合」，LDA 還原主題。
- **B 小資料**：腸道菌群樣本，LDA 找菌群「主題」。
- **C 完整**：主題 vs 宿主表型的關聯。

### 5-4 · Manifold Hypothesis（流形假設檢驗）
- **自然現象**：人體動作（低維肌骨自由度）。
- **資料集**：CMU Motion Capture ✅（`mocap.cs.cmu.edu`）。
- **A 模擬**：合成嵌入高維的低維流形（swiss roll），估內在維度。
- **B 小資料**：MoCap 姿態，估內在維度（應遠小於關節數）。
- **C 完整**：內在維度 vs 動作複雜度的關係。

### 5-5 · UMAP / t-SNE（高維可視化）
- **自然現象**：單細胞基因表達的細胞類型。
- **資料集**：10x Genomics PBMC scRNA-seq ✅（`10xgenomics.com/datasets`）。
- **A 模擬**：合成多簇高維，t-SNE / UMAP 可視化 + 保真度量測。
- **B 小資料**：PBMC scRNA-seq 降維看細胞群。
- **C 完整**：鄰域保真 / 全域結構失真的定量評估（trustworthiness / continuity）。

---

## 進度追蹤

| 狀態 | 意義 |
|---|---|
| ✅ 可跑 | 已有可執行程式碼 |
| 🟡 骨架 | 目錄與 README 已建，待實作 |
| ⬜ 規劃 | 僅本文件的設計 |

- **✅ A 層 Demo MVP（25/25 全部可跑）**：每個 `experiments/pX-Y-…/` 各有一支純 numpy、零第三方依賴的 MVP，
  隔離出該論文的核心機制並印出可驗證結果（見各資料夾 README 的「① Demo MVP」）。
- **🟡 B 層（小資料）**：各單元 README 的「② 文獻資料集：準備規劃」已列來源與前處理步驟，待下載實作。
- **⬜ C 層（規劃最大實驗）**：各單元 README 的「③ 規劃最大實驗」已寫出目標/設定/度量/判準。

> 推進原則（沿用 `projects.md`）：**每個實驗先做出「一張說服自己的圖」再談優化。** A 層先跑通機制（已完成），
> B 層驗證真實資料，C 層才追標度律與嚴謹度量。每個資料夾 README = ①MVP+資料 ②文獻資料集準備 ③規劃最大實驗。
