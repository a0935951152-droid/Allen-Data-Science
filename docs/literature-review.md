# 文獻回顧：發散 × 收斂 × 貝氏 × NN × 大集合

> 這份回顧的組織方式，來自一場「從第一性原理推導世界模型」的思辨。
> 主軸：**NN 是沒有時間概念的靜態幾何機器；世界是在時間中發散、被邏輯收斂的動態系統。**
> 五大理論支柱，每支柱挑 5 篇奠基/代表論文，並標注兩件事：
> 1. **發想方式** —— 這篇研究是靠哪一個「心智動作」被想出來的（這是你真正要學的東西）。
> 2. **原始碼** —— 官方或最權威的實作倉庫。
>
> 標注慣例：`✅` = 已查證官方倉庫存在；引用格式為「作者 年份 — 標題 (arXiv/期刊)」。

---

## 支柱一：神經網路與序列結構（NN 的本質是空間摺疊）

對應對話段落：「`y=σ(Wx+b)` 就是空間扭曲」「時間被平鋪成座標」「Mamba/RWKV 在已知模式裡找干擾」。

| # | 論文 | 發想方式 | 原始碼 |
|---|------|----------|--------|
| 1 | Cybenko 1989 / Hornik 1989 — *Universal Approximation Theorem* | 反問「單隱層 MLP 到底能表示哪些函數」，用泛函分析證明它稠密於連續函數空間 → 奠定「NN=通用函數逼近器」。 | 理論，無程式碼 |
| 2 | Rumelhart, Hinton & Williams 1986 — *Learning representations by back-propagating errors* (Nature 323:533) | 把「誤差如何分配到每個權重」用鏈式法則反向傳遞 → 讓多層網路可訓練。 | 理論，見任何 autograd 框架 |
| 3 | Vaswani et al. 2017 — *Attention Is All You Need* (arXiv:1706.03762) | 大膽拿掉遞迴，只留「全域兩兩對齊」的注意力，換取完全並行 → Transformer。 | ✅ `tensorflow/tensor2tensor`、`huggingface/transformers` |
| 4 | Gu & Dao 2023 — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (arXiv:2312.00752) | 讓狀態空間模型的 A,B,C 變成**輸入的函數**（selectivity），用 parallel scan 硬體並行 → O(N) 取代 attention 的 O(N²)。 | ✅ `state-spaces/mamba` |
| 5 | Peng et al. 2023 — *RWKV: Reinventing RNNs for the Transformer Era* (arXiv:2305.13048) | 把 softmax attention 用數學拆成可**遞迴**的線性形式（時間衰減 e^-w）→ RNN 的身體、Transformer 的訓練。 | ✅ `BlinkDL/RWKV-LM` |

**這支柱的統一發想**：不斷逼問「這個結構到底在幾何上做什麼」，然後把最貴的操作（遞迴 / O(N²) attention）換成等價但更便宜的形式。

---

## 支柱二：貝氏與機率推論（把「猜」工程化）

對應對話段落：「常數最終得用猜的」「貝氏用方差包住測不準」「把精確積分變成優化」。

| # | 論文 | 發想方式 | 原始碼 |
|---|------|----------|--------|
| 1 | Kingma & Welling 2013 — *Auto-Encoding Variational Bayes* (VAE, arXiv:1312.6114) | 用一個 NN **攤銷（amortize）**後驗推論，靠重參數化技巧讓「採樣」變得可微 → 把貝氏塞進梯度下降。 | ✅ `AntixK/PyTorch-VAE`（權威復現） |
| 2 | Blundell et al. 2015 — *Weight Uncertainty in Neural Networks* (Bayes by Backprop, arXiv:1505.05424) | 把每個權重從「一個數字」升級成「一個分佈」，對 ELBO 做重參數化梯度 → 網路自帶不確定性。 | ✅ 見 `pyro-ppl/pyro` tutorials；`nitarshan/bayes-by-backprop` |
| 3 | Hoffman et al. 2013 — *Stochastic Variational Inference* (arXiv:1206.7051) | 把變分推論改寫成**隨機優化**，每步只用一小批資料 → VI 能吃海量資料。 | 見 `pyro` / `pymc` |
| 4 | Hoffman & Gelman 2014 — *The No-U-Turn Sampler* (NUTS/HMC, arXiv:1111.4246) | 借用**哈密頓動力學**讓 MCMC 在高維空間沿等能量面高效移動，自動決定步數 → 高維採樣可行。 | ✅ `stan-dev/stan`、`pyro` |
| 5 | Bingham et al. 2019 — *Pyro: Deep Universal Probabilistic Programming* (arXiv:1810.09538) | 把「機率模型」與「深度學習框架」接起來，讓隨機變分推論成為一等公民 → 貝氏×NN 的工程平台。 | ✅ `pyro-ppl/pyro` |

**這支柱的統一發想**：精確貝氏要算不可能的積分，於是全部改問「能不能用一個可微的優化，逼近那個積分」——這正是「有紀律地猜」。

---

## 支柱三：收斂（生成回流 + 物理牆）

對應對話段落：「發散的資料由 NN 收斂」「物理牆」「猜發散方向 = score」。**這是你獨立重新發明的範式。**

| # | 論文 | 發想方式 | 原始碼 |
|---|------|----------|--------|
| 1 | Ho et al. 2020 — *Denoising Diffusion Probabilistic Models* (arXiv:2006.11239) | 前向不斷加噪把資料**發散**成高斯，訓練一個網路學會逐步去噪把它**收斂**回資料流形。 | ✅ `hojonathanho/diffusion`；`lucidrains/denoising-diffusion-pytorch` |
| 2 | Song et al. 2021 — *Score-Based Generative Modeling through SDEs* (arXiv:2011.13456) | 把離散擴散統一成連續 **SDE**，網路學的東西是 **score = ∇ₓ log p(x)**，即「機率質量往哪流」的向量場——**就是你說的「猜發散的方向」**。反向 SDE 即生成。 | ✅ `yang-song/score_sde`、`yang-song/score_sde_pytorch` |
| 3 | Chen et al. 2018 — *Neural Ordinary Differential Equations* (arXiv:1806.07366) | 把殘差網路的一層看成 ODE 的一個離散步，乾脆直接交給 ODE solver + 伴隨法 → 連續深度、O(1) 記憶。 | ✅ `rtqichen/torchdiffeq` |
| 4 | Raissi, Perdikaris & Karniadakis 2019 — *Physics-Informed Neural Networks* (JCP; arXiv:1711.10561) | 把 PDE 殘差**直接寫進損失函數**，逼網路的解自動滿足物理律 → 這就是「物理牆」的可微版本。 | ✅ `maziarraissi/PINNs` |
| 5 | Greydanus, Dzamba & Yosinski 2019 — *Hamiltonian Neural Networks* (arXiv:1906.01563) | 不讓網路直接學動力學，而是學**哈密頓量 H**，再用 ∂H 導出運動方程 → 能量守恆天生被滿足。 | ✅ `greydanus/hamiltonian-nn` |

**這支柱的統一發想**：不去限制「內容」，而去限制「總量／方向」。擴散限制的是「往資料流形收斂」，PINN/HNN 限制的是「守恆律」。你的「物理牆」= 受約束 / 引導擴散（constrained/guided diffusion）。

---

## 支柱四：發散（動力系統 + 從資料還原方程）

對應對話段落：「抽掉時間、看相空間、找吸引子」「真正的世界模型是加減乘除」「混沌與測不準」。

| # | 論文 | 發想方式 | 原始碼 |
|---|------|----------|--------|
| 1 | Brunton, Proctor & Kutz 2016 — *SINDy: Discovering governing equations from data* (PNAS 113:3932) | 假設「真正的動力學方程在一個函數庫裡是**稀疏**的」，用稀疏回歸把加減乘除／平方項挑出來 → 從資料直接還原微分方程骨架。 | ✅ `dynamicslab/pysindy` |
| 2 | Lusch, Kutz & Brunton 2018 — *Deep learning for universal linear embeddings of nonlinear dynamics* (Nat. Commun. 9:4950) | 用自編碼器找一組座標，讓非線性動力學在該座標下**變成線性**（Koopman）→ 你說的「抽離時間、相空間」的正式版。 | ✅ `BethanyL/DeepKoopman` |
| 3 | Li et al. 2020 — *Scalable Gradients for Stochastic Differential Equations* (Latent SDE, arXiv:2001.01328) | 把 Neural ODE 推廣到**隨機**：同時學 drift（漂移）與 diffusion（擴散），伴隨法反傳 → 神經 SDE。 | ✅ `google-research/torchsde` |
| 4 | Champion et al. 2019 — *Data-driven discovery of coordinates and governing equations* (PNAS 116:22445; arXiv:1904.02107) | 同時學「座標」（autoencoder）與「該座標下的稀疏動力學」（SINDy）→ 讓機器自己找到最簡潔的變數與方程。 | ✅ `kpchamp/SindyAutoencoders` |
| 5 | Lorenz 1963 — *Deterministic Nonperiodic Flow* (J. Atmos. Sci. 20:130) | 把對流方程截斷成**三條 ODE**，發現對初始條件的敏感依賴 → 混沌與蝴蝶效應的起源，一切「發散」的原型。 | 三行 ODE，任何 solver 可復現 |

**這支柱的統一發想**：與其硬記時間序列，不如問「產生這串資料的**方程**是什麼」。SINDy/Koopman 把「找模式」升級成「找律」。

---

## 支柱五：大集合 / 高維 / 非參數（無限膨脹 + 剪枝）

對應對話段落：「無限膨脹模型實驗」「維度隨驚訝度增生、無效維度萎縮」「維度詛咒」。

| # | 論文 | 發想方式 | 原始碼 |
|---|------|----------|--------|
| 1 | Griffiths & Ghahramani 2011 — *The Indian Buffet Process* (JMLR) | 定義一個「**無限多個**潛在特徵」的先驗，但只觀測到有限子集 → 讓模型維度可以自發增長。這就是「無限膨脹」的數學核心。 | ✅ `RobRomijnders/indian_buffet`、`echen/dirichlet-process` |
| 2 | Rasmussen 2000 — *The Infinite Gaussian Mixture Model* (NIPS) | 讓混合成分數趨於無窮（Dirichlet process），資料自己決定要用幾群 → 不用事先設定群數。 | ✅ `echen/dirichlet-process`（教學實作） |
| 3 | Blei, Ng & Jordan 2003 — *Latent Dirichlet Allocation* (JMLR) | 為「大集合文件」設計一個生成式混合先驗 → 大規模潛在結構發現的範式。 | ✅ `blei-lab/lda-c` |
| 4 | Fefferman, Mitter & Narayanan 2016 — *Testing the Manifold Hypothesis* (JAMS) | 正式檢驗「高維資料其實集中在低維流形」→ 解釋為何 NN 的降維/摺疊有效。 | 理論 |
| 5 | McInnes et al. 2018 — *UMAP* (arXiv:1802.03426) / van der Maaten & Hinton 2008 — *t-SNE* | 保留局部鄰居的（模糊）拓撲結構，把高維攤到低維可視化 → 「看清高維集合的形狀」。 | ✅ `lmcinnes/umap`、`lvdmaaten/bhtsne` |

**這支柱的統一發想**：不預設容量。要嘛承認資料在低維流形上（可壓縮），要嘛給模型一個「可以長到無限、但實際只用有限」的先驗（可增生+剪枝）。

---

## 跨支柱的「發想方法論」總結（你要學的元技能）

看完 25 篇，這些研究的「發想辦法」其實只有幾種可複用的心智動作：

1. **問結構、不問功能**：「這東西在幾何/代數上到底在做什麼？」（UAT、Transformer、Koopman）
2. **把貴的操作換成等價的便宜操作**：attention→線性遞迴（Mamba/RWKV）、精確積分→優化（VI）。
3. **把限制寫進損失，而非寫進資料**：PINN、HNN、引導擴散——物理牆的可微化。
4. **把「離散堆疊」看成「連續過程」**：ResNet→ODE→SDE（Neural ODE/SDE、擴散）。
5. **找律，不找模式**：SINDy/Champion——假設方程稀疏，用回歸挖出加減乘除。
6. **不預設容量**：非參數貝氏（IBP/DP）——先驗給無限、實際用有限。
7. **統一看似無關的東西**：Song 把 DDPM、score matching、SDE 統一成一個框架——這是最高階的發想。

> 你在對話裡自己走過了 1、3、4、5、6。你缺的不是想法，是**這些想法已有的名字與最短落地路徑**——那就是下一份文件 `projects.md`。
