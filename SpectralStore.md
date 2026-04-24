# SpectralStore: 基于鲁棒非对称张量谱分解的时序图近似查询处理系统

> **面向 ICDE 的研究计划文档**  
> 本文档为自包含研究规划，包含方法、数据集、Baseline 和完整实验方案，可供 AI 工具直接读取并自主推进。

---

## 目录

1. [研究背景与动机](#一研究背景与动机)
2. [方法概述](#二方法概述)
   - [问题定义](#21-问题定义)
   - [系统架构](#22-系统架构)
   - [核心算法：鲁棒非对称张量谱分解](#23-核心算法鲁棒非对称张量谱分解)
   - [查询处理与误差保证](#24-查询处理与误差保证)
3. [数据集](#三数据集)
4. [Baseline 方法](#四-baseline-方法)
5. [实验计划](#五实验计划)
6. [实现建议](#六实现建议)

---

## 一、研究背景与动机

大规模时序图（如社交网络、通信网络、合作网络）在工业界持续增长，带来存储和查询两方面的压力。传统做法是存储完整邻接结构，在其上执行精确查询。但在很多场景下，用户可以容忍一定的近似误差来换取数量级的存储压缩和查询加速——这正是**近似查询处理（Approximate Query Processing, AQP）**的核心思想。

在关系数据上，AQP 已经有成熟的工作（BlinkDB、样本 Synopsis、Sketch 等）。但面向图数据的 AQP 系统远不成熟，核心瓶颈在于：图的拓扑结构使得简单的随机采样无法提供有意义的误差保证（一条被采样掉的边可能改变整个连通分量的结构）。

**核心洞察：** 低秩谱分解可以作为时序图的一种物理压缩存储格式，使得一类重要的图查询可以直接在压缩域上回答。更进一步，最近在非对称随机矩阵理论方面的突破（Chen et al., *Annals of Statistics*; Agterberg et al. 等关于 entrywise eigenvector analysis 的工作）揭示了一个被忽视的机会——当我们拥有同一底层图结构的多个独立观测时（时序图的不同快照天然就是这种情形），**不应该对称化处理数据，而应该利用非对称结构来获得更精确的逐元素误差保证**。

本文将这一理论洞察落地为一个完整的系统，整合：
- 张量分解（处理时序维度）
- 贝叶斯先验（实现自适应压缩）
- 重尾建模（提供对抗鲁棒性）

最终形成一个面向时序图的 AQP 存储与查询引擎。

---

## 二、方法概述

### 2.1 问题定义

输入是一个时序图序列 G_1, G_2, ..., G_T，每个 G_t 共享同一节点集 V（|V| = n），邻接矩阵为 A_t ∈ R^{n×n}。假设存在底层低秩对称矩阵：

    M* = U* Λ* (U*)^T ∈ R^{n×n},  rank = r << n

每个快照 A_t 是对 M* 的一次独立含噪观测：

    A_t(i,j) = M*(i,j) + H_t(i,j)

其中 H_t 的各元素独立但不要求同分布（允许异方差），也不要求 H_t 对称。将所有快照堆叠为三阶张量 A ∈ R^{n×n×T}。

**目标：** 构建压缩表示 C，使得 size(C) << size(A)，并能在 C 上直接高效回答以下查询，同时提供理论误差上界。

**支持的查询类型：**

| 查询 | 签名 | 说明 |
|------|------|------|
| Q1 | `LINK_PROB(u, v, t)` | 节点 u, v 在时刻 t 的连接概率估计及误差上界 |
| Q2 | `TOP_NEIGHBOR(u, t, k)` | 时刻 t 与节点 u 连接概率最高的 k 个节点 |
| Q3 | `COMMUNITY(t)` | 时刻 t 的社区划分 |
| Q4 | `TEMPORAL_TREND(u, v, t1, t2)` | 节点对 (u,v) 在时间窗口 [t1, t2] 的连接概率变化曲线 |
| Q5 | `ANOMALY_DETECT(t, threshold)` | 时刻 t 中偏离底层模型超过 threshold 的边 |

---

### 2.2 系统架构

系统由四层组成：

#### 存储层（Factored Store）

压缩表示采用因子化格式存储：

**核心存储：**
- 左因子矩阵 U ∈ R^{n×r}
- 右因子矩阵 V ∈ R^{n×r}
- 时间模式矩阵 W ∈ R^{T×r}
- 核心权重向量 λ ∈ R^r
- 每个分量的不确定性参数

**辅助存储：** 稀疏残差矩阵 S_t = A_t - Â_t 的压缩稀疏格式（CSR），仅保留残差绝对值超过阈值 τ 的元素。

**总存储量：** O((2n + T)·r + nnz(S))，其中 nnz(S) 由阈值 τ 控制，形成存储精度的 trade-off 旋钮。

#### 压缩引擎（Compression Engine）

负责将原始时序图转化为因子化存储格式。核心算法为鲁棒非对称张量谱分解（详见 2.3 节）。

支持两种模式：
- **批量模式：** 一次性处理所有快照
- **增量模式：** 新快照到达时更新因子矩阵，无需重新处理历史数据

#### 查询处理层（Query Processor）

接收用户查询，判断能否完全在压缩域上回答：
- 若可以，直接在因子矩阵上计算并附带误差上界返回
- 若需要更高精度，选择性地访问稀疏残差存储来修正结果

查询优化器根据用户指定的误差容忍度选择执行路径——与传统 AQP 系统中根据置信度要求选择样本大小的逻辑同构。

#### 索引层（Index Layer）

在因子空间上构建索引结构以加速查询：
- 节点嵌入向量 U_u ∈ R^r 上建立基于乘积量化（PQ）的近似最近邻索引，用于加速 `TOP_NEIGHBOR` 查询
- 时间模式向量 W_t ∈ R^r 上建立一维时间索引（B+ 树变体），用于加速 `TEMPORAL_TREND` 查询
- 社区标签（由嵌入聚类得到）上建立倒排索引，用于加速 `COMMUNITY` 查询

---

### 2.3 核心算法：鲁棒非对称张量谱分解

#### 阶段一：张量展开与非对称化构造

将张量 A ∈ R^{n×n×T} 沿 mode-3 展开为矩阵 A_(3) ∈ R^{n×nT}，其中第 t 个 block 列为 A_t 的列。

同时构造非对称的"拼接矩阵"：选取时间快照的随机二分 {1,...,T} = T1 ∪ T2，令：

    Ā_1 = (1/|T1|) Σ_{t∈T1} A_t
    Ā_2 = (1/|T2|) Σ_{t∈T2} A_t

构造非对称矩阵 Ã：上三角取自 Ā_1，下三角取自 Ā_2，对角线取两者均值。  
由于 Ā_1 和 Ā_2 的上下三角噪声是独立的，这正是非对称特征分析理论适用的设定。

#### 阶段二：贝叶斯自适应秩估计与分解

对 Ã 进行截断 SVD 得到初始分解 Ã ≈ Û Σ̂ V̂^T。

引入自动相关性确定（ARD）先验：

    λ_j ~ Gamma(α_0, β_0),  j = 1, ..., r_max

整体生成模型：

    M* = Σ_{j=1}^{r_max} λ_j u_j v_j^T
    A_t(i,k) ~ p(· | [M*]_{ik}, σ²_{ik})

使用**变分推断（Variational Inference）**近似后验，变分分布为均场近似：

    q({u_j, v_j, λ_j}) = Π_j q(λ_j) Π_j q(u_j) Π_j q(v_j)

通过最大化 ELBO（Evidence Lower Bound）优化。ARD 先验的效果是：不重要分量的 E[λ_j] 自动推向零，有效秩自然浮现。

以阶段一的 SVD 结果作为 warm start：用 σ̂_j 初始化 E[λ_j]，用 û_j, v̂_j 初始化因子矩阵的变分参数。

#### 阶段三：鲁棒化——稀疏异常值分离

建模为：

    A_t = M* + S_t + H_t

其中 S_t 是稀疏的异常矩阵，H_t 是密集的随机噪声。

采用**交替优化**策略：
1. 在当前 M̂ 估计下，通过硬阈值算子估计 Ŝ_t——将残差 A_t - M̂ 中绝对值超过阈值 τ 的元素保留，其余置零
2. 在 A_t - Ŝ_t 上重新运行阶段一和阶段二的谱分解
3. 交替迭代直至收敛

阈值 τ 的选择基于 entrywise bound 理论：若 |A_t(i,j) - M̂(i,j)| 远大于理论预测的噪声标准差（可从变分后验中获得），则判定为异常。

将分离出的 Ŝ_t 以 CSR 格式存储在辅助存储中，同时为 `ANOMALY_DETECT` 查询服务。

#### 阶段四（在线阶段）：增量更新

当新快照 A_{T+1} 到达时：
1. 将 A_{T+1} 投影到当前左因子空间，得到新时间模式向量 w_{T+1} = U† A_{T+1} V / n（其中 U† 是伪逆）
2. 将 w_{T+1} 追加到 W 矩阵
3. 计算残差 R = A_{T+1} - U diag(w_{T+1}) V^T
   - 若残差范数超过阈值（底层结构发生显著变化）→ 触发完整重新分解
   - 否则 → 只做因子矩阵的秩一修正

> 类比数据库中物化视图的增量维护：小幅变更时增量更新，累积偏差过大时全量刷新。

---

### 2.4 查询处理与误差保证

#### Q1: `LINK_PROB(u, v, t)`

压缩域计算：

    p̂ = Σ_{j=1}^r λ_j U_{u,j} V_{v,j} W_{t,j}

误差上界（entrywise bound）：

    |M*(u,v) - p̂| ≤ C · σ_max √(r log n) / √(nT) · (1/√μ_u + 1/√μ_v)

其中 μ_u, μ_v 为节点的度相关量，σ_max 为噪声的最大标准差，C 为与模型参数相关的常数。若用户要求精度高于此 bound，查询处理器访问残差存储 S_t(u,v) 进行修正。

#### Q2: `TOP_NEIGHBOR(u, t, k)`

计算节点 u 在时刻 t 的"查询向量"：

    q_u = diag(W_t) · diag(λ) · U_u ∈ R^r

在 V 的行向量中找与 q_u 内积最大的 k 个，即**最大内积搜索（MIPS）**问题，利用索引层的 PQ 索引可在亚线性时间内完成。

#### Q3: `COMMUNITY(t)`

对加权嵌入 Ũ_u = diag(√λ · W_t) · U_u 运行 k-means 聚类（或直接用预计算的聚类结果通过倒排索引查找）。

#### Q5: `ANOMALY_DETECT(t, threshold)`

直接查询稀疏残差存储 S_t，返回所有 |S_t(i,j)| > threshold 的边。如果残差存储按值排序（或建有阈值索引），此查询可在 O(output size) 时间内完成。

---

## 三、数据集

### 3.1 合成数据

#### Synthetic-SBM

基于时序随机块模型生成。底层社区结构矩阵 M* = U* Λ* (U*)^T，U* 由 K 个社区的指示向量构成。

参数范围：
- n ∈ {500, 1000, 2000, 5000, 10000}
- T ∈ {5, 10, 20, 50, 100}
- K ∈ {3, 5, 10}
- 社区内连接概率 p = 0.3，社区间连接概率 q = 0.05

用途：验证理论 bound 的 tightness。

#### Synthetic-Spiked

Spiked 矩阵模型的张量版本。M* = Σ_{j=1}^r λ_j u_j u_j^T，u_j 从单位球上均匀采样，λ_j 按指数衰减。A_t = M* + σ_t E_t，E_t 各元素 iid 标准正态。

参数范围：
- n ∈ {200, 500, 1000, 5000}
- r ∈ {3, 5, 10}
- SNR = λ_1 / σ ∈ {0.5, 1, 2, 5}

#### Synthetic-Attack

在 Synthetic-SBM 基础上注入对抗性扰动，三种攻击模式：
- **Random：** 随机翻转 ε 比例的边，ε ∈ {0.05, 0.10, 0.15, 0.20, 0.30}
- **Targeted：** 选择跨社区的节点对优先添加边，旨在模糊社区边界
- **Injection：** 注入 δn 个虚假节点，δ ∈ {0.01, 0.05, 0.10}，每个虚假节点随机连接到某社区的节点

---

### 3.2 真实数据：中小规模

| 数据集 | 节点数 | 边数 | 快照数 | 下载地址 |
|--------|--------|------|--------|----------|
| Bitcoin-OTC | 5,881 | 35,592 | ~28（按月） | https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html |
| Bitcoin-Alpha | 3,783 | 24,186 | — | https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html |
| UCI Messages | 1,899 | 59,835 | ~20（按周） | SNAP/KONECT |
| Enron Email | ~150 核心员工 | ~500K 邮件 | ~42（按月） | — |

- **Bitcoin-OTC/Alpha：** 带时间戳的有向加权边（权重范围 -10 到 +10），适合精度验证和鲁棒性实验，可做交叉验证
- **UCI Messages：** 规模小，适合快速迭代和可视化调试
- **Enron Email：** 时间跨度约 3.5 年，适合 temporal query 的 case study

---

### 3.3 真实数据：中大规模

| 数据集 | 节点数 | 边数 | 快照数 | 下载地址 |
|--------|--------|------|--------|----------|
| ogbl-collab (OGB) | 235,868 | 1,285,465 | 55（1963–2017，按年） | https://ogb.stanford.edu/docs/linkprop/#ogbl-collab |
| Reddit Hyperlink | ~55,863 | ~858,490 | 按月 | https://snap.stanford.edu/data/soc-RedditHyperlinks.html |
| DBLP Co-authorship | ~317K | ~1M | ~50（按年） | SNAP/AMiner |
| Stack Overflow | ~260万 | ~63M | — | https://snap.stanford.edu/data/sx-stackoverflow.html |

- **ogbl-collab：** OGB 提供标准化链接预测评测协议（MRR 指标）
- **Reddit Hyperlink：** 社区结构明显，适合社区检测实验
- **Stack Overflow：** 需要采样子图使用，适合可扩展性实验（10K 到 500K 节点）

---

### 3.4 真实数据：知识图谱（可选）

| 数据集 | 实体数 | 关系数 | 三元组数 | 备注 |
|--------|--------|--------|----------|------|
| FB15k-237 | 14,541 | 237 | 310,116 | 选取对称关系子集实验 |
| ICEWS | ~7K | ~230 | ~460K | 含时间戳，可构造四阶张量或选取某类关系子集 |

---

## 四、Baseline 方法

### 4.1 谱方法 / 矩阵分解类（核心对比组）

| 方法 | 描述 | 实现 |
|------|------|------|
| **SymSVD** | 先均值 Ā = (1/T)Σ A_t，再对称化 (Ā + Ā^T)/2，最后截断 SVD。代表"先对称化再分解"的传统策略。 | numpy/scipy `svds` |
| **DirectSVD** | 对 Ā 直接截断 SVD，不做对称化处理 | numpy/scipy `svds` |
| **NMF** | 非负矩阵分解，针对元素非负的邻接矩阵 | sklearn `NMF` |

### 4.2 张量分解类

| 方法 | 描述 | 实现 |
|------|------|------|
| **CP-ALS** | 经典 CP 分解，交替最小二乘求解 | TensorLy `parafac` |
| **Tucker-ALS** | Tucker 分解，核心张量非对角，比 CP 更灵活 | TensorLy `tucker` |
| **BPTF** | Xiong et al. (2010) 贝叶斯概率张量分解，MCMC 推断，自动推断秩 | GitHub `bptf` 或自行实现 Gibbs 采样 |
| **COSTCO (VLDB 2019)** | 面向稀疏张量的可扩展 CP 分解，SGD 优化，代表 DB 社区方法 | — |

### 4.3 动态图嵌入类

| 方法 | 描述 | 实现 |
|------|------|------|
| **DynGEM** | Goyal et al. 动态图自编码器，增量更新嵌入 | 作者开源代码 |
| **EvolveGCN (AAAI 2020)** | 使用 RNN 演化 GCN 参数 | PyTorch Geometric 或作者开源代码 |
| **ROLAND (KDD 2022)** | 动态图学习框架，OGB benchmark 上表现优异 | 作者开源代码 |
| **TGN** | Rossi et al. 时序图网络 | PyTorch Geometric Temporal |

### 4.4 图压缩 / 摘要类（体现 DB 视角的关键对比组）

| 方法 | 描述 | 实现 |
|------|------|------|
| **SWeG (SIGMOD 2019)** | Shin et al. 可扩展图摘要，通过合并相似节点压缩图 | 作者开源代码 |
| **SSumM (SIGMOD 2020)** | Lee et al. 图摘要，优化压缩比和重构误差的 trade-off | 作者开源代码 |
| **Spectral Sparsification** | Spielman & Srivastava (2011)，通过有效电阻采样保留 O(n log n / ε²) 条边 | 自行实现或 GraphBLAS |
| **TCM** | 面向时序图的压缩方法，将相似快照合并 | — |

### 4.5 鲁棒图学习类

| 方法 | 描述 | 实现 |
|------|------|------|
| **RPCA+SVD** | 先用 Robust PCA（主成分追踪，Candès et al. 2011）分离低秩+稀疏，再对低秩部分做 SVD | ADMM 求解的开源 RPCA 实现 |
| **Pro-GNN (KDD 2020)** | Jin et al. 鲁棒图神经网络，联合优化图结构和 GNN 参数 | 作者开源代码 |
| **GCN-Jaccard** | Wu et al. (2019)，基于 Jaccard 相似度预处理去除可疑边后再做 GCN | — |

---

## 五、实验计划

### 实验一：理论验证——entrywise bound 的 tightness

**目标：** 验证非对称张量谱分解的 entrywise 误差是否达到理论预测的量级，以及相对于对称化方法的优势在实验中是否可观测。

**数据集：** Synthetic-SBM、Synthetic-Spiked

**实验设计：** 固定秩 r = 5，在 Synthetic-SBM 上分两组实验：
- 第一组：固定 T = 20，让 n 从 500 变化到 10000，记录各方法在 50 次独立重复下的平均最大 entrywise 误差 max_{i,j} |M*(i,j) - M̂(i,j)|
- 第二组：固定 n = 2000，让 T 从 5 变化到 100

在 Synthetic-Spiked 上做同样的两组实验，但额外变化 SNR。

**对比方法：** SpectralStore、SymSVD、DirectSVD、Tensor Unfolding + SVD、CP-ALS

**评测指标：** 最大 entrywise 误差、平均 entrywise 误差、Frobenius 误差。绘制 log-log 图并拟合幂律指数，与理论预测的 O(sqrt(log n / (nT))) 对比。

**预期结果：**
- SpectralStore 的 entrywise 误差最小，衰减速率与理论吻合
- SymSVD 由于丢弃了非对称噪声结构的信息，误差始终更大
- T 增大时 SpectralStore 的改善幅度大于 SymSVD

---

### 实验二：压缩精度与存储效率

**目标：** 在不同压缩比下比较各方法的图重构精度和下游查询精度。

**数据集：** Bitcoin-OTC、ogbl-collab、Reddit Hyperlink

**实验设计：** 定义压缩比 ρ = size(C) / size(A)。通过改变秩 r 和残差阈值 τ 调节压缩比，让 ρ 从 1% 变化到 50%。

**评测指标：**

重构指标：
- 相对 Frobenius 误差 ||A - Â||_F / ||A||_F
- 最大 entrywise 误差 ||A - Â||_∞

下游任务指标：
- ogbl-collab 上链接预测：MRR 和 Hits@50（OGB 标准协议）
- Bitcoin-OTC 上信任评分预测：RMSE 和 MAE
- Reddit 上社区检测：NMI 和 ARI

系统指标：
- 压缩时间
- 压缩后存储字节数
- `LINK_PROB` 平均响应时间

**对比方法：** SpectralStore、SymSVD、CP-ALS、Tucker-ALS、BPTF、SWeG、SSumM、Spectral Sparsification

**预期结果：** 在低压缩比（1%–10%）区间，SpectralStore 的 entrywise 误差显著优于其他方法；相对于 SWeG 和 SSumM 等图摘要方法，优势在于提供理论误差保证。

---

### 实验三：查询性能

**目标：** 验证在压缩域上执行查询相比在原始图上执行的加速效果。

**数据集：** ogbl-collab、Reddit Hyperlink、Stack Overflow（子采样到 100K 节点）

**实验设计：** 预先将图压缩到 ρ = 10%。生成 1000 条随机查询，均匀覆盖五种查询类型（Q1–Q5）。

执行模式对比：
- **Raw 模式：** 在原始邻接矩阵上执行查询（scipy 稀疏矩阵或 NetworkX）
- **SpectralStore：** 在因子化表示上执行查询，配合 PQ 索引
- **SWeG：** 在图摘要上执行查询
- **Neo4j baseline：** 将图导入 Neo4j 图数据库，使用 Cypher 查询语言执行等价查询（代表工业级图数据库系统）

**评测指标：** 平均查询延迟、p95 延迟、吞吐量、查询结果精度（与 Raw 模式对比的一致率）

**预期结果：**
- `LINK_PROB`：SpectralStore O(r) 时间，在冷缓存下比 Raw 模式快数个数量级
- `TOP_NEIGHBOR`：SpectralStore 借助 PQ 索引无需遍历整行
- `COMMUNITY`：SpectralStore 通过预计算聚类和倒排索引几乎 O(1) 返回

---

### 实验四：鲁棒性实验

**目标：** 验证在各种对抗性扰动下系统的退化行为。

**数据集：** Synthetic-Attack（精确控制扰动）、Bitcoin-Alpha 和 Bitcoin-OTC（模拟攻击）

**实验设计：**
- Synthetic-Attack 上进行三组实验（Random、Targeted、Injection），扰动比例从 0% 到 30%，每种设定重复 30 次
- 真实数据上使用 Random attack 和 Targeted attack（Metattack 简化版——随机选择跨社区节点对添加边），扰动比例从 0% 到 20%

**评测指标：**
- 社区检测 NMI 随扰动比例的变化曲线
- 节点嵌入 entrywise 误差随扰动比例的变化曲线
- 链接预测 AUC 随扰动比例的变化曲线

**对比方法：** SpectralStore（含鲁棒化）、SpectralStore-NoRobust（消融对比）、SymSVD、RPCA+SVD、Pro-GNN、GCN-Jaccard、BPTF

**预期结果：** SpectralStore 在所有攻击模式下展现最平缓的退化曲线。Targeted attack 对对称谱方法的攻击面不适用于 SpectralStore 的非对称特征分解。

---

### 实验五：贝叶斯秩选择的效果

**目标：** 验证 ARD 先验自动选秩的准确性和效率。

**数据集：** Synthetic-SBM（真实秩已知）、ogbl-collab（用交叉验证作为参考）

**实验设计：**
- Synthetic-SBM 上：设定真实秩 r* ∈ {3, 5, 10}，给所有方法设置 r_max = 30。记录贝叶斯方法估计的有效秩（定义为 E[λ_j] > 0.01 · E[λ_1] 的分量数）
- ogbl-collab 上：5 折交叉验证遍历 r ∈ {5, 10, 15, 20, 30, 50} 选出最优秩作为 reference，与贝叶斯方法自动选择的秩对比

**对比方法：** SpectralStore（ARD 先验）、CP-ALS + 交叉验证、Tucker-ALS + 交叉验证、BPTF、信息准则方法（BIC/AIC 应用于 SVD 奇异值）

**评测指标：**
- 秩估计的准确性（合成数据上与真实秩的差距）
- 下游任务精度（用估计的秩做压缩后的链接预测 MRR）
- 秩选择的总计算时间（ARD 自动选秩 vs 网格搜索交叉验证的 wall-clock time）

**预期结果：** 关键优势在于计算时间：ARD 只需一次变分推断（因为有 SVD warm start，通常收敛很快），而交叉验证需要对每个候选秩训练一次模型乘以 fold 数。

---

### 实验六：可扩展性

**目标：** 展示系统在大规模数据上的可行性。

**数据集：** Synthetic-SBM（n 从 1K 到 1M）、Stack Overflow 子图（10K 到 500K 节点）

**实验设计：** 固定 T = 10, r = 10。测量压缩阶段的 wall-clock 时间和峰值内存消耗。

SpectralStore 核心计算为截断 SVD（使用 `scipy.sparse.linalg.svds` 或 randomized SVD），复杂度为 O(nnz · r · n_iter)。

**对比方法：** SpectralStore、CP-ALS（TensorLy）、BPTF（MCMC 采样）、EvolveGCN（GPU 训练）、TGN（GPU 训练）、SWeG

**评测指标：** wall-clock 时间（秒）、峰值内存（GB）；如可能，分别测量 CPU-only 和 GPU-accelerated（对深度学习方法）的性能。

**预期结果：** SpectralStore 因为核心是稀疏截断 SVD，应在 CPU 上就能处理百万级节点的图，时间在分钟级别。BPTF 依赖 MCMC 采样，可能慢一到两个数量级。

---

### 实验七：消融实验

**目标：** 验证系统每个组件的贡献。

**数据集：** Bitcoin-OTC、ogbl-collab

系统变体：

| 变体 | 说明 |
|------|------|
| **SpectralStore-Full** | 完整系统 |
| **SpectralStore-NoAsym** | 去掉非对称化构造，改为对均值矩阵做对称 SVD |
| **SpectralStore-NoBayes** | 去掉贝叶斯 ARD 先验，改为手动设定秩 |
| **SpectralStore-NoRobust** | 去掉稀疏异常值分离 |
| **SpectralStore-NoResidual** | 去掉稀疏残差存储，查询只使用因子矩阵 |
| **SpectralStore-NoIndex** | 去掉 PQ 索引，`TOP_NEIGHBOR` 使用暴力扫描 |

**评测指标：**

无攻击场景：压缩精度（Frobenius 误差、entrywise 误差）、下游任务精度（链接预测 MRR、社区检测 NMI）、查询延迟

有攻击场景（Random attack 10%）：同一组指标

**预期结果：** 每个组件都有正向贡献，但贡献大小不同：
- 非对称化和鲁棒化在有攻击时贡献最大
- 贝叶斯秩选择在真实数据上贡献最大（真实秩未知）
- 索引对查询延迟贡献最大

---

## 六、实现建议

### 技术栈

**主要语言：** Python，性能关键路径用 Cython 或 Numba 加速

**核心依赖：**
- `numpy`, `scipy`（稀疏矩阵和截断 SVD）
- `tensorly`（张量分解 baseline）
- `scikit-learn`（NMF、k-means 等）
- `PyTorch Geometric Temporal`（深度学习 baseline）
- `ogb`（数据集加载和评测）

### 代码组织

```
spectralstore/
├── data_loader/      # 统一的数据加载接口，支持所有数据集
├── compression/      # 所有压缩方法的统一接口
├── query_engine/     # 查询处理器
├── index/            # PQ 索引和时间索引
├── evaluation/       # 所有评测指标的计算
└── baselines/        # 所有 baseline 的封装
```

### 实验管理

- 使用 **Hydra** 或 YAML 配置文件管理所有实验参数
- 使用 **MLflow** 或 **Weights & Biases** 记录实验结果
- 固定随机种子、记录环境信息，确保所有实验可复现
