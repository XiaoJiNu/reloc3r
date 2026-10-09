# Reloc3r 绝对位姿原理、Pipeline 与量产落地指南

日期：2026-09-23；2026-09-29 更新：公式与符号改用 LaTeX 书写并补充符号含义说明（内容、代码行号引用、实测数字均未改动）。基于本仓库代码逐行核对（关键文件与行号均在文中标出），配套实测复现见同目录《Reloc3r-MegaDepth1500推理复现_20260923_103737.md》。

## 〇、符号表（全文统一，公式中出现的每个符号都在此定义）

| 符号                                                                                                                                                                                                                                                                                | 含义                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| $w$                                                                                                                                                                                                                                                                               | 世界系（world），即建图/采集时定下的米制参考系                                                               |
| $d,\ d_i$                                                                                                                        | 数据库（db）第$i$ 帧相机系；$i=1,\dots,K$，评测里 $K=10$（top-10 候选）                                                                  |                                                                                                              |
| $q$                                                                                                                                                                                                                                                                               | 查询（query）相机系                                                                                          |
| $c$                                                                                                                              | 相机系通用记号（区别于世界系$w$）                                                                                                            |                                                                                                              |
| $T^{w}_{c} \in SE(3)$                                                                                                            | 相机系$c$ 到世界系 $w$ 的 4×4 位姿矩阵（camera-to-world, c2w），分块写为 $\begin{bmatrix} R^{w}_{c} & t^{w}_{c} \\ 0 & 1 \end{bmatrix}$ |                                                                                                              |
| $R^{w}_{c} \in SO(3)$                                                                                                                                                                                                                                                             | 上式的 3×3 旋转块：相机轴到世界轴的旋转                                                                     |
| $t^{w}_{c} \in \mathbb{R}^3$                                                                                                                                                                                                                                                      | 上式的平移块：相机光心在世界系的坐标（米，当且仅当 db 位姿是米制）                                           |
| $T_{d \leftarrow q}$                                                                                                                                                                                                                                                              | 模型回归的相对位姿（4×4）：把 query 相机系中的点变到 db 相机系（"from 2 to 1"，`reloc3r_relpose.py:161`） |
| $R_{d \leftarrow q},\ t_{d \leftarrow q}$ | 其旋转块与平移块；$t_{d \leftarrow q}$ 的几何意义 = query 光心在 db 相机系下的坐标                                                                                                                                                  |                                                                                                              |
| $\hat{(\cdot)}$ | 单位向量：$\hat d = d/\lVert d \rVert_2$；$\lVert\cdot\rVert_2$ 为欧氏范数                                                                                                                                                                                  |                                                                                                              |
| $I$                                                                                                                                                                                                                                                                               | 3×3 单位阵                                                                                                  |
| $x \in \mathbb{R}^3$                                                                                                                                                                                                                                                              | 待求的 query 光心世界坐标（LS 交点方程的未知量）                                                             |
| $B$                                                                                                                              | 基线：db 帧与 query 帧光心的真实距离$\lVert t^{w}_{d} - t^{w}_{q} \rVert_2$                                                                  |                                                                                                              |
| $\delta$                                                                                                                                                                                                                                                                          | 单对回归的平移方向角误差（rad 或 °）                                                                        |
| $q_i \in \mathbb{R}^4$                                                                                                           | 第$i$ 个候选旋转对应的单位四元数                                                                                                             |                                                                                                              |
| $\mathrm{tr}(\cdot)$                                                                                                                                                                                                                                                              | 矩阵的迹                                                                                                     |
| $p_{beg},\ p_{end}$                                                                                                                                                                                                                                                               | 运动平均中每条视线段的两端点（db 光心、由回归平移推出的 query 光心候选）                                     |
| $A,\ b$                                                                                                                          | 堆叠后的最小二乘系统$Ax=b$ 的矩阵与右端项                                                                                                    |                                                                                                              |
| $S,\ U,\ V$                                                                                                                                                                                                                                                                       | $A$ 的 SVD 分解 $A = U S V^{\top}$ 中的奇异值阵与左右奇异向量阵                                          |

约定：上标"到哪个系"、下标"从哪个系"，故 $T^{w}_{d}\,T_{d \leftarrow q}$ 的下标 $d$ 恰好对消得到 $T^{w}_{q}$（c2w 语义下矩阵乘法左乘 = 坐标系逐级回推）。

## 一、核心思想：为什么"回归相对位姿 + 运动平均"能得到绝对位姿

Reloc3r 不做传统的特征匹配 + PnP，也不做端到端绝对位姿回归（那样泛化差、且需要每场景训练）。它拆成两步：

1. **成对相对位姿回归网络**（Reloc3rRelpose）：输入任意两张图 $(d, q)$，直接回归 4×4 相对位姿 $T_{d \leftarrow q}$（query 相对 db 的相机姿态）。单对回归的**平移只有方向、没有米制尺度**（论文用方向角误差评价平移，Table 11 口径；尺度由"db 之间距离归一为 1 米"的约定锚定）。
2. **运动平均（motion averaging）**：一个 query 对 $K$ 个已知道位的 db 各回归一次，得 $K$ 个绝对位姿候选

   $$
   T^{w}_{q,i} = T^{w}_{d_i}\; T_{d_i \leftarrow q}, \qquad i = 1,\dots,K
   $$

   （$T^{w}_{d_i}$ = 第 $i$ 个 db 帧的已知 c2w 位姿；$T_{d_i\leftarrow q}$ = 网络对 $(d_i, q)$ 输出的相对位姿），再用 Robust 平均合并成一个绝对位姿 $T^{w}_{q}$。**$K$ 个已知位姿的参考系就是世界系 $w$**，所以"相对位姿 × db 绝对位姿"天然落在世界系里——这就是绝对位姿的来源，不需要任何 PnP 或 3D 点。

尺度说明（重要，量产时最容易踩）：数据库 db 位姿自带米制尺度（来自建图/采集时的 SLAM 或 GT），平移尺度继承自 db。模型输出的单对平移 $t_{d\leftarrow q}$ 本身是无尺度向量，但经过 $T^{w}_{d}\, T_{d\leftarrow q}$ 合成后由 db 之间的距离定标。README 的 wild_visloc 演示里就显式做了这件事：`pose2to1[0:3,3] /= norm`——把首尾帧平移按 $\hat{t} = t/\lVert t \rVert_2$ 归一为 1 米（见 wild_visloc.py:44）。若你的 db 位姿是米制，query 结果即为米制绝对位姿。

疑问：再用 Robust 平均合并成一个绝对位姿 $T^{w}_{q}$。这里具体是怎么做的？

疑问：模型输出的单对平移 $t_{d\leftarrow q}$ 本身是无尺度向量，但经过 $T^{w}_{d}\, T_{d\leftarrow q}$ 合成后由 db 之间的距离定标。这里我不明白，如果预测的相对位姿是没有尺度的，那计算出来的绝对位姿候选在真实世界坐标系中还准确吗？

## 二、Pipeline 全链路（对照 eval_visloc.py / wild_visloc.py 两条实现路径）

### 路径 A：标准视觉定位评测（eval_visloc.py，7Scenes/Cambridge）

```
[离线建库]
  db 图片 + 每张的 4×4 c2w 位姿 T^w_c (世界系, 米制)
      → NetVLAD(VGG16-Pitts30K) 提取全局描述子 (topk_retrieval.py)
      → 存 ./_db-q_pair_info/<scene>_db-descs_step=1.txt

[在线定位, 每张 query]
  1. 检索:   query 全局描述子 → top-K=10 db (余弦相似)
             → chess_q-retrieval_db-step=1_topk=10.npy
  2. 回归:   对 pair_id=0..9 共 10 轮, 每轮把 (d_i, q) 成对喂给
             Reloc3rRelpose-512 → T_{d_i←q}；同时落盘 T^w_{d_i}
             (eval_visloc.py:110-127, 输出 _pose-q2d/_pose-db/_pose-gt.txt)
  3. 平均:   Reloc3rVisloc.motion_averaging(poses_db, poses_q2d)
             (reloc3r/reloc3r_visloc.py:79-103)
             a) 候选位姿:  T^w_{q,i} = T^w_{d_i} T_{d_i←q}   (K=10 个)
             b) 旋转平均:  R^w_{q,i} → 四元数 q_i → 逐分量 median → 归一化 → SO(3)
                (rotation_averaging, reloc3r_visloc.py:11-27; median 抗离群)
             c) 平移求解:  每个候选贡献一条空间直线 L_i = { p_i + s·d̂_i : s∈R }
                  p_i = db 光心 T^w_{d_i} 的平移块, d̂_i = 射线单位方向
                  K 条直线做最小二乘交点 (camera_center_triangulation,
                reloc3r_visloc.py:29-48: 残差投影矩阵 (I − d̂d̂ᵀ) 堆叠后 SVD 解)
             d) 输出 T^w_q = [ R̄ | x* ]  (median 旋转 + LS 交点平移)
  4. 评价:   与 GT 比（见下方误差公式）, 报中位数
```

误差口径（第 4 步，$R^{gt}, t^{gt}$ = GT 旋转/平移，$\Delta R = \bar{R}^{\top} R^{gt}$）：

- 旋转误差（Rodrigues 角）：$\Delta\theta = \arccos\!\Big(\dfrac{\mathrm{tr}(\Delta R) - 1}{2}\Big) \in [0°, 180°]$——两旋转差的"绕某轴转多少度"。
- 平移误差：$\Delta t = \lVert x^{*} - t^{gt} \rVert_2$（世界系光心欧氏距离，米）。

#### 细化：第 2 步"回归"到底产出什么

**方向约定**（最容易搞反的地方）。模型输出 `pose_q2d` $= T_{d \leftarrow q}$（`reloc3r_relpose.py:161` 注释 "relative camera pose from 2 to 1"）：把 query 相机系中的点变到 db 相机系：

$$
p_d = R_{d\leftarrow q}\, p_q + t_{d\leftarrow q}
$$

（$p_q$ = 某 3D 点在 query 相机系坐标，$p_d$ = 同一 db 系坐标）。按 c2w 语义读，**$t_{d\leftarrow q}$ 就是 query 光心在 db 相机坐标系下的坐标**。因此它能与 db 的绝对 c2w 直接左乘合成 query 的绝对位姿：$T^{w}_{q} = T^{w}_{d}\, T_{d\leftarrow q}$。若理解成 $T_{q\leftarrow d}$，合成公式就得换成右乘——接第三方代码时先用一个已知位姿对拍一次方向。

**输入怎么喂**。view1 = $K$ 张 db 图沿 batch 维 cat，view2 = 同一张 query 图复制 $K$ 份（`eval_zc_qwen.py:193-198`、`wild_visloc.py:58-63`），一次 $B=K$ 前向出 $K$ 个相对位姿——在线延迟 ≈ 单对延迟。`eval_visloc.py:98-99` 的评测路径是 pair_id=0..9 分 10 轮跑并落 txt（便于离线复算），量产别照抄这个循环。

**网络内部**一句话概括：两张图过非对称 ViT、双向 cross-attend 后，query 侧 decoder token 图 → PoseHead → `fc_t` 出平移 $t \in \mathbb{R}^3$、`fc_rot` 出旋转表示 $r \in \mathbb{R}^9$（reshape 3×3 后经 SVD 正交化 $R = U \, \mathrm{diag}(1,1,\det(UV^{\top})) \, V^{\top}$ 投影回 SO(3)）拼成 4×4（`pose_head.py:57-82,106-115`）。**没有特征点、没有 PnP、不输出内参和深度**——"两视图对极几何"是训练时从 8M 图像对里学进 cross-attention 的，不是在线算出来的。

**输出的可信部分 / 不可信部分**：

| 量                                                                                                                                                                                                                                                                 | 可信度 | 原因                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | ---------------------------------------------------------------------- |
| $R_{d\leftarrow q}$（旋转）                                                                                                                                                   | 高     | 旋转与基线长度$B$ 无关，$K$ 个候选给出的是同一真旋转的 $K$ 种近似 |        |                                                                        |
| $t_{d\leftarrow q}$ 的**方向** $\hat{t}$                                                                                                                                                                                                                 | 高     | 方向 = "从 db 光心看向 query 光心"的射线；近基线候选方向角误差 ~1–3° |
| $t_{d\leftarrow q}$ 的**模长** $\lVert t \rVert$ | **不可信，非米制** | 训练对平移按方向评价（论文 AUC 口径），$\lVert t \rVert$ 不携带尺度；尺度必须从别处来                                                                                    |        |                                                                        |

这个"方向可信、尺度不可信"是理解聚合全部设计的前提：**绝对米制位置只从 db 光心的已知位置 $t^{w}_{d}$ 来，模型只提供 $K$ 条射线的方向 $\hat{d}_i$。**

**误差结构**（决定量产参数）。单对方向角误差 $\delta$ 随基线 $B$（db–query 真实距离）增大：视角差越大、共视越少，回归越难。ZC 实测逐对平移误差与基线 corr=0.993，$B$=0–1 m → 0.45 m、10–20 m → 11.8 m（《zc数据集上测试reloc3r效果_qwen3.8-flash》§复核增补）。聚合会把方向误差放大成杠杆臂（见下），所以远基线候选是纯毒药。

#### 细化：第 3 步"平均"为什么旋转用 median、平移用求交

对每个候选 $i$ 先合成 $T^{w}_{q,i} = T^{w}_{d_i}\, T_{d_i\leftarrow q}$（`reloc3r_visloc.py:86`），得 $K=10$ 个 query 绝对位姿候选。关键观察（$R^{w}_{d_i}, t^{w}_{d_i}$ 为 db 位姿的旋转/平移块，$R_i, t_i$ 为该候选回归输出）：

$$
\underbrace{t^{w}_{q,i} = t^{w}_{d_i} + R^{w}_{d_i}\, t_i}_{\text{光心：沿一条已知射线、待定标尺的点}}, \qquad \underbrace{R^{w}_{q,i} = R^{w}_{d_i}\, R_i}_{\text{旋转：与基线长度无关，} K \text{ 个候选都} \approx \text{真旋转}}
$$

即 **$K$ 个候选对旋转互相一致，对平移只一致地给出"方向"、各自给出一个不可信的"距离"**。两种误差分布不同，所以聚合器也不同。

**b) 旋转：逐分量四元数 median**（`reloc3r_visloc.py:19-25`）。$K$ 个绝对旋转 $R^{w}_{q,i}$ 转四元数 $q_i \in \mathbb{R}^4$、对 $x/y/z/w$ 四个分量分别取 median、再归一化 $\bar{q} = \mathrm{med}_i(q_i) / \lVert \mathrm{med}_i(q_i) \rVert_2$ 回 SO(3)。median 对个别跑飞的候选（检索错、给出近 180° 翻转的旋转）免疫，$K$ 票里 9 票对就不影响结果。实测 ZC 逐对 rot p90=2.8°，聚合后 1.4–2.3°——聚合几乎不改善旋转，但几乎不损害。两个已知坑：四元数 $q$ 与 $-q$ 表示同一旋转（双覆盖），`R.from_matrix().as_quat()` 的符号由矩阵决定、近同旋转通常同号，实践中安全，但候选分布在 $q$ 与 $-q$ 混合时逐分量 median 会互相抵消——出现偶发 180° 翻转先查这里（§五.4）；且逐分量 median 不是 SO(3) 上的测地均值（那是极小化 $\sum_i \lVert \log(\bar{R}^{\top} R_i)\rVert^2$ 的迭代解），是近似——候选互相差几度时误差可忽略。

**c) 平移：$K$ 条射线的最小二乘交点**（`camera_center_triangulation`，`reloc3r_visloc.py:35-48`）。每个候选贡献一条世界系直线 $L_i = \{ p_i + s\,\hat{d}_i : s \in \mathbb{R} \}$，其中 $p_i = t^{w}_{d_i}$（db 光心），$\hat{d}_i$ = 射线单位方向。代码里方向由 $p_{end} - p_{beg}$ 给出（$p_{end} = t^{w}_{d_i} + R^{w}_{d_i} t_i$），**$\lVert t_i \rVert$ 只改 $p_{end}$ 在线上的滑位，直线 $L_i$ 本身不变**——所以不可信尺度 $\lVert t_i \rVert$ 在这一步被自动丢弃，这就是"尺度由 db 间距锚定"的确切机制。点 $x$ 到直线 $L_i$ 的垂距用投影矩阵 $(I - \hat{d}_i \hat{d}_i^{\top})$ 表达（它把向量削掉沿 $\hat d_i$ 的分量），令 $x$ 满足 $K$ 条线垂距平方和最小：

$$
\min_x \; \sum_{i=1}^{K} \big\lVert (I - \hat{d}_i \hat{d}_i^{\top})(x - p_i) \big\rVert_2^2
$$

每个候选展开成 3 行线性方程 $(I - \hat{d}_i\hat{d}_i^{\top})\, x = (I - \hat{d}_i\hat{d}_i^{\top})\, p_i$，$K$ 个堆成 $A x = b$（$A \in \mathbb{R}^{3K\times 3}$），SVD 伪逆解（$A = U S V^{\top}$，$x^{*} = V S^{-1} U^{\top} b$）。解是"到 $K$ 条线垂直距离平方和最小"的点。一维直观：真值 $x^*=10$ m；db1@0 m 给射线 $+x$、db2@20 m 给射线 $-x$，交点=10 m ✓；混入 db3@100 m（远基线）且方向偏 $\delta=3°$，它的线在 $x=10$ 附近垂向偏 $\approx B\,\delta \approx 90 \times \sin 3° \approx 4.7$ m，一票等权就把交点拉偏米级——**杠杆臂 $B\,\delta$ 就是 p90 从 0.5 m 炸到 4 m+ 的全部原因**。

三个必知的性质：

1. **LS 等权、非鲁棒**——与旋转 median 恰成对照，一个坏候选就够毁掉平移。量产对策是**先过滤再平均**：粗位姿取 $K$ 个候选平移的 median $\tilde{x} = \mathrm{med}_i\, t^{w}_{q,i}$（纯自洽、不用 GT），只把 db 光心满足 $\lVert t^{w}_{d_i} - \tilde{x} \rVert_2 < $ `base_max` 米的候选喂进 motion_averaging（`eval_zc_qwen.py:206-214`）。实测同天 med/p90 0.58/4.34 → 0.25/1.64 m，跨天 0.46/4.16 → 0.26/2.06 m；阈值 1.5–3 m 都是平台，<1 m 开始丢有效票、p90 回升（扫描脚本 `docs/zc数据集上测试/tools/audit6_sweep_basemax.py`）。
2. **射线近平行时退化**：纯直线运动下所有 $\hat{d}_i$ 同向，$(I - \hat{d}_i\hat{d}_i^{\top})$ 沿射线方向的行全近零，$A$ 秩亏；代码直接 $1/S$ 求伪逆、**没有 rcond 截断**，最小奇异值 $\sigma_{\min}$ 对应的病态分量被 $1/\sigma_{\min}$ 放大 → 解沿轨迹方向漂几米（wild_visloc 声明不支持 linear motion 的数学根源；§五.4 的"最小奇异值占比门控"就是查这个）。db 在侧向有视差布设可解。
3. **模型不给逐候选置信度**，没有现成加权方案可抄；要更讲究就用"过滤 + 候选平移直接取 median"（绕开 LS，对退化也免疫）或 trimmed LS（离线对比框架 `docs/zc数据集上测试/tools/compare_averaging.py`）。

**d) 输出** $T^{w}_{q} = [\, \bar{R} \mid x^{*}\,]$（median 旋转 + LS 交点平移）：旋转和平移各走自己最稳的聚合器。注意基线过滤本质上是为平移存在的——旋转 median 全量喂入也行，只是同批过滤没有坏处。

### 路径 B：自采视频 demo（wild_visloc.py，无检索）

用"首帧+末帧"当 2 帧 db（要求全片环绕重叠，不支持直线运动），中间帧依次定位；`mode='seq'` 时额外把上一帧定位结果加入参考集。db 位姿是现场现算的：首帧 $T^{w}_{d_1} = I$（世界系就取首帧相机系），末帧 $T^{w}_{d_2} = T_{d_1\leftarrow d_2}$ 的回归结果（平移按 $\hat{t}=t/\lVert t\rVert$ 归一 1 m）。这是"从零自定坐标系"的最小用法，也是把它接进你自家系统的模板。

### 网络内部（路径 A/B 共用，reloc3r_relpose.py:19-165）

- 编码器：DUSt3R/CroCo 系不对称 ViT（enc 24 层 1024d + cross-attend dec 12 层 768d），RoPE100 位置编码，ManyAR 任意长宽比 patch embed（patch 16）。
- PoseHead（pose_head.py:32-140）：decoder token 图 → resconv → 全局池化 → MLP → 回归 $r \in \mathbb{R}^9$（SVD 正交化回 SO(3)）+ $t \in \mathbb{R}^3$，拼 4×4。
- 输入要求：RGB 图，resize 到 (512,384)（或 224²），**不需要内参、不需要深度、不需要特征点**——这是它与 hloc/PnP 路线的本质区别。

## 三、数据格式

### 模型 I/O 格式

| 项           | 格式                                                                                                  | 备注                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 单图输入     | dict:`img` (3,H,W float RGB 0~1), `true_shape` (2,) 原图 HxW                                      | load_images 会自动 EXIF 转正、Lanczos 缩放 (reloc3r/utils/image.py:65)                                     |
| 成对输入     | view1={img:B×3×H×W, true_shape}, view2=同上                                                        | $B=K$ 张 db 对 1 张 query 时直接沿 batch 维 cat（wild_visloc.py:58-63 的"一张 query 复制 $K$ 份"技巧） |
| 相对位姿输出 | (B,4,4) float，$T_{d \leftarrow q}$，c2w 约定：$p_w = R\,p_c + t$                                 | 平移无米制尺度（方向为准）                                                                                 |
| 绝对位姿     | 4×4 c2w$T^{w}_{q}$，世界系，尺度=db 尺度                                                           | 7Scenes 的 pose.txt 就是它                                                                                 |
| 检索库       | 图路径 + 4×4 位姿 txt（db 目录结构`<root>/<scene>/seq-XX/frame-XXXXXX.color.png` + `.pose.txt`） | depth.png 完全不用                                                                                         |

### 自建数据的最简格式（对齐 eval_visloc.py 的期望）

```
data/myloc/
  sceneA/seq-01/frame-000000.color.png     # 8bit RGB png/jpg
  sceneA/seq-01/frame-000000.pose.txt      # 4×4 空格分隔, c2w, 米制
  sceneA/seq-02/...                        # train= db, test= query
  ...
```

位姿来源：任何能给你米制 c2w $T^{w}_{d}$ 的东西（采集时 RTK+INS、总后 SLAM 建图、Leaky-Bucket 人工标定均可）。**先跑 wild_visloc.py 模式验证链路，不需要任何检索代码改动。**

## 四、应用到自己场景：数据准备与处理

1. **采集 db（关键约束来自模型本身）**
   - 分辨率 ≥512 宽即可（模型内部会下采样；更大无收益）。EXIF 方向必须是正的。
   - db 覆盖率决定下限：query 必须能与某个 db 图有足够共视（经验上 ≥30% 重叠、视角差 <60°），否则相对位姿回归失败、检索 top-1 也是错的。室内小场景 1~2 m 间距环绕拍；大场景沿路径每 2~5 m 一张。
   - 位姿标定一致性最重要：db 位姿 $T^{w}_{d}$ 如果整体有系统性旋转/尺度偏差，会直接平移进所有 query 结果。建库后先用 wild_visloc 或本仓库 eval 对几个已知位姿的测试点跑一遍，核对中位误差。
   - 动态物体少的时段建库（NetVLAD 是全局描述子，行人/车辆会同时毒化检索和回归）。
2. **不训练也能用**：论文卖点即 zero-shot 泛化（8M 对通用数据预训练）。实测 7Scenes chess 0.03 m / 0.88°、MegaDepth1500 AUC@5/10/20=49.6/67.8/81.2 与论文一致，证明用官方权重直推是成立的。
3. **什么时候才微调**：你的场景域外偏差明显（特殊光照/全景/鱼眼、纹理极少的走廊、纯白场景）且 zero-shot 精度不可用时，用 scripts/train_small.sh 在自有位姿对上微调 512 版；单卡 3090 级即可跑通，数据量建议 ≥5 万对。鱼眼/畸变大的镜头务必先出图级去畸变（模型按针孔训练）。
4. **topk 选择**：topk=$K=10$ 是默认。db 大且重叠稀疏时可到 20；db 很小（如视频 demo 的 2~3 帧）时 median 旋转 + LS 交点仍可用但离群剔除能力下降，建议 db 与 query 的重叠帧数 ≥5。

## 五、实际量产怎么做

1. **部署形态**：模型即一个 (2 图) → 1 相对位姿 的函数（Reloc3rRelpose 512：ViT-L enc + dec，约 0.7B 参数量级）。

   - 实测：RTX 5090 + fp16(amp=1) + PyTorch RoPE 回退 ≈ 64 pairs/s；编译 croco/models/curope CUDA kernel 后论文报 40 FPS（RTX 4090, Reloc3r-512）；Reloc3r-224 更轻。
   - 边缘端：导出 ONNX 可行（网络无动态控制流）；Jetson Orin 级按 224 版 + int8/fp16 评估，先实测再定；512 版建议上独立 GPU 或降帧用（定位 2~5 Hz + 里程计插值，量产机器人/车辆的典型口径）。
2. **系统架构建议（与现有里程计融合，不要裸用）**

   ```
   [建图期] 人工/自主环绕采集 → SLAM(米制 c2w) → 导出 db 图+pose → NetVLAD 描述子入库(一次性)
   [运行期] 每 N 帧: 相机帧 →(a)NetVLAD 检索 top-10 →(b)10×Reloc3r 回归 →(c)motion averaging → 绝对位姿
            绝对位姿 → 与里程计/IMU 做 EKF/位姿图融合 → 输出定位
   ```

   绝对位姿当低频修正量（重定位/回环角色），里程计当高频。单帧 Reloc3r 结果有噪声（本仓库 chess 中位 0.03 m 是室内理想值；室外宽基线时 AUC@5 只有 ~50，意味着近半单帧误差 >5°/方向偏差大），**必须**靠融合+多帧一致性门控（相邻 query 的 $\Delta T$ 与里程计 $\Delta T$ 偏差超阈值则丢弃本次修正）。
3. **检索加速**：NetVLAD 描述子建 IVF/HNSW 索引（faiss），db 十万级时单次检索 <5 ms；描述子离线算好，不占在线预算。
4. **已知坑（逐条都来自本仓库代码实测）**

   - 运动平均的平移是"$K$ 条射线交点"，对**纯直线运动退化**（射线近平行 → $A$ 病态，$\sigma_{\min} \to 0$）；wild_visloc 明确不支持 linear motion 就是这个原因。量产路径若为长直线，需保证 db 布设在侧向有视差，或把退化检测（门控条件 $\sigma_{\min} / \sigma_{\max}$ 低于阈值即弃用本帧）接入门控。
   - 尺度锚点只有 db 帧间距离 $\lVert t^{w}_{d_i} - t^{w}_{d_j} \rVert$。db 稀疏 → query 平移尺度误差大；密集建库是精度最便宜的保险。
   - `rotation_averaging` 用四元数逐分量 median（非流形最优），当候选分布在 $q$ 与 $-q$ 混叠时可能不稳——代码里 R.from_matrix 输出未做符号对齐，实测在 topk=10 室内无问题；若你的场景出现偶发 180° 翻转，先检查这个函数。
   - 单卡 batch 化 $K$ 次成对回归（wild_visloc.py 的 cat 技巧）而非 $K$ 次串行，在线延迟直接除以接近 $K$。
   - fp16(amp=1) 实测与 fp32 结果一致（MegaDepth 复现即 amp=1），量产默认开。
5. **验收指标口径**：内部测试集用本仓库 eval_visloc.py 的中位 (m, °)（误差公式见 §路径 A 第 4 步：$\Delta\theta$ + $\lVert\Delta t\rVert_2$）+ eval_relpose.py 的 AUC@5/10/20（reloc3r/utils/metric.py 定义，$\max(\Delta\theta, \Delta t^{\circ})$ 口径，即平移折算成角度后与旋转取大者）；上线前对每个场景先建 ≥200 帧有 GT 的 test 序列跑通这两套脚本再放量。

## 六、一句话总结

绝对位姿 $T^{w}_{q} = [\bar{R} \mid x^{*}]$ = 已知米制位姿 $T^{w}_{d_i}$ 的 db 图 + 网络回归的"query 相对 db"变换 $T_{d_i\leftarrow q}$，经四元数逐分量 median 的旋转平均 $\bar{R}$ 与 $\min_x \sum_i \lVert (I-\hat d_i\hat d_i^{\top})(x-p_i)\rVert^2$ 的最小二乘射线交点 $x^{*}$ 合并到 db 世界系；部署上它是一个 zero-shot 的"图像对→相对位姿"回归器，量产靠 建库(带位姿)+检索+运动平均+里程计融合 四件套，核心纪律是 db 覆盖率和 db 位姿质量决定一切上限。

## 七、答疑：运动平均的两处高频疑问（2026-09-29 补）

### Q1 "用 Robust 平均合并成一个绝对位姿 $T^{w}_{q}$"具体是怎么做的？

即 `Reloc3rVisloc.motion_averaging`（reloc3r_visloc.py:74-103），对 $K$ 个候选 $(T^{w}_{d_i},\ T_{d_i\leftarrow q})$，$i=1,\dots,K$，执行三步：

1. **逐候选合成**（:86-91）：

   $$
   T^{w}_{q,i} = T^{w}_{d_i}\, T_{d_i\leftarrow q}, \qquad R^{w}_{q,i} = \big[T^{w}_{q,i}\big]_{1:3,1:3}, \qquad \text{直线 } L_i:\ p_i = t^{w}_{d_i},\ \ p^{end}_i = t^{w}_{d_i} + R^{w}_{d_i}\, t_i
   $$

   其中 $t_i$ = 回归平移块；$p_i$、$p^{end}_i$ 只是 $L_i$ 上两点，方向 $\hat d_i = (p^{end}_i - p_i)/\lVert p^{end}_i - p_i \rVert_2$。
2. **旋转支路 = 四元数逐分量 median**（rotation_averaging，:12-26）：

   $$
   q_i = \mathrm{quat}(R^{w}_{q,i}) \in \mathbb{R}^4, \qquad \bar q = \big[\mathrm{median}_i\,(q_i)_x,\ \mathrm{median}_i\,(q_i)_y,\ \mathrm{median}_i\,(q_i)_z,\ \mathrm{median}_i\,(q_i)_w\big], \qquad \bar R = \mathrm{quat}^{-1}\!\big(\bar q/\lVert\bar q\rVert\big)
   $$

   "Robust" 的确切含义在这条支路：median 是 $L_1$ 型估计，$\le \lceil K/2 \rceil - 1$ 个离群旋转（检索错帧给出的 90°/180° 跑飞候选）完全不影响结果——这是它与均值（$L_2$，一票离群整批污染）的本质区别。
3. **平移支路 = $K$ 条直线垂距最小二乘交点**（camera_center_triangulation，:28-49）：

   $$
   x^* = \arg\min_x \sum_{i=1}^{K} \big\lVert (I - \hat d_i \hat d_i^{\top})(x - p_i) \big\rVert_2^2 \quad \Longrightarrow \quad Ax=b
   $$

   **逐项含义**：$x \in \mathbb{R}^3$ 是假定的 query 光心世界坐标（整个支路的唯一未知量）；$x - p_i$ 是从第 $i$ 条直线的锚点 $p_i$（db 光心，米制已知）指向 $x$ 的向量；$(I - \hat d_i \hat d_i^{\top})$ 是**垂向投影矩阵**——对任意向量 $v$，$\hat d_i \hat d_i^{\top} v$ 是 $v$ 沿射线方向 $\hat d_i$ 的分量，$I$ 减掉它就只剩垂直于射线的分量。因此 $(I-\hat d_i\hat d_i^{\top})(x-p_i)$ 恰是"点 $x$ 到直线 $L_i$"的垂向量，其范数 $\big\lVert(I-\hat d_i\hat d_i^{\top})(x-p_i)\big\rVert_2$ 就是点到直线的垂直距离（投影矩阵幂等 $P^{\top}P = P$，故 $\lVert Pv \rVert^2 = v^{\top}Pv$，范数平方可直接当二次型用）。整个目标函数的含义一句话：**找一个点 $x$，使它到全部 $K$ 条射线的垂直距离平方和最小**。$K$ 条射线在方向噪声下不严格过同一个点，就取这个"最接近公共交点"的位置作为平移估计——"垂距最小二乘交点"名字即来源于此（$\arg\min_x$ 读作"使右侧最小的 $x$ 取值"）。
   **怎么变成 $Ax=b$**：目标是 $x$ 的二次函数，令梯度 $2\sum_i (I-\hat d_i\hat d_i^{\top})(x - p_i) = 0$，等价于每个候选各给 3 行线性方程 $(I-\hat d_i\hat d_i^{\top})x = (I-\hat d_i\hat d_i^{\top})p_i$，堆成过定系统 $A x = b$，$A = \big[(I-\hat d_1\hat d_1^{\top});\,(I-\hat d_2\hat d_2^{\top});\,\dots;\,(I-\hat d_K\hat d_K^{\top})\big] \in \mathbb{R}^{3K\times 3}$，$b$ 用同样的行拼法。两个要点：**(i)** 每个候选的行在沿 $\hat d_i$ 的方向上整列为零——未知尺度 $\lambda_i$ 只滑动沿射线的点、被投影清零，所以根本进不了方程（Q2 的消尺度机制在此公式层面的体现）；**(ii)** 若 $K$ 条射线方向全近平行（直线运动），$A$ 的列空间缺行进方向，最小奇异值 $\sigma_{\min}\to 0$，SVD 伪逆 $x^* = VS^{-1}U^{\top}b$（$A = USV^{\top}$ 的分解，$S^{-1} = \mathrm{diag}(1/\sigma_j)$）在 $1/\sigma_{\min}$ 处把该方向的噪声无限放大——代码未做 rcond 截断，故解沿轨迹方向直接发散（§二·性质 2）。无退化时，$x^*$ 是这组超定垂距方程的最小二乘解，也是二次目标的全局最小点。

输出 $T^{w}_{q} = [\bar{R} \mid x^{*}]$。两点诚实的限定：**(a)** "Robust" 只属于旋转支路——平移的 LS 是等权非鲁棒的，一个远基线坏候选就能把交点拉偏 $B\delta$（$B$=其基线、$\delta$=其方向角误差），这正是 §二·性质 1 里"先按基线过滤再平均"存在的原因；**(b)** 旋转与平移走两条完全独立的支路、最后拼装成 4×4，不存在对 6 自由度的联合优化。

### Q2 单对平移 $t_{d\leftarrow q}$ 无尺度，那合成的候选 $T^{w}_{q,i}$ 在世界系里还准吗？

**单个候选不准——准的只是它所在的射线，不准的是沿射线的距离。**聚合步骤的存在理由正在于此，机制如下：

把模型对候选 $i$ 的平移写成"方向 + 未知尺度"：$t_i = \lambda_i \hat t^{\,true}_i$，$\lambda_i > 0$ 未知且逐候选各不相同（训练目标只约束方向 $\hat t$，不约束 $\lVert t_i \rVert$）。于是候选光心

$$
t^{w}_{q,i} = t^{w}_{d_i} + \lambda_i\, R^{w}_{d_i}\, \hat t^{\,true}_i
$$

= 从**已知米制锚点** $t^{w}_{d_i}$ 出发、沿**正确方向**射线 $L_i$ 上的一个**滑点**——射线本身是对的（起点准、方向准），$\lambda_i$ 只决定滑到哪儿（错）。

关键观察：**真值光心 $x^{true}$ 同时落在全部 $K$ 条射线上**（每条射线按构造都过它，滑动的 $\lambda_i$ 不改直线）。而垂距投影 $(I - \hat d_i \hat d_i^{\top})$ 恰好把沿射线方向的分量整列清零——所以 Q1 第 3 步的法方程 $Ax=b$ 里**根本不含 $\lambda_i$**：求交点时每个候选的"错误距离"被自动丢弃，$K$ 个独立未知尺度无需逐个估计就被消掉了。这就是"经过 $T^{w}_{d}\,T_{d\leftarrow q}$ 合成后由 db 之间的距离定标"的确切含义：**米制尺度不来自任何一个 $\lambda_i$，而来自射线起点 $t^{w}_{d_i}$ 之间的已知间距**（锚点几何），模型只贡献方向。与双目测深同理：绝对深度 = 已知基线长 × 视差角，从不依赖"单条视线自带距离"。

由此，"准"的条件与失效模式都有明确判据：

- 交点误差由方向角误差经杠杆臂放大：$\lVert x^* - x^{true} \rVert \sim B\,\delta$（$B$ = 候选到真值的基线）。ZC 实测与此吻合：$B$=0–1 m 的候选平移误差 0.45 m、$B$=10–20 m 时 11.8 m（§二·误差结构，corr=0.993）——同一个 $\delta$，$B$ 大 10 倍误差就大 10 倍，所以远基线候选只有毒、没有票的价值；基线过滤/排序质量（换 MegaLoc 的增益来源）都在改善这个。
- 射线近平行（纯直线运动，$\hat d_i$ 同向）时 $(I-\hat d_i\hat d_i^{\top})$ 沿行进方向集体失去约束，$A$ 秩亏、最小奇异值 $\sigma_{\min}\to 0$，伪逆按 $1/\sigma_{\min}$ 放大噪声 → 交点沿行进方向漂几米。这就是"横向由 db 视差锚定、沿行进方向是机制性弱项"（ZC 误差分解 med 比 ~4:1、p90 比 ~10:1）的数学根源：横向与竖直有 $K$ 个锚点的视差交叉约束，唯独行进方向各射线近似共线、约束退化。db 在侧向布设视差、或时序/里程计给该方向供尺度，是仅有的两条出路。
- 旋转完全不受尺度问题影响：$\lambda$ 只乘在平移上，$R^{w}_{q,i} = R^{w}_{d_i} R_i$ 与 $\lambda_i$ 无关，每个候选的旋转支路本来就是"真旋转的近似"，median 只是剔掉检索跑飞的少数票。

---

文档时间戳：2026-09-29 09:55 CST（LaTeX 公式化更新；10:20 增补 §七 运动平均答疑 Q1/Q2）
