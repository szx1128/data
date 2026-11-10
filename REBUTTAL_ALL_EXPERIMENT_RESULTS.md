# ContRS：ICLR 2027 Rebuttal 补充实验完整配置与结果汇总

论文：**ContRS: Generalizing Randomized Smoothing to Continuous Outputs**  
整理日期：2026-11-10  
语言：中文；原始结果表中的英文列名和方法名保留，避免转录时改变含义。

> 本文档把现有正式结果表**原样内嵌**在一个 Markdown 文件中，同时补充统一配置、比较口径和证据边界。它不是只放链接的索引，也没有重新估算或手工改动表内数值。正文的范围说明和补充注释优先于内嵌旧表中的过时提示。

## 0. 先读：实验覆盖范围

目前有四组主要实验包，**它们不是“300 图 × 所有 p × 所有架构 × 所有 baseline”的全组合**：

| 实验包 | 模型与设置 | 每组图像数 | 已完成内容 |
|---|---|---:|---|
| 原任务扩展 | MTCNN、CIFAR-VAE、CLIP ViT-L/14；另有 VAE 像素 ℓ∞ | 300 | ContRS、safe region（p=0.55）、center smoothing、ContRS-M；精确平均半径补跑；95% CI |
| 新 baseline | MTCNN/Jaccard/σ=0.1，CLIP/cosine/σ=0.25，VAE/像素 ℓ∞/σ=0.25 | 300 | 中位数平滑、均值/嵌入平滑、α-平滑、ContRS-M；95% CI |
| 多架构 | 检测 3 个、重建 3 个、检索 3 个核心模型；另有 2 个官方预处理对照和 VAE-ℓ∞ | 50 | ContRS、safe region（p=0.55）、center smoothing；完整表另含逐图匹配及旧 joint 诊断 |
| safe-region target-p 扫描 | MTCNN、CIFAR-VAE、CLIP；p=0.6/0.7/0.8，σ=0.1–0.5 | 50 | 共 45 个模型–σ–p 比较行；MTCNN 含认证率、饱和率和描述性 CI |
| 早期 baseline | MTCNN 49 图；CLIP 与 VAE-ℓ∞ 各 50 图 | 49/50 | 中位数平滑、α-平滑、ContRS-M；保留作历史对照，不与 300 图主表混写 |
| 独立新种子复核 | MTCNN σ=0.5 的 safe-region 比较；CLIP σ=0.25 的 center 和新增 baseline | 300 | seed=20261109，只复核这两组例外，不是全套重跑 |

补充说明：

1. **300 图 safe-region 主表固定 p=0.55**；改 p 的扫描仍然是每行 50 图。
2. 新增六个架构的正式结果是每组 50 图，只比较 center smoothing 和 safe region；不能把原模型的 300 图新增 baseline 结果写成所有新架构都已跑完。
3. VAE 的新增逐坐标 baseline 使用**像素颜色值的 ℓ∞ 距离**，不是 LPIPS。普通 VAE-LPIPS 的 center/safe 结果仍单独保留。
4. 本汇总不包含攻击实验；按当前 rebuttal 范围，只整理已完成的认证、平滑误差、架构和 baseline 实验。
5. “300 图”表示同一任务的固定 300 图样本池在不同设置下复用，不是每个设置重新独立抽 300 张。

## 1. 统一符号、认证和比较口径

### 1.1 输入扰动与输出距离

对输入图像 \(x\)，加噪样本为

\[
x+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma^2I),
\]

加噪后按模型输入定义裁剪到合法颜色范围，再进入 base predictor。三种任务输出距离为：

| 任务 | 输出 | 距离 \(d\) | \(\kappa\) | 代码邻域半径 \(s\) 到论文输出容差的换算 |
|---|---|---|---:|---:|
| 人脸/人物检测 | xyxy 框，漏检用非法框标记 | Jaccard distance = \(1-\mathrm{IoU}\) | 1 | \(r=2s\) |
| 图像重建 | 重建图像 | LPIPS | 2 | \(r=4s\) |
| 图文检索 | L2 归一化图像嵌入 | cosine distance = \(1-\cos\) | 2 | \(r=4s\) |
| 重建的 baseline 对照 | \([-1,1]\) 像素颜色值 | \(\ell_\infty=\max_i\lvert y_i-y'_i\rvert\) | 1 | 表内直接报告相应容差与半径 |

平滑误差始终写作

\[
\operatorname{err}=d(\text{所选平滑中心/预测},f(x)),
\]

**不乘 \(2\kappa\)**。表中的 `ContRS err`、`SR err`、`CS err` 等均遵循这一约定。

### 1.2 ContRS：split 三样本池认证

正式实验使用 split 认证器，不使用旧 joint 认证器作为主结果：

- P1：\(N_1=5000\)，候选中心；
- Psel：\(N_{\rm sel}=5000\)，独立选择中心；
- P2：\(N_2=10000\)，独立认证；
- 总失败参数 \(\alpha=0.005\)。

对候选中心 \(c\) 和代码邻域半径 \(s\)：

\[
k=\#\{j:d(y_j,c)\le s\},\qquad
p_L=\operatorname{ClopperPearsonLower}(k/N_2),
\]

\[
R=\sigma\Phi^{-1}(p_L)\quad(p_L>0.5),\qquad R=0\quad\text{otherwise}.
\]

代码通过 `proportion_confint(k, N2, alpha=0.01, method="beta")[0]` 得到单侧下界，等价于总 alpha=0.005 的实现口径。\(R\) 是输入 \(\ell_2\) 认证半径，越大越好；\(r\) 是输出变化容差，作为输出界时越小越紧。

有限样本 ContRS 的直接理论表述是：扰动后的平滑输出相对**部署时样本中心 \(\hat z\)** 的变化界。不能不加说明地把 \(\hat z\) 替换成总体 \(g(x)\)，再声称同样的有限样本保证。

### 1.3 safe region：先构造球，再认证

safe region 的 \(p\) 是 P1 上构造最小包围球时的**目标样本覆盖率**，不是认证时的 \(p_L\)。本实现取

\[
q_{\rm idx}=\lceil pN_1\rceil-1,
\]

对每个 P1 候选求其到 P1 样本的第 \(q_{\rm idx}+1\) 小距离，选择该半径最小的候选作为中心，距离并列时保留第一个。随后用独立 P2 计数和同一 CP 下界计算 \(R\)。

原算法和扩展实验配置：

- `n_pred=5000`，`n_cert=10000`，`alpha=0.005`；
- 独立旧 KDD 驱动与本次 p 扫描的 KDD 端批大小均为 32；300 图/多架构共用主驱动按注册表 batch 采样，样本数与认证算法不变；
- 300 图和 50 图多架构主实验固定 `p=0.55`；
- p 扫描使用 `p=0.6/0.7/0.8`，分别对应每图第 3000、3500、4000 小距离。

**论文式组均值尺度匹配：**对每个 \((model,\sigma,p)\)，先求

\[
\bar s=\operatorname{mean}_{n\ images}(r_{SR}),
\]

再把同一个 \(\bar s\) 用于该组全部 ContRS 图像，并比较 \(R\) 与平滑误差；输出容差为 \(r=2\kappa\bar s\)。这不是逐图 \(s_i=r_{SR,i}\) 匹配。逐图匹配结果仅在多架构完整诊断表中另列，不能与组均值口径混写。

safe region 给出的是 safe-region membership certificate，不能直接改名为 ContRS 相对部署中心的 prediction-change 证书。

### 1.4 center smoothing：输入半径匹配，比较输出界

对每个预设 ContRS 输出容差 \(r\)：

1. 取该 \(r\) 下 300/50 图的 ContRS 平均认证半径作为 center smoothing 的输入半径：\(\varepsilon_{in}=\overline R\)；
2. center smoothing 输出一个认证输出界 \(\varepsilon_{out}\)；
3. 比较 **ContRS 的 \(r\)** 与 **center smoothing 的 \(\varepsilon_{out}\)**，两者越小越紧；
4. 同时比较双方相对 clean 输出的平滑误差。

不能直接把 ContRS 的输入半径 \(R\) 与 center smoothing 的输出界 \(\varepsilon_{out}\) 当作同类量比较。

center 配置：P1/Psel/P2 与 ContRS 同池；`CS_ALPHA1=CS_ALPHA2=0.005`，`CS_DELTA=0.05`。未乘系数的分位数半径为

\[
q=\Phi\!\left(\Phi^{-1}(0.5+\delta)+\varepsilon_{in}/\sigma\right)
+\sqrt{\frac{\log(1/\alpha_2)}{2N_2}},\qquad \delta=0.05,
\]

在 N2 个到中心距离上取该分位数。center 检查失败或分位数超出 \([0,1]\) 时弃权；输出界只在未弃权图上平均，弃权率另报。

理论有效系数：

- Jaccard 与像素 ℓ∞：3；
- LPIPS/cosine：松弛三角系数 \(\gamma=2\)，故 \(\gamma(1+2\gamma)=10\)。

表中的 `coef 3` 对 LPIPS/cosine 只是投稿 Fig.1 旧口径对照，**不能替代理论有效系数 10 的主结论**。

### 1.5 查询预算与噪声池

- split ContRS：不含共同 clean 前向为 \(5000+5000+10000=20000\) 次/图；含 clean 为 20001。
- safe region：不含 clean 为 \(5000+10000=15000\) 次/图；含 clean 为 15001。
- 因此 ContRS vs safe region **不是等查询预算比较**。
- 多架构与 300 图主驱动中，ContRS、safe region、center smoothing 共用 P1/Psel/P2；这是共享样本池的主实验协议。
- p 扫描中，ContRS 与 safe region 使用同图但独立噪声池；三个 p 在 safe region 内共用 P1/P2，ContRS 一次采样同时评估三个目标半径。
- baseline 驱动另用 seed 命名空间生成 N=10000 样本，median/mean/alpha 在该 baseline 池内共用样本；不能与主驱动的共享池协议混为一谈。

## 2. 数据、模型、样本和计算配置

### 2.1 300 图固定清单

| 任务 | 数据源与输入值域 | 固定清单 | 说明 |
|---|---|---|---|
| MTCNN | 本机 CelebA aligned，RGB/[0,1] | [mtcnn_celeba_300.csv](ext300/manifests/mtcnn_celeba_300.csv) | 原 50 张加 seed=20261107 新抽 250 张；使用本机镜像重压缩图，三方法同图比较 |
| CIFAR-VAE | CIFAR-10 test，RGB 归一化到 [-1,1] | [cifar10_300.csv](ext300/manifests/cifar10_300.csv) | 固定 CIFAR test 索引 |
| CLIP | COCO val2014，RGB/[0,1] | [clip_coco_300.csv](ext300/manifests/clip_coco_300.csv) | 固定 300 个 COCO 文件名 |

三份清单均为 300 行，图像索引/文件名各 300 个唯一值。CelebA 本机版本与用户机器的官方原图像素哈希不同；用户决定不上传官方原图，因此 300 图 MTCNN 使用本机数据。该结果支持同机同图方法比较，不能声称与用户另一环境的官方像素结果逐点一致。

### 2.2 50 图多架构模型与预处理

| 注册名 | 架构 | 数据/任务 | 距离 | 推荐 batch | 关键预处理或输出约定 |
|---|---|---|---|---:|---|
| mtcnn | 级联 CNN | CelebA wild test/人脸检测 | Jaccard | 256 | 输出人脸框 |
| retinaface | RetinaFace-R50/CNN | 同 MTCNN 样本池 | Jaccard | 128 | R50 权重 |
| detr | DETR-R50/Transformer | COCO 2017 val 大人物框 | Jaccard | 128 | 人物面积≥图像 5%，取最大人物 GT；短边 512、ImageNet 归一化、人物概率阈值 0.5 |
| cifar_vae | CIFAR VAE/CNN | CIFAR-10 test/重建 | LPIPS | 2500 | 32×32，[-1,1] |
| sd_vae | SD-VAE/CNN | 同一 CIFAR 样本池 | LPIPS | 1000 | 输入 bicubic 上采样 2×，重建后 area 下采样回 32×32 |
| mae | MAE ViT-B/Transformer | 同一 CIFAR 样本池 | LPIPS | 1000 | 上采样 224，固定 75% mask，mask_seed=0，输出全部 patch 预测后下采样 |
| clip_l14 | CLIP ViT-L/14/Transformer | COCO val2014/检索 | cosine | 500 | 整图 bilinear 缩放到方形输入，checkpoint 均值方差，L2 归一化 |
| clip_rn50x16 | CLIP RN50x16/CNN | 同一 COCO 样本池 | cosine | 250 | 与主 CLIP 驱动一致的整图缩放预处理 |
| siglip | SigLIP ViT-B/16/Transformer | 同一 COCO 样本池 | cosine | 500 | 与主 CLIP 驱动一致的整图缩放预处理 |
| clip_rn50x16_off | 同 RN50x16 checkpoint | 预处理对照 | cosine | 250 | 官方 bicubic 短边缩放 384、antialias、中心裁剪 384 |
| siglip_off | 同 SigLIP checkpoint | 预处理对照 | cosine | 500 | 官方 bicubic 224×224、antialias，不裁剪 |
| cifar_vae_linf | 同 CIFAR VAE | 像素颜色值对照 | ℓ∞ | 2500 | 同一 VAE，不是新架构 |

实际 batch 还会按单图元素数限制为 `min(batch, 600000000 // x.numel())`，只改变分批，不改变每张图的样本数或认证算法。TF32 关闭。

50 图数据选择：CIFAR/COCO2014 使用 NumPy `default_rng(42)` 无放回抽样后排序；CelebA wild 使用 `torch.Generator().manual_seed(42)` 和 `torch.randperm`；COCO2017 先按人物规则筛选，再以 NumPy seed=42 抽样。

### 2.3 σ、r 网格与种子

- safe-region 主扫描：\(\sigma\in\{0.1,0.2,0.3,0.4,0.5\}\)。
- center smoothing：检测 σ=0.1，重建/检索 σ=0.25。
- 300 图主结果还保留 VAE/CLIP σ=0.25 和 VAE-ℓ∞ σ=0.25。
- 原三模型的 r 使用论文网格：MTCNN `0.08/0.10/0.12/0.14/0.16`；VAE-LPIPS `0.012/0.014/0.016/0.018/0.020`；CLIP `0.080/0.088/0.100/0.108/0.120`。
- 新架构在密集对数 s 网格上选择 5 个 r，使组平均 ContRS \(R\in[0.3\sigma,1.8\sigma]\)，网格下标等距取点。
- 主驱动 seed=42，逐图 seed 为 SHA256(`42:model:sigma:image`) 的前 8 个 little-endian 字节模 \(2^{31}\)。同图同 σ 的精确半径补跑复用该 seed，并把精确 \(\bar s\) 加入网格。
- p 扫描 MTCNN 本地 KDD 使用独立命名空间 SHA256(`42:kdd-p-sweep:mtcnn_kdd50:sigma:image`)；ContRS 使用主驱动命名空间。它是本地独立重跑，不是另一会话全局 RNG 流的续跑。
- 独立复核只把 seed 改为 20261109，其他样本清单与算法配置不变。

### 2.4 新增 baseline 的输入、认证和比较方式

正式 300 图 baseline 每图 N=10000，与 ContRS 的 N2 数量相同；MTCNN σ=0.1、CLIP σ=0.25、VAE-ℓ∞ σ=0.25。每个 baseline NPZ 同时保存 median/mean/alpha 结果。

**中位数平滑（Chiang et al., NeurIPS 2020）**

- 预测为每个坐标排序后的第 `n//2` 个值。
- 正向比较：令 \(\varepsilon_{in}\) 为同一 r 下 ContRS 的平均 R，构造逐坐标区间 [L,U]；检测枚举坐标盒 16 个角点求最坏 Jaccard，检索使用坐标盒到参考嵌入的保守 cosine 上界，像素 ℓ∞ 取盒内最大逐坐标偏差。
- 比较 ContRS r 与 median 的 \(\varepsilon_{out}\)，以及双方平滑误差。
- 反向比较：固定同一 r，用 25 次二分在 \([0,4\sigma]\) 内求最大可认证输入半径，比较 ContRS R 与 median R。
- 逐坐标置信度经并集界设为 \(1-\alpha/(2D)\)，总失败参数与 ContRS 的 0.005 对齐。

**随机 α-平滑（Rekavandi et al., NeurIPS 2024）**

- 官方默认算法参数：α=0.35、`n_sample=10`、P=0.9；本实现 β=0.01，使失败概率口径与 ContRS 对齐。
- 以 clean 输出 \(f(x)\) 为参考，接受条件为 \(\max_i|y_i-f_i(x)|<\varepsilon_y\)。从接受计数得到 \(p_{lower}\)、\(p_{success}\)，再计算
  \(R_\alpha=\max\{0,\sigma[\Phi^{-1}(p_{lower})-\Phi^{-1}(p_{success})]\}\)。
- 为与 ContRS 输出容差对应，先令 safe 邻域 \(s=r/(2\kappa)\)，再把 s 转成保证任务距离所需的逐坐标 \(\varepsilon_y\)：Jaccard 对坐标盒 16 角点二分；cosine 使用 \(\varepsilon_y=\sqrt{(2s-s^2)/D}\)；像素 ℓ∞ 直接取 s。
- 平滑预测为每组 10 个样本的 α-截尾均值，共 1000 组；表中平滑误差是这些组预测误差的平均。

**均值/嵌入平滑：不是一篇独立的新论文 baseline**

- 检测/重建：逐坐标期望平滑，使用 Salman et al. 2019 的期望界；有界坐标经经验 Bernstein 区间和并集界估计，再映射为任务输出盒。检测框坐标裁剪到合法范围，VAE 像素裁剪到 [-1,1]。
- 检索：使用 Pautov et al., NeurIPS 2022 的 Smoothed Embeddings 半径界，参考点为经验均值嵌入，并加入有限样本集中项。
- 与中位数平滑一样，同时给出“ε_in=ContRS 平均 R 时的 ε_out”和“同一 r 下的 R”两种比较。
- **不能把这些结果包装成一篇名为“Mean Smoothing”的独立新论文。**Pautov 论文直接支撑检索部分，不直接支撑检测/重建部分。

**ContRS-M：内部消融，不是外部 baseline**

在 P1 的 5000 个样本候选之外，只额外加入 P1 的逐坐标中位数和均值；Psel 仍独立选中心，P2 仍独立认证，证书公式不变，平局优先原样本候选。其作用是检验统计量中心能否缩小与中位数平滑的差距。

**为什么 VAE baseline 改用像素 ℓ∞：**中位数、α 和逐坐标期望方法需要坐标式接受域或坐标区间；LPIPS 是深度网络诱导的非坐标距离，无法直接套用这些证书。因此不能声称在 LPIPS 上完成了这三个 baseline 的认证，改用同一 VAE 输出的像素颜色值 ℓ∞ 距离。该对照与 VAE-LPIPS 主实验是两种不同输出度量。

## 3. 结果摘要（详细表见后文）

### 3.1 300 图主实验

- safe region p=0.55、组平均半径匹配下，VAE 和 CLIP 各 5/5 个 σ 的 ContRS 平均 R 更大；MTCNN 在 σ=0.1–0.4 更大，σ=0.5 反转。
- MTCNN σ=0.5 的映射输出容差约 1.1747，大于 Jaccard 距离上界 1，是平凡/饱和边界案例，不能只按 R 大小解释成有意义的人脸框优势。
- 对理论有效 center 系数，20/20 个 r 点均为 CS \(\varepsilon_{out}>r\)，即 ContRS 的预设输出界更紧。平滑误差需逐行看配对 CI，不能笼统声称所有点完全相同。
- 300 图未校正逐行 CI 中，safe-region ΔR 为 14/15 全正、1/15 全负；center 的 \(\varepsilon_{out}-r\) 为 20/20 全正。区间未做多重比较校正。

### 3.2 300 图新增 baseline

- 中位数平滑在 15/15 个同容差点上的 R 大于原 ContRS；ContRS-M 在 CLIP/VAE-ℓ∞ 明显改善，但 15/15 点仍低于中位数。这是必须如实保留的负面结果。
- α-平滑在 MTCNN/CLIP 上几乎或全部为 0，在 VAE-ℓ∞ 上非零但仍低于 ContRS；ContRS−α 的 15/15 行 CI 全正。
- 均值/嵌入平滑：ContRS−mean 的 13/15 行 CI 全正，2 行跨 0，0 行全负。两个跨 0 点是 CLIP r=0.08 和 VAE-ℓ∞ r=0.2852。
- baseline 往往有更小的平滑误差；半径更大不能写成所有指标都更好。

### 3.3 多架构和 p 扫描

- 六个新增架构均有 center 与 safe 两套 50 图表；RN50x16/SigLIP 另有官方预处理对照。
- p 扫描共 45 行：VAE/CLIP 的 30 行在所提供 mean_safe_r 数值上均为 ContRS 平均 R 更大；MTCNN 为 12/15 正、σ=0.5 三行负。
- MTCNN σ=0.5 的 safe_r=1 饱和率为 p=0.6/0.7/0.8 下 50%/52%/56%，且 ContRS 的 r>1，因此三行反转和高 R 都带有边界饱和语义。
- 上述计数是描述性结果，不是跨样本、跨 p、跨架构无条件显著性结论。

## 4. 置信区间约定

均值区间为双侧 95% Student-t CI：

\[
\bar x\pm t_{0.975,n-1}\frac{\operatorname{sd}(x)}{\sqrt n}.
\]

配对差先按同一 image ID 相减再求区间：ΔR=ContRS−baseline，越大越有利于 ContRS；Δerr=ContRS−baseline，越小越有利于 ContRS。center 输出界只在未弃权图上统计，弃权率使用 Wilson 95% CI。

所有逐行 CI **未做多重比较校正**。此外，safe-region 组均值 \(\bar s\) 本身由同一批图像估计，固定 \(\bar s\) 的普通配对 t 区间没有传播这部分不确定性；p 扫描还存在三个 p 共用 P1/P2、同一图片跨组重复和行间相关。因此这些区间应称为描述性逐行区间，不应据“多少行排除 0”写成经过多重校正的整体优越性证明。

---

# 第一部分：300 图原任务主实验完整结果

以下先嵌入论文 Fig.1 口径主表，再嵌入含 n、逐图匹配和诊断字段的完整表。主表 safe-region 数值均为精确平均半径补跑结果；内嵌源文件中“nearest lower grid”的通用生成提示不代表当前仍缺精确值。


## 1.1 论文格式：center smoothing 与 safe region（300 图）

样本量均为 300；center 的有效样本数因弃权而略少，精确数量见第七部分 CI。VAE-ℓ∞ 只有 center 表，没有列入 15 组 safe-region 精确补跑。

以下来源表已完整内嵌：[ext300/summary/paper_format.md](ext300/summary/paper_format.md)。

#### Paper-format tables (Fig.1 protocol)

##### vs center smoothing: ContRS certified R is used as center smoothing's ε_in; compare ContRS's r with center smoothing's ε_out, and both smoothing errors

ε_out uses center smoothing's own constant (3 for Jaccard, γ(1+2γ)=10 for LPIPS/cosine).


**mtcnn300** (CNN (cascade), σ=0.1)

| r (ContRS) | ContRS R = ε_in | ε_out CS | smooth err ContRS | smooth err CS |
|---|---|---|---|---|
| 0.08 | 0.017 | 0.1570 | 0.0546 | 0.0546 |
| 0.1 | 0.048 | 0.1879 | 0.0547 | 0.0546 |
| 0.12 | 0.082 | 0.2248 | 0.0549 | 0.0546 |
| 0.14 | 0.117 | 0.2679 | 0.0562 | 0.0546 |
| 0.16 | 0.150 | 0.3226 | 0.0577 | 0.0546 |

**cifar_vae300** (CNN (VAE), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS | smooth err ContRS | smooth err CS |
|---|---|---|---|---|
| 0.012 | 0.147 | 0.0394 | 0.0014 | 0.0014 |
| 0.014 | 0.209 | 0.0438 | 0.0014 | 0.0014 |
| 0.016 | 0.272 | 0.0490 | 0.0014 | 0.0014 |
| 0.018 | 0.332 | 0.0551 | 0.0015 | 0.0014 |
| 0.02 | 0.386 | 0.0619 | 0.0015 | 0.0014 |

**clip_l14_300** (Transformer (ViT-L/14), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS | smooth err ContRS | smooth err CS |
|---|---|---|---|---|
| 0.08 | 0.096 | 0.2643 | 0.2356 | 0.2355 |
| 0.088 | 0.134 | 0.2772 | 0.2350 | 0.2355 |
| 0.1 | 0.197 | 0.3012 | 0.2349 | 0.2355 |
| 0.108 | 0.240 | 0.3205 | 0.2355 | 0.2355 |
| 0.12 | 0.303 | 0.3541 | 0.2356 | 0.2355 |

**cifar_vae300_linf** (CNN (VAE), pixel l_inf, σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS | smooth err ContRS | smooth err CS |
|---|---|---|---|---|
| 0.2852 | 0.090 | 0.4677 | 0.0880 | 0.0871 |
| 0.3097 | 0.152 | 0.4893 | 0.0883 | 0.0871 |
| 0.3364 | 0.238 | 0.5231 | 0.0897 | 0.0871 |
| 0.3653 | 0.341 | 0.5714 | 0.0910 | 0.0871 |
| 0.3967 | 0.444 | 0.6446 | 0.0928 | 0.0871 |

##### vs safe region: run safe region (p=0.55) first, set ContRS r = 2κ·mean(r_SR) at each σ; compare R and smoothing errors

(* = evaluated at the nearest lower grid point; replaced automatically by the exact rerun once it finishes)


**mtcnn300**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0913 | 0.0330 | 0.0086 | 0.0545 | 0.0538 |
| 0.2 | 0.1218 | 0.0947 | 0.0214 | 0.0758 | 0.0776 |
| 0.3 | 0.2234 | 0.3783 | 0.0710 | 0.1060 | 0.1261 |
| 0.4 | 0.5226 | 0.6655 | 0.2760 | 0.1518 | 0.2517 |
| 0.5 | 1.1747 | 0.3196 | 0.9040 | 0.1866 | 0.5219 |

**cifar_vae300**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0021 | 0.0621 | 0.0093 | 0.0001 | 0.0001 |
| 0.2 | 0.0080 | 0.1238 | 0.0181 | 0.0008 | 0.0008 |
| 0.3 | 0.0172 | 0.1840 | 0.0270 | 0.0023 | 0.0023 |
| 0.4 | 0.0290 | 0.2448 | 0.0370 | 0.0045 | 0.0046 |
| 0.5 | 0.0428 | 0.2973 | 0.0455 | 0.0075 | 0.0078 |

**clip_l14_300**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0563 | 0.0731 | 0.0090 | 0.0914 | 0.0913 |
| 0.2 | 0.0858 | 0.1251 | 0.0178 | 0.1889 | 0.1888 |
| 0.3 | 0.0997 | 0.1962 | 0.0268 | 0.2726 | 0.2727 |
| 0.4 | 0.1085 | 0.2326 | 0.0366 | 0.3306 | 0.3310 |
| 0.5 | 0.1132 | 0.2508 | 0.0446 | 0.3785 | 0.3785 |


## 1.2 完整汇总：样本数、逐图匹配、组均值匹配与诊断字段（300 图）

本表保留两种 safe-region 比较口径。正式论文式结论是口径 (b) 的组平均半径；口径 (a) 是逐图匹配。`joint` 仅是旧认证器诊断，不是有效主证书。

以下来源表已完整内嵌：[ext300/summary/tables.md](ext300/summary/tables.md)。

#### Multi-architecture results (auto-generated)

##### Completion (images per sigma)

| model | σ=0.1 | σ=0.2 | σ=0.25 | σ=0.3 | σ=0.4 | σ=0.5 |
|---|---|---|---|---|---|---|
| mtcnn300 | 300 | 300 | 0 | 300 | 300 | 300 |
| cifar_vae300 | 300 | 300 | 300 | 300 | 300 | 300 |
| clip_l14_300 | 300 | 300 | 300 | 300 | 300 | 300 |
| cifar_vae300_linf | 0 | 0 | 300 | 0 | 0 | 0 |

##### Clean task quality of each base model (σ-independent)

| model | family | task | clean metric |
|---|---|---|---|
| mtcnn300 | CNN (cascade) | det | embedding model (Recall not computed here) |
| cifar_vae300 | CNN (VAE) | recon | LPIPS(f(x), x): mean 0.1999 |
| clip_l14_300 | Transformer (ViT-L/14) | retr | embedding model (Recall not computed here) |
| cifar_vae300_linf | CNN (VAE), pixel l_inf | recon | embedding model (Recall not computed here) |

##### ContRS vs center smoothing (input radius matched: ε_in = mean ContRS R)

ρ_CS: center smoothing certified output bound at ε_in, averaged over images where it does not abstain (abstain = failed center check or quantile > 1). 'valid' uses the coefficient required by center smoothing's theory (3 for the Jaccard metric, γ(1+2γ)=10 for γ=2 relaxed-triangle LPIPS/cosine); 'coef 3' reproduces the constant used in the paper's Fig.1. ContRS output bound is the prescribed r. Lower is tighter.


**mtcnn300** (CNN (cascade), σ=0.1, n=300, r grid: paper grid)

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.08 | 0.017 | 47% | 0.0546 | 0.1570 | 1.96 | 0.1570 | 1.96 | 1% | 0.0546 |
| 0.1 | 0.048 | 78% | 0.0547 | 0.1879 | 1.88 | 0.1879 | 1.88 | 1% | 0.0546 |
| 0.12 | 0.082 | 92% | 0.0549 | 0.2248 | 1.87 | 0.2248 | 1.87 | 1% | 0.0546 |
| 0.14 | 0.117 | 97% | 0.0562 | 0.2679 | 1.91 | 0.2679 | 1.91 | 1% | 0.0546 |
| 0.16 | 0.150 | 98% | 0.0577 | 0.3226 | 2.02 | 0.3226 | 2.02 | 1% | 0.0546 |

**cifar_vae300** (CNN (VAE), σ=0.25, n=300, r grid: paper grid)

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.012 | 0.147 | 58% | 0.0014 | 0.0394 | 3.28 | 0.0118 | 0.99 | 1% | 0.0014 |
| 0.014 | 0.209 | 72% | 0.0014 | 0.0438 | 3.13 | 0.0131 | 0.94 | 1% | 0.0014 |
| 0.016 | 0.272 | 82% | 0.0014 | 0.0490 | 3.06 | 0.0147 | 0.92 | 1% | 0.0014 |
| 0.018 | 0.332 | 89% | 0.0015 | 0.0551 | 3.06 | 0.0165 | 0.92 | 1% | 0.0014 |
| 0.02 | 0.386 | 91% | 0.0015 | 0.0619 | 3.10 | 0.0186 | 0.93 | 1% | 0.0014 |

**clip_l14_300** (Transformer (ViT-L/14), σ=0.25, n=300, r grid: paper grid)

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.08 | 0.096 | 42% | 0.2356 | 0.2643 | 3.30 | 0.0793 | 0.99 | 0% | 0.2355 |
| 0.088 | 0.134 | 53% | 0.2350 | 0.2772 | 3.15 | 0.0832 | 0.95 | 0% | 0.2355 |
| 0.1 | 0.197 | 66% | 0.2349 | 0.3012 | 3.01 | 0.0904 | 0.90 | 0% | 0.2355 |
| 0.108 | 0.240 | 73% | 0.2355 | 0.3205 | 2.97 | 0.0962 | 0.89 | 0% | 0.2355 |
| 0.12 | 0.303 | 83% | 0.2356 | 0.3541 | 2.95 | 0.1062 | 0.89 | 0% | 0.2355 |

**cifar_vae300_linf** (CNN (VAE), pixel l_inf, σ=0.25, n=300, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.2852 | 0.090 | 45% | 0.0880 | 0.4677 | 1.64 | 0.4677 | 1.64 | 1% | 0.0871 |
| 0.3097 | 0.152 | 67% | 0.0883 | 0.4893 | 1.58 | 0.4893 | 1.58 | 1% | 0.0871 |
| 0.3364 | 0.238 | 89% | 0.0897 | 0.5231 | 1.56 | 0.5231 | 1.56 | 1% | 0.0871 |
| 0.3653 | 0.341 | 99% | 0.0910 | 0.5714 | 1.56 | 0.5714 | 1.56 | 1% | 0.0871 |
| 0.3967 | 0.444 | 100% | 0.0928 | 0.6446 | 1.62 | 0.6446 | 1.62 | 1% | 0.0871 |

##### ContRS vs safe region (p = 0.55) at matched output-ball radius

(a) per-image matching s = r_SR (Appendix I as written); (b) one radius per σ = mean r_SR over all images (Fig.1 top style), evaluated exactly at that radius by a rerun with identical seeds (marked * when only the nearest-lower grid point is available yet); joint = old joint certifier on the same pools (diagnostic only, not a valid certificate). Safe-region σ grid follows Table 4.


**mtcnn300**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 300 | 0.0913 | 0.0086 | 0.0086 | 31%/37%/32% | 0.0330 | 0.0091 | 0.0287 | 0.0538 | 0.0538 |
| 0.2 | 300 | 0.1218 | 0.0214 | 0.0215 | 34%/37%/30% | 0.0947 | 0.0223 | 0.0901 | 0.0776 | 0.0772 |
| 0.3 | 300 | 0.2234 | 0.0710 | 0.0711 | 36%/36%/27% | 0.3783 | 0.0725 | 0.3416 | 0.1261 | 0.1268 |
| 0.4 | 300 | 0.5226 | 0.2760 | 0.2757 | 28%/40%/32% | 0.6655 | 0.2775 | 0.6687 | 0.2517 | 0.2514 |
| 0.5 | 300 | 1.1747 | 0.9040 | 0.9039 | 18%/64%/18% | 0.3196 | 0.9050 | 0.3196 | 0.5219 | 0.5217 |

**cifar_vae300**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 300 | 0.0021 | 0.0093 | 0.0092 | 8%/82%/10% | 0.0621 | 0.0095 | 0.0525 | 0.0001 | 0.0001 |
| 0.2 | 300 | 0.0080 | 0.0181 | 0.0183 | 12%/81%/7% | 0.1238 | 0.0187 | 0.1165 | 0.0008 | 0.0008 |
| 0.3 | 300 | 0.0172 | 0.0270 | 0.0272 | 9%/83%/8% | 0.1840 | 0.0277 | 0.1577 | 0.0023 | 0.0023 |
| 0.4 | 300 | 0.0290 | 0.0370 | 0.0370 | 9%/86%/5% | 0.2448 | 0.0377 | 0.2420 | 0.0046 | 0.0046 |
| 0.5 | 300 | 0.0428 | 0.0455 | 0.0454 | 7%/85%/8% | 0.2973 | 0.0463 | 0.2740 | 0.0078 | 0.0077 |

**clip_l14_300**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 300 | 0.0563 | 0.0090 | 0.0090 | 8%/84%/8% | 0.0731 | 0.0092 | 0.0704 | 0.0913 | 0.0912 |
| 0.2 | 300 | 0.0858 | 0.0178 | 0.0179 | 8%/84%/8% | 0.1251 | 0.0182 | 0.1246 | 0.1888 | 0.1888 |
| 0.3 | 300 | 0.0997 | 0.0268 | 0.0269 | 6%/88%/6% | 0.1962 | 0.0272 | 0.1921 | 0.2727 | 0.2726 |
| 0.4 | 300 | 0.1085 | 0.0366 | 0.0366 | 8%/85%/8% | 0.2326 | 0.0372 | 0.2303 | 0.3310 | 0.3309 |
| 0.5 | 300 | 0.1132 | 0.0446 | 0.0445 | 7%/86%/7% | 0.2508 | 0.0454 | 0.2125 | 0.3785 | 0.3786 |

**cifar_vae300_linf**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|


---

# 第二部分：300 图新增 baseline 完整结果

本部分只覆盖 MTCNN-Jaccard/σ=0.1、CLIP-cosine/σ=0.25、CIFAR-VAE 像素 ℓ∞/σ=0.25 三个设置，每个设置 300 图。普通 VAE-LPIPS 不适用这些逐坐标证书。表 A 是输入半径匹配后的输出界；表 A′ 是同一输出容差下的输入认证半径。


## 2.1 中位数、均值/嵌入、α-平滑与 ContRS-M（300 图）

“均值平滑”是 Salman 2019 期望界与 Pautov 2022 嵌入平滑的组合实现，不是独立的新论文；ContRS-M 是内部消融。

以下来源表已完整内嵌：[ext300/summary/baseline_tables.md](ext300/summary/baseline_tables.md)。

#### 新 baseline 对比（论文原任务与原模型）


#### 检测（Jaccard）：mtcnn300（σ=0.1，300 张图）

**表 A：对比中位数平滑**（中位数平滑的 ε_in = ContRS 的平均 R）

| r（ContRS） | ContRS R = 中位数平滑 ε_in | 中位数平滑 ε_out | 平滑误差 ContRS | 平滑误差 中位数平滑 | 中位数平滑在同一 r 下的 R |
|---|---|---|---|---|---|
| 0.0800 | 0.017 | 0.0138 | 0.0546 | 0.0531 | 0.1234 |
| 0.1000 | 0.048 | 0.0351 | 0.0547 | 0.0531 | 0.1529 |
| 0.1200 | 0.082 | 0.0586 | 0.0549 | 0.0531 | 0.1816 |
| 0.1400 | 0.117 | 0.0823 | 0.0562 | 0.0531 | 0.2092 |
| 0.1600 | 0.150 | 0.1054 | 0.0577 | 0.0531 | 0.2341 |

**表 A'：同一容差 r 下的认证半径**（ContRS-M = 候选中心加入逐坐标中位数/均值，证书不变）

| r | ContRS R | ContRS-M R | 中位数平滑 R | 平滑误差 ContRS | 平滑误差 ContRS-M | 平滑误差 中位数平滑 |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0168 | 0.0170 | 0.1234 | 0.0546 | 0.0547 | 0.0531 |
| 0.1000 | 0.0476 | 0.0477 | 0.1529 | 0.0547 | 0.0546 | 0.0531 |
| 0.1200 | 0.0821 | 0.0822 | 0.1816 | 0.0549 | 0.0549 | 0.0531 |
| 0.1400 | 0.1168 | 0.1169 | 0.2092 | 0.0562 | 0.0562 | 0.0531 |
| 0.1600 | 0.1501 | 0.1502 | 0.2341 | 0.0577 | 0.0578 | 0.0531 |

**表 C：对比均值平滑**（ε_in = ContRS 的平均 R；检测/重建为逐坐标期望平滑 Salman et al. 2019，检索为 Smoothed Embeddings, Pautov et al. 2022）

| r（ContRS） | ContRS R = 均值平滑 ε_in | 均值平滑 ε_out | 平滑误差 ContRS | 平滑误差 均值平滑 | 均值平滑在同一 r 下的 R | ContRS R |
|---|---|---|---|---|---|---|
| 0.0800 | 0.017 | 0.4139 | 0.0546 | 0.0537 | 0.0025 | 0.0168 |
| 0.1000 | 0.048 | 0.9041 | 0.0547 | 0.0537 | 0.0033 | 0.0476 |
| 0.1200 | 0.082 | 0.9996 | 0.0549 | 0.0537 | 0.0041 | 0.0821 |
| 0.1400 | 0.117 | 1.0000 | 0.0562 | 0.0537 | 0.0050 | 0.1168 |
| 0.1600 | 0.150 | 1.0000 | 0.0577 | 0.0537 | 0.0058 | 0.1501 |

**表 B：对比 α-平滑**（同一容差 r；α=0.35、P=0.9、n=10 为官方默认，置信度与 ContRS 一致）

| r | ContRS R | α-平滑 R | 平滑误差 ContRS | 平滑误差 α-平滑 |
|---|---|---|---|---|
| 0.0800 | 0.0168 | 0.0000 | 0.0546 | 0.0554 |
| 0.1000 | 0.0476 | 0.0000 | 0.0547 | 0.0554 |
| 0.1200 | 0.0821 | 0.0001 | 0.0549 | 0.0554 |
| 0.1400 | 0.1168 | 0.0005 | 0.0562 | 0.0554 |
| 0.1600 | 0.1501 | 0.0009 | 0.0577 | 0.0554 |

#### 检索（cosine）：clip_l14_300（σ=0.25，300 张图）

**表 A：对比中位数平滑**（中位数平滑的 ε_in = ContRS 的平均 R）

| r（ContRS） | ContRS R = 中位数平滑 ε_in | 中位数平滑 ε_out | 平滑误差 ContRS | 平滑误差 中位数平滑 | 中位数平滑在同一 r 下的 R |
|---|---|---|---|---|---|
| 0.0800 | 0.096 | 0.0037 | 0.2356 | 0.2301 | 0.4858 |
| 0.0880 | 0.134 | 0.0067 | 0.2350 | 0.2301 | 0.5055 |
| 0.1000 | 0.197 | 0.0137 | 0.2349 | 0.2301 | 0.5323 |
| 0.1080 | 0.240 | 0.0201 | 0.2355 | 0.2301 | 0.5484 |
| 0.1200 | 0.303 | 0.0322 | 0.2356 | 0.2301 | 0.5703 |

**表 A'：同一容差 r 下的认证半径**（ContRS-M = 候选中心加入逐坐标中位数/均值，证书不变）

| r | ContRS R | ContRS-M R | 中位数平滑 R | 平滑误差 ContRS | 平滑误差 ContRS-M | 平滑误差 中位数平滑 |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0961 | 0.2203 | 0.4858 | 0.2356 | 0.2301 | 0.2301 |
| 0.0880 | 0.1344 | 0.2704 | 0.5055 | 0.2350 | 0.2302 | 0.2301 |
| 0.1000 | 0.1967 | 0.3404 | 0.5323 | 0.2349 | 0.2302 | 0.2301 |
| 0.1080 | 0.2397 | 0.3841 | 0.5484 | 0.2355 | 0.2303 | 0.2301 |
| 0.1200 | 0.3034 | 0.4426 | 0.5703 | 0.2356 | 0.2304 | 0.2301 |

**表 C：对比均值平滑**（ε_in = ContRS 的平均 R；检测/重建为逐坐标期望平滑 Salman et al. 2019，检索为 Smoothed Embeddings, Pautov et al. 2022）

| r（ContRS） | ContRS R = 均值平滑 ε_in | 均值平滑 ε_out | 平滑误差 ContRS | 平滑误差 均值平滑 | 均值平滑在同一 r 下的 R | ContRS R |
|---|---|---|---|---|---|---|
| 0.0800 | 0.096 | 0.0653 | 0.2356 | 0.2302 | 0.1073 | 0.0961 |
| 0.0880 | 0.134 | 0.1227 | 0.2350 | 0.2302 | 0.1129 | 0.1344 |
| 0.1000 | 0.197 | 0.2690 | 0.2349 | 0.2302 | 0.1208 | 0.1967 |
| 0.1080 | 0.240 | 0.4308 | 0.2355 | 0.2302 | 0.1258 | 0.2397 |
| 0.1200 | 0.303 | 2.0000 | 0.2356 | 0.2302 | 0.1329 | 0.3034 |

**表 B：对比 α-平滑**（同一容差 r；α=0.35、P=0.9、n=10 为官方默认，置信度与 ContRS 一致）

| r | ContRS R | α-平滑 R | 平滑误差 ContRS | 平滑误差 α-平滑 |
|---|---|---|---|---|
| 0.0800 | 0.0961 | 0.0000 | 0.2356 | 0.2318 |
| 0.0880 | 0.1344 | 0.0000 | 0.2350 | 0.2318 |
| 0.1000 | 0.1967 | 0.0000 | 0.2349 | 0.2318 |
| 0.1080 | 0.2397 | 0.0000 | 0.2355 | 0.2318 |
| 0.1200 | 0.3034 | 0.0000 | 0.2356 | 0.2318 |

#### 重建（像素 ℓ∞；LPIPS 下两个 baseline 均无法认证）：cifar_vae300_linf（σ=0.25，300 张图）

**表 A：对比中位数平滑**（中位数平滑的 ε_in = ContRS 的平均 R）

| r（ContRS） | ContRS R = 中位数平滑 ε_in | 中位数平滑 ε_out | 平滑误差 ContRS | 平滑误差 中位数平滑 | 中位数平滑在同一 r 下的 R |
|---|---|---|---|---|---|
| 0.2852 | 0.090 | 0.0273 | 0.0880 | 0.0468 | 0.7282 |
| 0.3097 | 0.152 | 0.0435 | 0.0883 | 0.0468 | 0.7314 |
| 0.3364 | 0.238 | 0.0663 | 0.0897 | 0.0468 | 0.7326 |
| 0.3653 | 0.341 | 0.0941 | 0.0910 | 0.0468 | 0.7327 |
| 0.3967 | 0.444 | 0.1232 | 0.0928 | 0.0468 | 0.7328 |

**表 A'：同一容差 r 下的认证半径**（ContRS-M = 候选中心加入逐坐标中位数/均值，证书不变）

| r | ContRS R | ContRS-M R | 中位数平滑 R | 平滑误差 ContRS | 平滑误差 ContRS-M | 平滑误差 中位数平滑 |
|---|---|---|---|---|---|---|
| 0.2852 | 0.0895 | 0.1875 | 0.7282 | 0.0880 | 0.0478 | 0.0468 |
| 0.3097 | 0.1517 | 0.2805 | 0.7314 | 0.0883 | 0.0479 | 0.0468 |
| 0.3364 | 0.2381 | 0.3773 | 0.7326 | 0.0897 | 0.0487 | 0.0468 |
| 0.3653 | 0.3409 | 0.4683 | 0.7327 | 0.0910 | 0.0493 | 0.0468 |
| 0.3967 | 0.4440 | 0.5524 | 0.7328 | 0.0928 | 0.0509 | 0.0468 |

**表 C：对比均值平滑**（ε_in = ContRS 的平均 R；检测/重建为逐坐标期望平滑 Salman et al. 2019，检索为 Smoothed Embeddings, Pautov et al. 2022）

| r（ContRS） | ContRS R = 均值平滑 ε_in | 均值平滑 ε_out | 平滑误差 ContRS | 平滑误差 均值平滑 | 均值平滑在同一 r 下的 R | ContRS R |
|---|---|---|---|---|---|---|
| 0.2852 | 0.090 | 0.2942 | 0.0880 | 0.0472 | 0.0866 | 0.0895 |
| 0.3097 | 0.152 | 0.4868 | 0.0883 | 0.0472 | 0.0945 | 0.1517 |
| 0.3364 | 0.238 | 0.7417 | 0.0897 | 0.0472 | 0.1030 | 0.2381 |
| 0.3653 | 0.341 | 1.0184 | 0.0910 | 0.0472 | 0.1123 | 0.3409 |
| 0.3967 | 0.444 | 1.2585 | 0.0928 | 0.0472 | 0.1224 | 0.4440 |

**表 B：对比 α-平滑**（同一容差 r；α=0.35、P=0.9、n=10 为官方默认，置信度与 ContRS 一致）

| r | ContRS R | α-平滑 R | 平滑误差 ContRS | 平滑误差 α-平滑 |
|---|---|---|---|---|
| 0.2852 | 0.0895 | 0.0535 | 0.0880 | 0.0646 |
| 0.3097 | 0.1517 | 0.1023 | 0.0883 | 0.0646 |
| 0.3364 | 0.2381 | 0.1711 | 0.0897 | 0.0646 |
| 0.3653 | 0.3409 | 0.2563 | 0.0910 | 0.0646 |
| 0.3967 | 0.4440 | 0.3461 | 0.0928 | 0.0646 |


---

# 第三部分：多架构与预处理对照完整结果（每组 50 图）

核心模型为 MTCNN、RetinaFace-R50、DETR-R50、CIFAR-VAE、SD-VAE、MAE ViT-B、CLIP ViT-L/14、CLIP RN50x16、SigLIP ViT-B/16。另有 RN50x16/SigLIP 官方预处理对照和 VAE 像素 ℓ∞ 对照。不要把本部分写成 300 图。


## 3.1 论文格式：九个核心模型及对照的 center/safe 表（50 图）

所有 safe-region 组均值半径均已完成精确补跑；源表中的星号说明只是通用模板提示。LPIPS/cosine 的 `valid` 系数为 10，`coef 3` 仅复现投稿旧口径。

以下来源表已完整内嵌：[rebuttal/summary/paper_format.md](rebuttal/summary/paper_format.md)。

#### Paper-format tables (Fig.1 protocol)

##### vs center smoothing: ContRS certified R is used as center smoothing's ε_in; compare ContRS's r with center smoothing's ε_out, and both smoothing errors

ε_out(valid) uses center smoothing's own constant (3 for Jaccard, γ(1+2γ)=10 for LPIPS/cosine); ε_out(coef 3) is the constant used in the submitted Fig.1.


**mtcnn** (CNN (cascade), σ=0.1)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.08 | 0.054 | 0.1669 | 0.1669 | 0.0503 | 0.0504 |
| 0.1 | 0.086 | 0.2092 | 0.2092 | 0.0532 | 0.0504 |
| 0.12 | 0.124 | 0.2585 | 0.2585 | 0.0534 | 0.0504 |
| 0.14 | 0.163 | 0.3215 | 0.3215 | 0.0540 | 0.0504 |
| 0.16 | 0.194 | 0.3982 | 0.3982 | 0.0528 | 0.0504 |

**retinaface** (CNN (ResNet-50), σ=0.1)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.02975 | 0.037 | 0.0631 | 0.0631 | 0.0446 | 0.0443 |
| 0.03439 | 0.064 | 0.0702 | 0.0702 | 0.0446 | 0.0443 |
| 0.03976 | 0.096 | 0.0796 | 0.0796 | 0.0448 | 0.0443 |
| 0.04596 | 0.132 | 0.0921 | 0.0921 | 0.0449 | 0.0443 |
| 0.05313 | 0.168 | 0.1097 | 0.1097 | 0.0447 | 0.0443 |

**detr** (Transformer, σ=0.1)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.0134 | 0.031 | 0.4629 | 0.4629 | 0.1631 | 0.1717 |
| 0.0207 | 0.054 | 0.5956 | 0.5956 | 0.1460 | 0.1717 |
| 0.03199 | 0.098 | 0.8203 | 0.8203 | 0.1413 | 0.1717 |
| 0.04596 | 0.139 | 0.8619 | 0.8619 | 0.1415 | 0.1717 |
| 0.071 | 0.179 | 0.9709 | 0.9709 | 0.1417 | 0.1717 |

**cifar_vae** (CNN (VAE), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.012 | 0.198 | 0.0394 | 0.0118 | 0.0013 | 0.0013 |
| 0.014 | 0.264 | 0.0443 | 0.0133 | 0.0013 | 0.0013 |
| 0.016 | 0.322 | 0.0496 | 0.0149 | 0.0013 | 0.0013 |
| 0.018 | 0.376 | 0.0556 | 0.0167 | 0.0013 | 0.0013 |
| 0.02 | 0.433 | 0.0645 | 0.0194 | 0.0014 | 0.0013 |

**sd_vae** (CNN (SD-VAE), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.0595 | 0.089 | 0.2222 | 0.0667 | 0.0264 | 0.0253 |
| 0.06878 | 0.143 | 0.2360 | 0.0708 | 0.0252 | 0.0253 |
| 0.08549 | 0.249 | 0.2691 | 0.0807 | 0.0270 | 0.0253 |
| 0.09883 | 0.333 | 0.3037 | 0.0911 | 0.0270 | 0.0253 |
| 0.1143 | 0.421 | 0.3567 | 0.1070 | 0.0274 | 0.0253 |

**mae** (Transformer (ViT-MAE-B), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.0595 | 0.091 | 0.1927 | 0.0578 | 0.0294 | 0.0284 |
| 0.06878 | 0.163 | 0.2071 | 0.0621 | 0.0295 | 0.0284 |
| 0.07395 | 0.202 | 0.2159 | 0.0648 | 0.0293 | 0.0284 |
| 0.08549 | 0.306 | 0.2437 | 0.0731 | 0.0300 | 0.0284 |
| 0.09883 | 0.416 | 0.2884 | 0.0865 | 0.0303 | 0.0284 |

**clip_l14** (Transformer (ViT-L/14), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.08 | 0.080 | 0.2853 | 0.0856 | 0.2491 | 0.2464 |
| 0.088 | 0.109 | 0.2954 | 0.0886 | 0.2472 | 0.2464 |
| 0.1 | 0.155 | 0.3128 | 0.0938 | 0.2462 | 0.2464 |
| 0.108 | 0.189 | 0.3261 | 0.0978 | 0.2462 | 0.2464 |
| 0.12 | 0.242 | 0.3499 | 0.1050 | 0.2465 | 0.2464 |

**clip_rn50x16** (CNN (ResNet-50x16), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.06397 | 0.079 | 0.2467 | 0.0740 | 0.2257 | 0.2257 |
| 0.08 | 0.149 | 0.2691 | 0.0807 | 0.2244 | 0.2257 |
| 0.09883 | 0.242 | 0.3066 | 0.0920 | 0.2260 | 0.2257 |
| 0.1143 | 0.321 | 0.3472 | 0.1042 | 0.2268 | 0.2257 |
| 0.142 | 0.443 | 0.4433 | 0.1330 | 0.2278 | 0.2257 |

**siglip** (Transformer (SigLIP-B/16), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.07951 | 0.077 | 0.2638 | 0.0791 | 0.2603 | 0.2625 |
| 0.088 | 0.114 | 0.2745 | 0.0823 | 0.2618 | 0.2625 |
| 0.1 | 0.170 | 0.2925 | 0.0878 | 0.2626 | 0.2625 |
| 0.12 | 0.275 | 0.3328 | 0.0998 | 0.2630 | 0.2625 |
| 0.142 | 0.402 | 0.4067 | 0.1220 | 0.2641 | 0.2625 |

**cifar_vae_linf** (CNN (VAE), pixel l_inf, σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.2852 | 0.089 | 0.4712 | 0.4712 | 0.0904 | 0.0897 |
| 0.3097 | 0.148 | 0.4916 | 0.4916 | 0.0900 | 0.0897 |
| 0.3364 | 0.226 | 0.5214 | 0.5214 | 0.0920 | 0.0897 |
| 0.3653 | 0.327 | 0.5685 | 0.5685 | 0.0951 | 0.0897 |
| 0.3967 | 0.431 | 0.6350 | 0.6350 | 0.0955 | 0.0897 |

**clip_rn50x16_off** (CNN (ResNet-50x16), official preprocessing, σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.0595 | 0.081 | 0.2367 | 0.0710 | 0.2342 | 0.2357 |
| 0.07951 | 0.178 | 0.2664 | 0.0799 | 0.2350 | 0.2357 |
| 0.09192 | 0.237 | 0.2897 | 0.0869 | 0.2361 | 0.2357 |
| 0.108 | 0.313 | 0.3269 | 0.0981 | 0.2371 | 0.2357 |
| 0.1321 | 0.428 | 0.4099 | 0.1230 | 0.2390 | 0.2357 |

**siglip_off** (Transformer (SigLIP-B/16), official preprocessing, σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS (valid) | ε_out CS (coef 3) | smooth err ContRS | smooth err CS |
|---|---|---|---|---|---|
| 0.06878 | 0.099 | 0.2174 | 0.0652 | 0.1698 | 0.1712 |
| 0.08 | 0.173 | 0.2364 | 0.0709 | 0.1704 | 0.1712 |
| 0.09192 | 0.251 | 0.2600 | 0.0780 | 0.1695 | 0.1712 |
| 0.108 | 0.358 | 0.3037 | 0.0911 | 0.1710 | 0.1712 |
| 0.1228 | 0.447 | 0.3664 | 0.1099 | 0.1713 | 0.1712 |

##### vs safe region: run safe region (p=0.55) first, set ContRS r = 2κ·mean(r_SR) at each σ; compare R and smoothing errors

(* = evaluated at the nearest lower grid point; replaced automatically by the exact rerun once it finishes)


**mtcnn**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0724 | 0.0447 | 0.0089 | 0.0501 | 0.0485 |
| 0.2 | 0.0923 | 0.0884 | 0.0181 | 0.0595 | 0.0577 |
| 0.3 | 0.1828 | 0.4483 | 0.0642 | 0.0763 | 0.1054 |
| 0.4 | 0.3418 | 0.9364 | 0.1867 | 0.0915 | 0.1735 |
| 0.5 | 0.5048 | 1.1223 | 0.3607 | 0.1286 | 0.2747 |

**retinaface**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0338 | 0.0604 | 0.0091 | 0.0446 | 0.0445 |
| 0.2 | 0.0512 | 0.1166 | 0.0175 | 0.0706 | 0.0700 |
| 0.3 | 0.1800 | 0.7824 | 0.0827 | 0.1055 | 0.1383 |
| 0.4 | 0.3145 | 1.0116 | 0.1861 | 0.1843 | 0.1930 |
| 0.5 | 0.5628 | 1.0639 | 0.4263 | 0.2291 | 0.3394 |

**detr**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.1918 | 0.2116 | 0.0341 | 0.1521 | 0.1718 |
| 0.2 | 0.2972 | 0.3416 | 0.0806 | 0.2139 | 0.2572 |
| 0.3 | 0.4951 | 0.4104 | 0.1789 | 0.3276 | 0.3921 |
| 0.4 | 0.6173 | 0.4395 | 0.3150 | 0.3937 | 0.4575 |
| 0.5 | 0.9256 | 0.4154 | 0.6185 | 0.5353 | 0.5993 |

**cifar_vae**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0020 | 0.0688 | 0.0092 | 0.0001 | 0.0001 |
| 0.2 | 0.0074 | 0.1371 | 0.0191 | 0.0007 | 0.0007 |
| 0.3 | 0.0158 | 0.2030 | 0.0278 | 0.0020 | 0.0021 |
| 0.4 | 0.0268 | 0.2669 | 0.0383 | 0.0039 | 0.0041 |
| 0.5 | 0.0398 | 0.3192 | 0.0470 | 0.0068 | 0.0071 |

**sd_vae**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0302 | 0.1033 | 0.0095 | 0.0077 | 0.0068 |
| 0.2 | 0.0640 | 0.1849 | 0.0187 | 0.0191 | 0.0177 |
| 0.3 | 0.0955 | 0.2431 | 0.0277 | 0.0360 | 0.0343 |
| 0.4 | 0.1225 | 0.2837 | 0.0381 | 0.0581 | 0.0552 |
| 0.5 | 0.1465 | 0.3102 | 0.0453 | 0.0766 | 0.0747 |

**mae**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0229 | 0.0774 | 0.0088 | 0.0052 | 0.0051 |
| 0.2 | 0.0547 | 0.1468 | 0.0187 | 0.0194 | 0.0190 |
| 0.3 | 0.0853 | 0.2016 | 0.0278 | 0.0409 | 0.0407 |
| 0.4 | 0.1136 | 0.2421 | 0.0359 | 0.0679 | 0.0677 |
| 0.5 | 0.1381 | 0.2743 | 0.0451 | 0.0941 | 0.0954 |

**clip_l14**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0637 | 0.0783 | 0.0091 | 0.0914 | 0.0922 |
| 0.2 | 0.0967 | 0.1338 | 0.0186 | 0.1919 | 0.1934 |
| 0.3 | 0.1063 | 0.2053 | 0.0269 | 0.2907 | 0.2889 |
| 0.4 | 0.1152 | 0.2543 | 0.0374 | 0.3518 | 0.3517 |
| 0.5 | 0.1191 | 0.2529 | 0.0466 | 0.4022 | 0.4022 |

**clip_rn50x16**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0668 | 0.0826 | 0.0092 | 0.0971 | 0.0959 |
| 0.2 | 0.0871 | 0.1558 | 0.0181 | 0.1822 | 0.1798 |
| 0.3 | 0.0891 | 0.2182 | 0.0276 | 0.2734 | 0.2723 |
| 0.4 | 0.0862 | 0.2649 | 0.0371 | 0.3512 | 0.3515 |
| 0.5 | 0.0838 | 0.3266 | 0.0450 | 0.4119 | 0.4139 |

**siglip**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0629 | 0.0822 | 0.0091 | 0.0866 | 0.0871 |
| 0.2 | 0.0900 | 0.1300 | 0.0178 | 0.2058 | 0.2052 |
| 0.3 | 0.0984 | 0.1586 | 0.0274 | 0.3086 | 0.3076 |
| 0.4 | 0.1047 | 0.1712 | 0.0373 | 0.3826 | 0.3824 |
| 0.5 | 0.1044 | 0.2375 | 0.0441 | 0.4411 | 0.4415 |

**clip_rn50x16_off**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0618 | 0.0902 | 0.0094 | 0.1076 | 0.1058 |
| 0.2 | 0.0807 | 0.1695 | 0.0185 | 0.1926 | 0.1922 |
| 0.3 | 0.0874 | 0.2462 | 0.0282 | 0.2742 | 0.2754 |
| 0.4 | 0.0847 | 0.2959 | 0.0357 | 0.3513 | 0.3526 |
| 0.5 | 0.0815 | 0.3296 | 0.0461 | 0.4111 | 0.4123 |

**siglip_off**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.1 | 0.0420 | 0.0789 | 0.0092 | 0.0654 | 0.0653 |
| 0.2 | 0.0697 | 0.1373 | 0.0193 | 0.1297 | 0.1299 |
| 0.3 | 0.0837 | 0.1714 | 0.0274 | 0.2122 | 0.2113 |
| 0.4 | 0.0921 | 0.1900 | 0.0363 | 0.2855 | 0.2838 |
| 0.5 | 0.0992 | 0.1872 | 0.0446 | 0.3442 | 0.3458 |


## 3.2 完整汇总：n、center 弃权、两种 safe 匹配与旧 joint 诊断（50 图）

完整表用于核查覆盖范围与派生字段。`joint` 结果不是正式有效证书，不能用来替代 split ContRS。

以下来源表已完整内嵌：[rebuttal/summary/tables.md](rebuttal/summary/tables.md)。

#### Multi-architecture results (auto-generated)

##### Completion (images per sigma)

| model | σ=0.1 | σ=0.2 | σ=0.25 | σ=0.3 | σ=0.4 | σ=0.5 |
|---|---|---|---|---|---|---|
| mtcnn | 50 | 50 | 50 | 50 | 50 | 50 |
| retinaface | 50 | 50 | 50 | 50 | 50 | 50 |
| detr | 50 | 50 | 50 | 50 | 50 | 50 |
| cifar_vae | 50 | 50 | 50 | 50 | 50 | 50 |
| sd_vae | 50 | 50 | 50 | 50 | 50 | 50 |
| mae | 50 | 50 | 50 | 50 | 50 | 50 |
| clip_l14 | 50 | 50 | 50 | 50 | 50 | 50 |
| clip_rn50x16 | 50 | 50 | 50 | 50 | 50 | 50 |
| siglip | 50 | 50 | 50 | 50 | 50 | 50 |
| cifar_vae_linf | 0 | 0 | 50 | 0 | 0 | 0 |
| clip_rn50x16_off | 50 | 50 | 50 | 50 | 50 | 50 |
| siglip_off | 50 | 50 | 50 | 50 | 50 | 50 |

##### Clean task quality of each base model (σ-independent)

| model | family | task | clean metric |
|---|---|---|---|
| mtcnn | CNN (cascade) | det | IoU with GT box: mean 0.639, IoU>0.5 for 92% |
| retinaface | CNN (ResNet-50) | det | IoU with GT box: mean 0.654, IoU>0.5 for 96% |
| detr | Transformer | det | IoU with GT box: mean 0.809, IoU>0.5 for 86% |
| cifar_vae | CNN (VAE) | recon | LPIPS(f(x), x): mean 0.1810 |
| sd_vae | CNN (SD-VAE) | recon | LPIPS(f(x), x): mean 0.0175 |
| mae | Transformer (ViT-MAE-B) | recon | LPIPS(f(x), x): mean 0.0135 |
| clip_l14 | Transformer (ViT-L/14) | retr | embedding model (Recall not computed here) |
| clip_rn50x16 | CNN (ResNet-50x16) | retr | embedding model (Recall not computed here) |
| siglip | Transformer (SigLIP-B/16) | retr | embedding model (Recall not computed here) |
| cifar_vae_linf | CNN (VAE), pixel l_inf | recon | embedding model (Recall not computed here) |
| clip_rn50x16_off | CNN (ResNet-50x16), official preprocessing | retr | embedding model (Recall not computed here) |
| siglip_off | Transformer (SigLIP-B/16), official preprocessing | retr | embedding model (Recall not computed here) |

##### ContRS vs center smoothing (input radius matched: ε_in = mean ContRS R)

ρ_CS: center smoothing certified output bound at ε_in, averaged over images where it does not abstain (abstain = failed center check or quantile > 1). 'valid' uses the coefficient required by center smoothing's theory (3 for the Jaccard metric, γ(1+2γ)=10 for γ=2 relaxed-triangle LPIPS/cosine); 'coef 3' reproduces the constant used in the paper's Fig.1. ContRS output bound is the prescribed r. Lower is tighter.


**mtcnn** (CNN (cascade), σ=0.1, n=50, r grid: paper grid)

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.08 | 0.054 | 66% | 0.0503 | 0.1669 | 2.09 | 0.1669 | 2.09 | 4% | 0.0504 |
| 0.1 | 0.086 | 88% | 0.0532 | 0.2092 | 2.09 | 0.2092 | 2.09 | 4% | 0.0504 |
| 0.12 | 0.124 | 96% | 0.0534 | 0.2585 | 2.15 | 0.2585 | 2.15 | 4% | 0.0504 |
| 0.14 | 0.163 | 98% | 0.0540 | 0.3215 | 2.30 | 0.3215 | 2.30 | 4% | 0.0504 |
| 0.16 | 0.194 | 98% | 0.0528 | 0.3982 | 2.49 | 0.3982 | 2.49 | 4% | 0.0504 |

**retinaface** (CNN (ResNet-50), σ=0.1, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.02975 | 0.037 | 60% | 0.0446 | 0.0631 | 2.12 | 0.0631 | 2.12 | 2% | 0.0443 |
| 0.03439 | 0.064 | 74% | 0.0446 | 0.0702 | 2.04 | 0.0702 | 2.04 | 2% | 0.0443 |
| 0.03976 | 0.096 | 84% | 0.0448 | 0.0796 | 2.00 | 0.0796 | 2.00 | 2% | 0.0443 |
| 0.04596 | 0.132 | 92% | 0.0449 | 0.0921 | 2.00 | 0.0921 | 2.00 | 2% | 0.0443 |
| 0.05313 | 0.168 | 92% | 0.0447 | 0.1097 | 2.06 | 0.1097 | 2.06 | 2% | 0.0443 |

**detr** (Transformer, σ=0.1, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.0134 | 0.031 | 20% | 0.1631 | 0.4629 | 34.54 | 0.4629 | 34.54 | 0% | 0.1717 |
| 0.0207 | 0.054 | 40% | 0.1460 | 0.5956 | 28.77 | 0.5956 | 28.77 | 0% | 0.1717 |
| 0.03199 | 0.098 | 64% | 0.1413 | 0.8203 | 25.65 | 0.8203 | 25.65 | 0% | 0.1717 |
| 0.04596 | 0.139 | 74% | 0.1415 | 0.8619 | 18.75 | 0.8619 | 18.75 | 0% | 0.1717 |
| 0.071 | 0.179 | 82% | 0.1417 | 0.9709 | 13.67 | 0.9709 | 13.67 | 0% | 0.1717 |

**cifar_vae** (CNN (VAE), σ=0.25, n=50, r grid: paper grid)

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.012 | 0.198 | 68% | 0.0013 | 0.0394 | 3.29 | 0.0118 | 0.99 | 0% | 0.0013 |
| 0.014 | 0.264 | 74% | 0.0013 | 0.0443 | 3.17 | 0.0133 | 0.95 | 0% | 0.0013 |
| 0.016 | 0.322 | 80% | 0.0013 | 0.0496 | 3.10 | 0.0149 | 0.93 | 0% | 0.0013 |
| 0.018 | 0.376 | 94% | 0.0013 | 0.0556 | 3.09 | 0.0167 | 0.93 | 0% | 0.0013 |
| 0.02 | 0.433 | 94% | 0.0014 | 0.0645 | 3.23 | 0.0194 | 0.97 | 0% | 0.0013 |

**sd_vae** (CNN (SD-VAE), σ=0.25, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.0595 | 0.089 | 40% | 0.0264 | 0.2222 | 3.73 | 0.0667 | 1.12 | 0% | 0.0253 |
| 0.06878 | 0.143 | 50% | 0.0252 | 0.2360 | 3.43 | 0.0708 | 1.03 | 0% | 0.0253 |
| 0.08549 | 0.249 | 66% | 0.0270 | 0.2691 | 3.15 | 0.0807 | 0.94 | 0% | 0.0253 |
| 0.09883 | 0.333 | 78% | 0.0270 | 0.3037 | 3.07 | 0.0911 | 0.92 | 0% | 0.0253 |
| 0.1143 | 0.421 | 84% | 0.0274 | 0.3567 | 3.12 | 0.1070 | 0.94 | 0% | 0.0253 |

**mae** (Transformer (ViT-MAE-B), σ=0.25, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.0595 | 0.091 | 48% | 0.0294 | 0.1927 | 3.24 | 0.0578 | 0.97 | 0% | 0.0284 |
| 0.06878 | 0.163 | 58% | 0.0295 | 0.2071 | 3.01 | 0.0621 | 0.90 | 0% | 0.0284 |
| 0.07395 | 0.202 | 60% | 0.0293 | 0.2159 | 2.92 | 0.0648 | 0.88 | 0% | 0.0284 |
| 0.08549 | 0.306 | 84% | 0.0300 | 0.2437 | 2.85 | 0.0731 | 0.86 | 0% | 0.0284 |
| 0.09883 | 0.416 | 90% | 0.0303 | 0.2884 | 2.92 | 0.0865 | 0.88 | 0% | 0.0284 |

**clip_l14** (Transformer (ViT-L/14), σ=0.25, n=50, r grid: paper grid)

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.08 | 0.080 | 34% | 0.2491 | 0.2853 | 3.57 | 0.0856 | 1.07 | 0% | 0.2464 |
| 0.088 | 0.109 | 38% | 0.2472 | 0.2954 | 3.36 | 0.0886 | 1.01 | 0% | 0.2464 |
| 0.1 | 0.155 | 50% | 0.2462 | 0.3128 | 3.13 | 0.0938 | 0.94 | 0% | 0.2464 |
| 0.108 | 0.189 | 56% | 0.2462 | 0.3261 | 3.02 | 0.0978 | 0.91 | 0% | 0.2464 |
| 0.12 | 0.242 | 74% | 0.2465 | 0.3499 | 2.92 | 0.1050 | 0.87 | 0% | 0.2464 |

**clip_rn50x16** (CNN (ResNet-50x16), σ=0.25, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.06397 | 0.079 | 34% | 0.2257 | 0.2467 | 3.86 | 0.0740 | 1.16 | 0% | 0.2257 |
| 0.08 | 0.149 | 42% | 0.2244 | 0.2691 | 3.36 | 0.0807 | 1.01 | 0% | 0.2257 |
| 0.09883 | 0.242 | 68% | 0.2260 | 0.3066 | 3.10 | 0.0920 | 0.93 | 0% | 0.2257 |
| 0.1143 | 0.321 | 80% | 0.2268 | 0.3472 | 3.04 | 0.1042 | 0.91 | 0% | 0.2257 |
| 0.142 | 0.443 | 88% | 0.2278 | 0.4433 | 3.12 | 0.1330 | 0.94 | 0% | 0.2257 |

**siglip** (Transformer (SigLIP-B/16), σ=0.25, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.07951 | 0.077 | 34% | 0.2603 | 0.2638 | 3.32 | 0.0791 | 1.00 | 0% | 0.2625 |
| 0.088 | 0.114 | 44% | 0.2618 | 0.2745 | 3.12 | 0.0823 | 0.94 | 0% | 0.2625 |
| 0.1 | 0.170 | 58% | 0.2626 | 0.2925 | 2.93 | 0.0878 | 0.88 | 0% | 0.2625 |
| 0.12 | 0.275 | 78% | 0.2630 | 0.3328 | 2.77 | 0.0998 | 0.83 | 0% | 0.2625 |
| 0.142 | 0.402 | 96% | 0.2641 | 0.4067 | 2.86 | 0.1220 | 0.86 | 0% | 0.2625 |

**cifar_vae_linf** (CNN (VAE), pixel l_inf, σ=0.25, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.2852 | 0.089 | 40% | 0.0904 | 0.4712 | 1.65 | 0.4712 | 1.65 | 0% | 0.0897 |
| 0.3097 | 0.148 | 62% | 0.0900 | 0.4916 | 1.59 | 0.4916 | 1.59 | 0% | 0.0897 |
| 0.3364 | 0.226 | 84% | 0.0920 | 0.5214 | 1.55 | 0.5214 | 1.55 | 0% | 0.0897 |
| 0.3653 | 0.327 | 100% | 0.0951 | 0.5685 | 1.56 | 0.5685 | 1.56 | 0% | 0.0897 |
| 0.3967 | 0.431 | 100% | 0.0955 | 0.6350 | 1.60 | 0.6350 | 1.60 | 0% | 0.0897 |

**clip_rn50x16_off** (CNN (ResNet-50x16), official preprocessing, σ=0.25, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.0595 | 0.081 | 26% | 0.2342 | 0.2367 | 3.98 | 0.0710 | 1.19 | 2% | 0.2357 |
| 0.07951 | 0.178 | 50% | 0.2350 | 0.2664 | 3.35 | 0.0799 | 1.01 | 2% | 0.2357 |
| 0.09192 | 0.237 | 62% | 0.2361 | 0.2897 | 3.15 | 0.0869 | 0.95 | 2% | 0.2357 |
| 0.108 | 0.313 | 72% | 0.2371 | 0.3269 | 3.03 | 0.0981 | 0.91 | 2% | 0.2357 |
| 0.1321 | 0.428 | 92% | 0.2390 | 0.4099 | 3.10 | 0.1230 | 0.93 | 2% | 0.2357 |

**siglip_off** (Transformer (SigLIP-B/16), official preprocessing, σ=0.25, n=50, r grid: rule: mean R in [0.3σ, 1.8σ])

| r (ContRS bound) | mean R | certified | ContRS smooth err | ρ_CS valid | ρ_CS valid / r | ρ_CS coef 3 | ρ_CS coef 3 / r | CS abstain | CS smooth err |
|---|---|---|---|---|---|---|---|---|---|
| 0.06878 | 0.099 | 48% | 0.1698 | 0.2174 | 3.16 | 0.0652 | 0.95 | 0% | 0.1712 |
| 0.08 | 0.173 | 62% | 0.1704 | 0.2364 | 2.96 | 0.0709 | 0.89 | 0% | 0.1712 |
| 0.09192 | 0.251 | 78% | 0.1695 | 0.2600 | 2.83 | 0.0780 | 0.85 | 0% | 0.1712 |
| 0.108 | 0.358 | 90% | 0.1710 | 0.3037 | 2.81 | 0.0911 | 0.84 | 0% | 0.1712 |
| 0.1228 | 0.447 | 96% | 0.1713 | 0.3664 | 2.98 | 0.1099 | 0.89 | 0% | 0.1712 |

##### ContRS vs safe region (p = 0.55) at matched output-ball radius

(a) per-image matching s = r_SR (Appendix I as written); (b) one radius per σ = mean r_SR over the 50 images (Fig.1 top style), evaluated exactly at that radius by a rerun with identical seeds (marked * when only the nearest-lower grid point is available yet); joint = old joint certifier on the same pools (diagnostic only, not a valid certificate). Safe-region σ grid follows Table 4.


**mtcnn**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0724 | 0.0089 | 0.0088 | 22%/44%/34% | 0.0447 | 0.0085 | 0.2593 | 0.0485 | 0.0488 |
| 0.2 | 50 | 0.0923 | 0.0181 | 0.0183 | 42%/38%/20% | 0.0884 | 0.0190 | 0.0760 | 0.0577 | 0.0586 |
| 0.3 | 50 | 0.1828 | 0.0642 | 0.0639 | 32%/34%/34% | 0.4483 | 0.0657 | 0.4523 | 0.1054 | 0.1072 |
| 0.4 | 50 | 0.3418 | 0.1867 | 0.1864 | 30%/44%/26% | 0.9364 | 0.1880 | 0.9284 | 0.1735 | 0.1732 |
| 0.5 | 50 | 0.5048 | 0.3607 | 0.3609 | 32%/44%/24% | 1.1223 | 0.3628 | 1.1344 | 0.2747 | 0.2750 |

**retinaface**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0338 | 0.0091 | 0.0089 | 26%/32%/42% | 0.0604 | 0.0094 | 0.0504 | 0.0445 | 0.0449 |
| 0.2 | 50 | 0.0512 | 0.0175 | 0.0172 | 30%/40%/30% | 0.1166 | 0.0183 | 0.1035 | 0.0700 | 0.0695 |
| 0.3 | 50 | 0.1800 | 0.0827 | 0.0827 | 34%/40%/26% | 0.7824 | 0.0838 | 0.7989 | 0.1383 | 0.1385 |
| 0.4 | 50 | 0.3145 | 0.1861 | 0.1855 | 20%/48%/32% | 1.0116 | 0.1871 | 1.0146 | 0.1930 | 0.1928 |
| 0.5 | 50 | 0.5628 | 0.4263 | 0.4258 | 20%/50%/30% | 1.0639 | 0.4277 | 1.0622 | 0.3394 | 0.3396 |

**detr**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.1918 | 0.0341 | 0.0342 | 38%/24%/38% | 0.2116 | 0.0346 | 0.2110 | 0.1718 | 0.1721 |
| 0.2 | 50 | 0.2972 | 0.0806 | 0.0806 | 34%/42%/24% | 0.3416 | 0.0813 | 0.3414 | 0.2572 | 0.2573 |
| 0.3 | 50 | 0.4951 | 0.1789 | 0.1783 | 20%/48%/32% | 0.4104 | 0.1794 | 0.4068 | 0.3921 | 0.3919 |
| 0.4 | 50 | 0.6173 | 0.3150 | 0.3150 | 22%/54%/24% | 0.4395 | 0.3160 | 0.4288 | 0.4575 | 0.4565 |
| 0.5 | 50 | 0.9256 | 0.6185 | 0.6178 | 8%/68%/24% | 0.4154 | 0.6193 | 0.4137 | 0.5993 | 0.6008 |

**cifar_vae**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0020 | 0.0092 | 0.0093 | 8%/86%/6% | 0.0688 | 0.0095 | 0.0603 | 0.0001 | 0.0001 |
| 0.2 | 50 | 0.0074 | 0.0191 | 0.0191 | 6%/88%/6% | 0.1371 | 0.0194 | 0.1343 | 0.0007 | 0.0007 |
| 0.3 | 50 | 0.0158 | 0.0278 | 0.0273 | 8%/74%/18% | 0.2030 | 0.0283 | 0.1814 | 0.0021 | 0.0020 |
| 0.4 | 50 | 0.0268 | 0.0383 | 0.0381 | 4%/92%/4% | 0.2669 | 0.0387 | 0.2262 | 0.0041 | 0.0041 |
| 0.5 | 50 | 0.0398 | 0.0470 | 0.0467 | 6%/88%/6% | 0.3192 | 0.0474 | 0.2997 | 0.0071 | 0.0071 |

**sd_vae**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0302 | 0.0095 | 0.0095 | 10%/86%/4% | 0.1033 | 0.0096 | 0.0956 | 0.0068 | 0.0068 |
| 0.2 | 50 | 0.0640 | 0.0187 | 0.0187 | 8%/84%/8% | 0.1849 | 0.0190 | 0.1862 | 0.0177 | 0.0178 |
| 0.3 | 50 | 0.0955 | 0.0277 | 0.0278 | 8%/80%/12% | 0.2431 | 0.0285 | 0.2192 | 0.0343 | 0.0351 |
| 0.4 | 50 | 0.1225 | 0.0381 | 0.0383 | 8%/88%/4% | 0.2837 | 0.0387 | 0.2310 | 0.0552 | 0.0546 |
| 0.5 | 50 | 0.1465 | 0.0453 | 0.0455 | 8%/90%/2% | 0.3102 | 0.0459 | 0.2809 | 0.0747 | 0.0749 |

**mae**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0229 | 0.0088 | 0.0088 | 4%/92%/4% | 0.0774 | 0.0089 | 0.0703 | 0.0051 | 0.0052 |
| 0.2 | 50 | 0.0547 | 0.0187 | 0.0184 | 4%/86%/10% | 0.1468 | 0.0189 | 0.1219 | 0.0190 | 0.0188 |
| 0.3 | 50 | 0.0853 | 0.0278 | 0.0276 | 6%/86%/8% | 0.2016 | 0.0290 | 0.1556 | 0.0407 | 0.0409 |
| 0.4 | 50 | 0.1136 | 0.0359 | 0.0356 | 12%/76%/12% | 0.2421 | 0.0366 | 0.1815 | 0.0677 | 0.0680 |
| 0.5 | 50 | 0.1381 | 0.0451 | 0.0445 | 6%/86%/8% | 0.2743 | 0.0458 | 0.2198 | 0.0954 | 0.0955 |

**clip_l14**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0637 | 0.0091 | 0.0091 | 10%/84%/6% | 0.0783 | 0.0092 | 0.0663 | 0.0922 | 0.0917 |
| 0.2 | 50 | 0.0967 | 0.0186 | 0.0187 | 4%/94%/2% | 0.1338 | 0.0187 | 0.1169 | 0.1934 | 0.1936 |
| 0.3 | 50 | 0.1063 | 0.0269 | 0.0266 | 4%/88%/8% | 0.2053 | 0.0270 | 0.1736 | 0.2889 | 0.2900 |
| 0.4 | 50 | 0.1152 | 0.0374 | 0.0369 | 2%/86%/12% | 0.2543 | 0.0379 | 0.2490 | 0.3517 | 0.3525 |
| 0.5 | 50 | 0.1191 | 0.0466 | 0.0460 | 8%/78%/14% | 0.2529 | 0.0476 | 0.2201 | 0.4022 | 0.4025 |

**clip_rn50x16**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0668 | 0.0092 | 0.0091 | 6%/80%/14% | 0.0826 | 0.0094 | 0.0777 | 0.0959 | 0.0966 |
| 0.2 | 50 | 0.0871 | 0.0181 | 0.0181 | 6%/86%/8% | 0.1558 | 0.0183 | 0.1516 | 0.1798 | 0.1803 |
| 0.3 | 50 | 0.0891 | 0.0276 | 0.0271 | 2%/88%/10% | 0.2182 | 0.0278 | 0.2155 | 0.2723 | 0.2726 |
| 0.4 | 50 | 0.0862 | 0.0371 | 0.0370 | 6%/84%/10% | 0.2649 | 0.0374 | 0.2614 | 0.3515 | 0.3525 |
| 0.5 | 50 | 0.0838 | 0.0450 | 0.0448 | 6%/86%/8% | 0.3266 | 0.0456 | 0.2892 | 0.4139 | 0.4134 |

**siglip**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0629 | 0.0091 | 0.0091 | 8%/88%/4% | 0.0822 | 0.0093 | 0.0702 | 0.0871 | 0.0869 |
| 0.2 | 50 | 0.0900 | 0.0178 | 0.0178 | 4%/92%/4% | 0.1300 | 0.0182 | 0.1231 | 0.2052 | 0.2054 |
| 0.3 | 50 | 0.0984 | 0.0274 | 0.0271 | 0%/94%/6% | 0.1586 | 0.0276 | 0.1254 | 0.3076 | 0.3073 |
| 0.4 | 50 | 0.1047 | 0.0373 | 0.0374 | 4%/90%/6% | 0.1712 | 0.0377 | 0.1402 | 0.3824 | 0.3830 |
| 0.5 | 50 | 0.1044 | 0.0441 | 0.0447 | 6%/90%/4% | 0.2375 | 0.0452 | 0.1988 | 0.4415 | 0.4419 |

**cifar_vae_linf**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|

**clip_rn50x16_off**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0618 | 0.0094 | 0.0094 | 6%/90%/4% | 0.0902 | 0.0095 | 0.0845 | 0.1058 | 0.1063 |
| 0.2 | 50 | 0.0807 | 0.0185 | 0.0186 | 4%/94%/2% | 0.1695 | 0.0188 | 0.1689 | 0.1922 | 0.1919 |
| 0.3 | 50 | 0.0874 | 0.0282 | 0.0284 | 6%/94%/0% | 0.2462 | 0.0285 | 0.2417 | 0.2754 | 0.2746 |
| 0.4 | 50 | 0.0847 | 0.0357 | 0.0354 | 4%/90%/6% | 0.2959 | 0.0361 | 0.2670 | 0.3526 | 0.3518 |
| 0.5 | 50 | 0.0815 | 0.0461 | 0.0466 | 8%/86%/6% | 0.3296 | 0.0472 | 0.3215 | 0.4123 | 0.4129 |

**siglip_off**

| σ | n | mean r (=2κ r_SR) | R_SR | R_ContRS (a) | win/tie/lose (a) | R_ContRS (b) | R_joint (a) | R_joint (b) | err SR | err ContRS (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 0.0420 | 0.0092 | 0.0093 | 14%/80%/6% | 0.0789 | 0.0094 | 0.0760 | 0.0653 | 0.0651 |
| 0.2 | 50 | 0.0697 | 0.0193 | 0.0190 | 6%/82%/12% | 0.1373 | 0.0195 | 0.1317 | 0.1299 | 0.1297 |
| 0.3 | 50 | 0.0837 | 0.0274 | 0.0270 | 2%/88%/10% | 0.1714 | 0.0279 | 0.1470 | 0.2113 | 0.2117 |
| 0.4 | 50 | 0.0921 | 0.0363 | 0.0364 | 2%/96%/2% | 0.1900 | 0.0368 | 0.1890 | 0.2838 | 0.2844 |
| 0.5 | 50 | 0.0992 | 0.0446 | 0.0448 | 4%/86%/10% | 0.1872 | 0.0452 | 0.1847 | 0.3458 | 0.3453 |


## 3.3 六个新增架构的中文紧凑表（50 图，便于 rebuttal 摘取）

本小节与 3.1 的相应行是同一结果的中文紧凑展示，不是额外一批独立实验，不能重复计数。

以下来源表已完整内嵌：[rebuttal/summary/rebuttal_new_arch_tables.md](rebuttal/summary/rebuttal_new_arch_tables.md)。


#### 一、检测（Jaccard）

##### RetinaFace-R50（CNN）

**表 1：对比 center smoothing**（σ=0.1）

| r（ContRS） | ContRS R = CS ε_in | CS ε_out | 平滑误差 ContRS | 平滑误差 CS |
|---|---|---|---|---|
| 0.0297 | 0.037 | 0.0631 | 0.0446 | 0.0443 |
| 0.0344 | 0.064 | 0.0702 | 0.0446 | 0.0443 |
| 0.0398 | 0.096 | 0.0796 | 0.0448 | 0.0443 |
| 0.0460 | 0.132 | 0.0921 | 0.0449 | 0.0443 |
| 0.0531 | 0.168 | 0.1097 | 0.0447 | 0.0443 |

**表 2：对比 safe region**

| σ | r | ContRS R | SR R | 平滑误差 ContRS | 平滑误差 SR |
|---|---|---|---|---|---|
| 0.1 | 0.0338 | 0.0604 | 0.0091 | 0.0446 | 0.0445 |
| 0.2 | 0.0512 | 0.1166 | 0.0175 | 0.0706 | 0.0700 |
| 0.3 | 0.1800 | 0.7824 | 0.0827 | 0.1055 | 0.1383 |
| 0.4 | 0.3145 | 1.0116 | 0.1861 | 0.1843 | 0.1930 |
| 0.5 | 0.5628 | 1.0639 | 0.4263 | 0.2291 | 0.3394 |

##### DETR-R50（Transformer）

**表 1：对比 center smoothing**（σ=0.1）

| r（ContRS） | ContRS R = CS ε_in | CS ε_out | 平滑误差 ContRS | 平滑误差 CS |
|---|---|---|---|---|
| 0.0134 | 0.031 | 0.4629 | 0.1631 | 0.1717 |
| 0.0207 | 0.054 | 0.5956 | 0.1460 | 0.1717 |
| 0.0320 | 0.098 | 0.8203 | 0.1413 | 0.1717 |
| 0.0460 | 0.139 | 0.8619 | 0.1415 | 0.1717 |
| 0.0710 | 0.179 | 0.9709 | 0.1417 | 0.1717 |

**表 2：对比 safe region**

| σ | r | ContRS R | SR R | 平滑误差 ContRS | 平滑误差 SR |
|---|---|---|---|---|---|
| 0.1 | 0.1918 | 0.2116 | 0.0341 | 0.1521 | 0.1718 |
| 0.2 | 0.2972 | 0.3416 | 0.0806 | 0.2139 | 0.2572 |
| 0.3 | 0.4951 | 0.4104 | 0.1789 | 0.3276 | 0.3921 |
| 0.4 | 0.6173 | 0.4395 | 0.3150 | 0.3937 | 0.4575 |
| 0.5 | 0.9256 | 0.4154 | 0.6185 | 0.5353 | 0.5993 |

#### 二、重建（LPIPS）

##### SD-VAE（CNN）

**表 1：对比 center smoothing**（σ=0.25）

| r（ContRS） | ContRS R = CS ε_in | CS ε_out | 平滑误差 ContRS | 平滑误差 CS |
|---|---|---|---|---|
| 0.0595 | 0.089 | 0.2222 | 0.0264 | 0.0253 |
| 0.0688 | 0.143 | 0.2360 | 0.0252 | 0.0253 |
| 0.0855 | 0.249 | 0.2691 | 0.0270 | 0.0253 |
| 0.0988 | 0.333 | 0.3037 | 0.0270 | 0.0253 |
| 0.1143 | 0.421 | 0.3567 | 0.0274 | 0.0253 |

**表 2：对比 safe region**

| σ | r | ContRS R | SR R | 平滑误差 ContRS | 平滑误差 SR |
|---|---|---|---|---|---|
| 0.1 | 0.0302 | 0.1033 | 0.0095 | 0.0077 | 0.0068 |
| 0.2 | 0.0640 | 0.1849 | 0.0187 | 0.0191 | 0.0177 |
| 0.3 | 0.0955 | 0.2431 | 0.0277 | 0.0360 | 0.0343 |
| 0.4 | 0.1225 | 0.2837 | 0.0381 | 0.0581 | 0.0552 |
| 0.5 | 0.1465 | 0.3102 | 0.0453 | 0.0766 | 0.0747 |

##### MAE ViT-B（Transformer）

**表 1：对比 center smoothing**（σ=0.25）

| r（ContRS） | ContRS R = CS ε_in | CS ε_out | 平滑误差 ContRS | 平滑误差 CS |
|---|---|---|---|---|
| 0.0595 | 0.091 | 0.1927 | 0.0294 | 0.0284 |
| 0.0688 | 0.163 | 0.2072 | 0.0295 | 0.0284 |
| 0.0740 | 0.202 | 0.2159 | 0.0293 | 0.0284 |
| 0.0855 | 0.306 | 0.2437 | 0.0300 | 0.0284 |
| 0.0988 | 0.416 | 0.2884 | 0.0303 | 0.0284 |

**表 2：对比 safe region**

| σ | r | ContRS R | SR R | 平滑误差 ContRS | 平滑误差 SR |
|---|---|---|---|---|---|
| 0.1 | 0.0229 | 0.0774 | 0.0088 | 0.0052 | 0.0051 |
| 0.2 | 0.0547 | 0.1468 | 0.0187 | 0.0194 | 0.0190 |
| 0.3 | 0.0853 | 0.2016 | 0.0278 | 0.0409 | 0.0407 |
| 0.4 | 0.1136 | 0.2421 | 0.0359 | 0.0679 | 0.0677 |
| 0.5 | 0.1381 | 0.2743 | 0.0451 | 0.0941 | 0.0954 |

#### 三、检索（cosine）

##### CLIP RN50x16（CNN）

**表 1：对比 center smoothing**（σ=0.25）

| r（ContRS） | ContRS R = CS ε_in | CS ε_out | 平滑误差 ContRS | 平滑误差 CS |
|---|---|---|---|---|
| 0.0640 | 0.079 | 0.2467 | 0.2257 | 0.2257 |
| 0.0800 | 0.149 | 0.2691 | 0.2244 | 0.2257 |
| 0.0988 | 0.242 | 0.3066 | 0.2260 | 0.2257 |
| 0.1143 | 0.321 | 0.3472 | 0.2268 | 0.2257 |
| 0.1420 | 0.443 | 0.4433 | 0.2278 | 0.2257 |

**表 2：对比 safe region**

| σ | r | ContRS R | SR R | 平滑误差 ContRS | 平滑误差 SR |
|---|---|---|---|---|---|
| 0.1 | 0.0668 | 0.0826 | 0.0092 | 0.0971 | 0.0959 |
| 0.2 | 0.0871 | 0.1558 | 0.0181 | 0.1822 | 0.1798 |
| 0.3 | 0.0891 | 0.2182 | 0.0276 | 0.2734 | 0.2723 |
| 0.4 | 0.0862 | 0.2649 | 0.0371 | 0.3512 | 0.3515 |
| 0.5 | 0.0838 | 0.3266 | 0.0450 | 0.4119 | 0.4139 |

##### SigLIP ViT-B/16（Transformer）

**表 1：对比 center smoothing**（σ=0.25）

| r（ContRS） | ContRS R = CS ε_in | CS ε_out | 平滑误差 ContRS | 平滑误差 CS |
|---|---|---|---|---|
| 0.0795 | 0.077 | 0.2638 | 0.2603 | 0.2625 |
| 0.0880 | 0.114 | 0.2745 | 0.2618 | 0.2625 |
| 0.1000 | 0.170 | 0.2925 | 0.2626 | 0.2625 |
| 0.1200 | 0.275 | 0.3328 | 0.2630 | 0.2625 |
| 0.1420 | 0.402 | 0.4067 | 0.2641 | 0.2625 |

**表 2：对比 safe region**

| σ | r | ContRS R | SR R | 平滑误差 ContRS | 平滑误差 SR |
|---|---|---|---|---|---|
| 0.1 | 0.0629 | 0.0822 | 0.0091 | 0.0866 | 0.0871 |
| 0.2 | 0.0900 | 0.1300 | 0.0178 | 0.2058 | 0.2052 |
| 0.3 | 0.0984 | 0.1586 | 0.0274 | 0.3086 | 0.3076 |
| 0.4 | 0.1047 | 0.1712 | 0.0373 | 0.3826 | 0.3824 |
| 0.5 | 0.1044 | 0.2375 | 0.0441 | 0.4411 | 0.4415 |


---

# 第四部分：早期 baseline 结果（49/50 图，历史对照）

这部分早于 300 图扩展：**MTCNN 是 49 图，CLIP 和 VAE-ℓ∞ 各 50 图**。保留它是为了完整追踪已有结果；rebuttal 主文优先使用第二部分的 300 图表，不得把这里的数值与 300 图版本混成同一样本量。


## 4.1 中位数、α-平滑与 ContRS-M 早期表

以下来源表已完整内嵌：[rebuttal/summary/baseline_tables.md](rebuttal/summary/baseline_tables.md)。

#### 新 baseline 对比（论文原任务与原模型）


#### 检测（Jaccard）：mtcnn（σ=0.1，49 张图）

**表 A：对比中位数平滑**（中位数平滑的 ε_in = ContRS 的平均 R）

| r（ContRS） | ContRS R = 中位数平滑 ε_in | 中位数平滑 ε_out | 平滑误差 ContRS | 平滑误差 中位数平滑 | 中位数平滑在同一 r 下的 R |
|---|---|---|---|---|---|
| 0.0800 | 0.054 | 0.0338 | 0.0509 | 0.0481 | 0.1393 |
| 0.1000 | 0.086 | 0.0543 | 0.0539 | 0.0481 | 0.1727 |
| 0.1200 | 0.124 | 0.0797 | 0.0545 | 0.0481 | 0.2056 |
| 0.1400 | 0.163 | 0.1051 | 0.0551 | 0.0481 | 0.2324 |
| 0.1600 | 0.194 | 0.1252 | 0.0540 | 0.0481 | 0.2524 |

**表 A'：同一容差 r 下的认证半径**（ContRS-M = 候选中心加入逐坐标中位数/均值，证书不变）

| r | ContRS R | ContRS-M R | 中位数平滑 R | 平滑误差 ContRS | 平滑误差 ContRS-M | 平滑误差 中位数平滑 |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0497 | 0.0498 | 0.1393 | 0.0509 | 0.0509 | 0.0481 |
| 0.1000 | 0.0816 | 0.0819 | 0.1727 | 0.0539 | 0.0540 | 0.0481 |
| 0.1200 | 0.1197 | 0.1200 | 0.2056 | 0.0545 | 0.0542 | 0.0481 |
| 0.1400 | 0.1594 | 0.1595 | 0.2324 | 0.0551 | 0.0550 | 0.0481 |
| 0.1600 | 0.1930 | 0.1930 | 0.2524 | 0.0540 | 0.0540 | 0.0481 |

**表 B：对比 α-平滑**（同一容差 r；α=0.35、P=0.9、n=10 为官方默认，置信度与 ContRS 一致）

| r | ContRS R | α-平滑 R | 平滑误差 ContRS | 平滑误差 α-平滑 |
|---|---|---|---|---|
| 0.0800 | 0.0497 | 0.0002 | 0.0509 | 0.0501 |
| 0.1000 | 0.0816 | 0.0018 | 0.0539 | 0.0501 |
| 0.1200 | 0.1197 | 0.0034 | 0.0545 | 0.0501 |
| 0.1400 | 0.1594 | 0.0054 | 0.0551 | 0.0501 |
| 0.1600 | 0.1930 | 0.0073 | 0.0540 | 0.0501 |

#### 检索（cosine）：clip_l14（σ=0.25，50 张图）

**表 A：对比中位数平滑**（中位数平滑的 ε_in = ContRS 的平均 R）

| r（ContRS） | ContRS R = 中位数平滑 ε_in | 中位数平滑 ε_out | 平滑误差 ContRS | 平滑误差 中位数平滑 | 中位数平滑在同一 r 下的 R |
|---|---|---|---|---|---|
| 0.0800 | 0.080 | 0.0029 | 0.2491 | 0.2405 | 0.4681 |
| 0.0880 | 0.109 | 0.0050 | 0.2472 | 0.2405 | 0.4877 |
| 0.1000 | 0.155 | 0.0096 | 0.2462 | 0.2405 | 0.5142 |
| 0.1080 | 0.189 | 0.0138 | 0.2462 | 0.2405 | 0.5302 |
| 0.1200 | 0.242 | 0.0224 | 0.2465 | 0.2405 | 0.5523 |

**表 A'：同一容差 r 下的认证半径**（ContRS-M = 候选中心加入逐坐标中位数/均值，证书不变）

| r | ContRS R | ContRS-M R | 中位数平滑 R | 平滑误差 ContRS | 平滑误差 ContRS-M | 平滑误差 中位数平滑 |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0796 | 0.1747 | 0.4681 | 0.2491 | 0.2404 | 0.2405 |
| 0.0880 | 0.1088 | 0.2187 | 0.4877 | 0.2472 | 0.2404 | 0.2405 |
| 0.1000 | 0.1555 | 0.2871 | 0.5142 | 0.2462 | 0.2404 | 0.2405 |
| 0.1080 | 0.1889 | 0.3325 | 0.5302 | 0.2462 | 0.2404 | 0.2405 |
| 0.1200 | 0.2425 | 0.3937 | 0.5523 | 0.2465 | 0.2406 | 0.2405 |

**表 B：对比 α-平滑**（同一容差 r；α=0.35、P=0.9、n=10 为官方默认，置信度与 ContRS 一致）

| r | ContRS R | α-平滑 R | 平滑误差 ContRS | 平滑误差 α-平滑 |
|---|---|---|---|---|
| 0.0800 | 0.0796 | 0.0000 | 0.2491 | 0.2422 |
| 0.0880 | 0.1088 | 0.0000 | 0.2472 | 0.2422 |
| 0.1000 | 0.1555 | 0.0000 | 0.2462 | 0.2422 |
| 0.1080 | 0.1889 | 0.0000 | 0.2462 | 0.2422 |
| 0.1200 | 0.2425 | 0.0000 | 0.2465 | 0.2422 |

#### 重建（像素 ℓ∞；LPIPS 下两个 baseline 均无法认证）：cifar_vae_linf（σ=0.25，50 张图）

**表 A：对比中位数平滑**（中位数平滑的 ε_in = ContRS 的平均 R）

| r（ContRS） | ContRS R = 中位数平滑 ε_in | 中位数平滑 ε_out | 平滑误差 ContRS | 平滑误差 中位数平滑 | 中位数平滑在同一 r 下的 R |
|---|---|---|---|---|---|
| 0.2852 | 0.089 | 0.0270 | 0.0904 | 0.0491 | 0.7290 |
| 0.3097 | 0.148 | 0.0422 | 0.0900 | 0.0491 | 0.7323 |
| 0.3364 | 0.226 | 0.0625 | 0.0920 | 0.0491 | 0.7328 |
| 0.3653 | 0.327 | 0.0898 | 0.0951 | 0.0491 | 0.7328 |
| 0.3967 | 0.431 | 0.1189 | 0.0955 | 0.0491 | 0.7328 |

**表 A'：同一容差 r 下的认证半径**（ContRS-M = 候选中心加入逐坐标中位数/均值，证书不变）

| r | ContRS R | ContRS-M R | 中位数平滑 R | 平滑误差 ContRS | 平滑误差 ContRS-M | 平滑误差 中位数平滑 |
|---|---|---|---|---|---|---|
| 0.2852 | 0.0891 | 0.1779 | 0.7290 | 0.0904 | 0.0507 | 0.0491 |
| 0.3097 | 0.1480 | 0.2718 | 0.7323 | 0.0900 | 0.0505 | 0.0491 |
| 0.3364 | 0.2255 | 0.3682 | 0.7328 | 0.0920 | 0.0516 | 0.0491 |
| 0.3653 | 0.3275 | 0.4594 | 0.7328 | 0.0951 | 0.0512 | 0.0491 |
| 0.3967 | 0.4314 | 0.5461 | 0.7328 | 0.0955 | 0.0525 | 0.0491 |

**表 B：对比 α-平滑**（同一容差 r；α=0.35、P=0.9、n=10 为官方默认，置信度与 ContRS 一致）

| r | ContRS R | α-平滑 R | 平滑误差 ContRS | 平滑误差 α-平滑 |
|---|---|---|---|---|
| 0.2852 | 0.0891 | 0.0549 | 0.0904 | 0.0668 |
| 0.3097 | 0.1480 | 0.0974 | 0.0900 | 0.0668 |
| 0.3364 | 0.2255 | 0.1604 | 0.0920 | 0.0668 |
| 0.3653 | 0.3275 | 0.2461 | 0.0951 | 0.0668 |
| 0.3967 | 0.4314 | 0.3378 | 0.0955 | 0.0668 |


---

# 第五部分：safe-region target-p 扫描（每行 50 图，全 45 行）

统一设置：p=0.6/0.7/0.8，σ=0.1/0.2/0.3/0.4/0.5；KDD P1=5000、P2=10000、alpha=0.005、batch=32；ContRS N1/Nsel/N2=5000/5000/10000。每个 (σ,p) 以 safe-region 组均值 `mean_safe_r` 作为 ContRS 的统一 s。双方不是等预算比较。

VAE/CLIP 使用外部提供的六位 mean_safe_r 和本地 ContRS 独立噪声池；MTCNN 是本地 32 卡逐图并行独立重跑，KDD 与 ContRS 也使用不同种子命名空间。三个 p 共用 safe-region P1/P2，但每个 p 的边际构造与认证算法不变，因此跨 p 结果相关。


## 5.1 CIFAR-VAE 与 CLIP：30 行完整结果

这里的 30/30 正向只对所提供 mean_safe_r 数值严格成立。外部旧驱动逐图只写四位小数；若均值来自这些 TSV，半径不确定度最高约 ±5×10⁻⁵，且外部逐图 ID/像素哈希未闭合。

以下来源表已完整内嵌：[p_sweep/summary/p_sweep_tables.md](p_sweep/summary/p_sweep_tables.md)。

#### Safe-region p sweep vs ContRS (50 images)

For each (model, sigma, p), ContRS uses the exact s = safe-region mean_safe_r. The reported output tolerance is r = 2*kappa*s (kappa=2 for LPIPS/cosine). Delta R = ContRS - safe region; larger R is better. Smoothing error is measured against the clean base-model output.

Safe-region values are user-provided aggregates displayed to six decimals. The legacy VAE/CLIP KDD drivers write each image's safe radius to TSV with only four decimals, and all 30 supplied means lie on the resulting 2e-6 grid. If the supplied aggregates came from those TSV files, each mean_safe_r has up to ±5e-5 rounding uncertainty; six displayed decimals do not imply full precision. ContRS is evaluated exactly at each supplied value, so these rows are results at the provided radii, pending raw full-precision outputs or a ±5e-5 sensitivity analysis. The local ContRS image IDs match the locally reconstructed manifests, but no external per-image IDs or pixel hashes are available to prove the external runs used identical data. The two methods use independent noise pools. Excluding the common clean evaluation, KDD uses 15,000 noisy forwards per image and split ContRS uses 20,000; including it gives 15,001 and 20,001. This is group-level mean-scale matching, not an equal-query or shared-noise comparison.
Per-image radius matching, paired CIs, external ID verification and full-precision confirmation require the original per-image outputs and are not inferred from these aggregates.

##### cifar_vae

| sigma | p | r = 2κ mean(safe_r) | ContRS R | SR R | ΔR | ContRS cert | ContRS err | SR err | Δerr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.6 | 0.002072 | 0.077971 | 0.022370 | +0.055601 | 70% | 0.000139 | 0.000150 | -0.000011 |
| 0.1 | 0.7 | 0.002312 | 0.096765 | 0.049346 | +0.047419 | 72% | 0.000138 | 0.000150 | -0.000012 |
| 0.1 | 0.8 | 0.002608 | 0.117210 | 0.081030 | +0.036180 | 78% | 0.000142 | 0.000148 | -0.000006 |
| 0.2 | 0.6 | 0.007768 | 0.154784 | 0.044796 | +0.109988 | 68% | 0.000749 | 0.000754 | -0.000005 |
| 0.2 | 0.7 | 0.008624 | 0.190388 | 0.098800 | +0.091588 | 72% | 0.000723 | 0.000754 | -0.000031 |
| 0.2 | 0.8 | 0.009784 | 0.233440 | 0.161676 | +0.071764 | 76% | 0.000757 | 0.000748 | +0.000009 |
| 0.3 | 0.6 | 0.016632 | 0.229870 | 0.066642 | +0.163228 | 68% | 0.002076 | 0.002064 | +0.000012 |
| 0.3 | 0.7 | 0.018480 | 0.283276 | 0.147262 | +0.136014 | 70% | 0.002088 | 0.002078 | +0.000010 |
| 0.3 | 0.8 | 0.020912 | 0.347881 | 0.242898 | +0.104983 | 76% | 0.002116 | 0.002104 | +0.000012 |
| 0.4 | 0.6 | 0.028352 | 0.304228 | 0.088802 | +0.215426 | 66% | 0.003918 | 0.004226 | -0.000308 |
| 0.4 | 0.7 | 0.031352 | 0.372982 | 0.196628 | +0.176354 | 70% | 0.003929 | 0.004282 | -0.000353 |
| 0.4 | 0.8 | 0.035440 | 0.460055 | 0.323342 | +0.136713 | 78% | 0.004120 | 0.004388 | -0.000268 |
| 0.5 | 0.6 | 0.041936 | 0.369052 | 0.109174 | +0.259878 | 68% | 0.007254 | 0.007256 | -0.000002 |
| 0.5 | 0.7 | 0.046400 | 0.457813 | 0.246532 | +0.211281 | 70% | 0.007365 | 0.007434 | -0.000069 |
| 0.5 | 0.8 | 0.052224 | 0.562471 | 0.404746 | +0.157725 | 76% | 0.007514 | 0.007426 | +0.000088 |

##### clip_l14

| sigma | p | r = 2κ mean(safe_r) | ContRS R | SR R | ΔR | ContRS cert | ContRS err | SR err | Δerr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.6 | 0.065712 | 0.084405 | 0.021226 | +0.063179 | 62% | 0.091766 | 0.091418 | +0.000348 |
| 0.1 | 0.7 | 0.070624 | 0.099269 | 0.047676 | +0.051593 | 64% | 0.092071 | 0.091612 | +0.000459 |
| 0.1 | 0.8 | 0.077152 | 0.115573 | 0.079456 | +0.036117 | 70% | 0.092374 | 0.091796 | +0.000578 |
| 0.2 | 0.6 | 0.100248 | 0.147506 | 0.043964 | +0.103542 | 58% | 0.194079 | 0.193520 | +0.000559 |
| 0.2 | 0.7 | 0.108008 | 0.176284 | 0.097506 | +0.078778 | 68% | 0.193671 | 0.193144 | +0.000527 |
| 0.2 | 0.8 | 0.118056 | 0.218058 | 0.160262 | +0.057796 | 80% | 0.193396 | 0.194478 | -0.001082 |
| 0.3 | 0.6 | 0.109872 | 0.223104 | 0.065760 | +0.157344 | 56% | 0.290973 | 0.289430 | +0.001543 |
| 0.3 | 0.7 | 0.117912 | 0.262942 | 0.146886 | +0.116056 | 64% | 0.291234 | 0.288910 | +0.002324 |
| 0.3 | 0.8 | 0.128344 | 0.318098 | 0.240338 | +0.077760 | 76% | 0.291091 | 0.289168 | +0.001923 |
| 0.4 | 0.6 | 0.119200 | 0.278104 | 0.089590 | +0.188514 | 58% | 0.350991 | 0.353100 | -0.002109 |
| 0.4 | 0.7 | 0.128640 | 0.337261 | 0.197274 | +0.139987 | 64% | 0.353365 | 0.354820 | -0.001455 |
| 0.4 | 0.8 | 0.141008 | 0.409705 | 0.322908 | +0.086797 | 76% | 0.354599 | 0.355032 | -0.000433 |
| 0.5 | 0.6 | 0.123720 | 0.286656 | 0.110820 | +0.175836 | 60% | 0.401900 | 0.401246 | +0.000654 |
| 0.5 | 0.7 | 0.133928 | 0.363677 | 0.245156 | +0.118521 | 64% | 0.400973 | 0.402840 | -0.001867 |
| 0.5 | 0.8 | 0.147288 | 0.473594 | 0.401272 | +0.072322 | 82% | 0.400866 | 0.402464 | -0.001598 |


## 5.2 MTCNN：15 行完整结果、认证率、饱和率和描述性 CI

当前 selected50 与 legacy50 的交集为 0，外部实际文件名仍未确认，因此不能称为另一会话同一 50 图的精确复现。表中 safe_r=1 是覆盖整个有界 Jaccard 输出空间的边界饱和，不等于稳定、有意义的人脸框。

以下来源表已完整内嵌：[p_sweep/summary/mtcnn_p_sweep_tables.md](p_sweep/summary/mtcnn_p_sweep_tables.md)。

#### MTCNN safe-region p sweep vs ContRS (50 images)

Local full-precision results for the current frozen selected-50 manifest. External/historical filename confirmation is still absent, so this is not claimed as a reproduction of the other session's exact 50-image run. For each (sigma,p), ContRS is evaluated at the exact group mean of the 50 KDD safe_r values; r=2s for Jaccard. Both methods use the same image IDs but independent per-image noise pools (distinct SHA256 seed namespaces). KDD uses 15,000 forwards/image; split ContRS uses 20,000. This is group-level mean-scale matching, not per-image radius matching or an equal-query comparison.

Delta R = ContRS - SR. Row-wise 95% CIs are descriptive intervals paired across the 50 images for the difference in group summaries; they are not shared-noise paired intervals, are not multiplicity-corrected, and rows are dependent because the three p values share P1/P2 and images recur across groups. The shared s_bar is estimated from the same 50 KDD images; holding it fixed in these row-wise t intervals does not propagate that estimation uncertainty or induce independence across the 50 ContRS evaluations. SR is a safe-region membership certificate, not by itself ContRS's finite-sample guarantee relative to the deployed sample-selected center.

SR saturation is the fraction with safe_r=1. Under the implemented Jaccard convention this closed ball covers the entire bounded output space, so kcert=10000 and the resulting large R do not by themselves demonstrate stable, meaningful face boxes. At sigma=0.5 the mapped ContRS output tolerance r=2s also exceeds the Jaccard upper bound 1 and is therefore vacuous as a prediction-change bound.

| sigma | p | r=2 mean(safe_r) | ContRS R | SR R | ΔR ± 95% CI | ContRS cert | SR cert | SR safe_r=1 | ContRS err | SR err | Δerr ± 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.6 | 0.094202 | 0.037973 | 0.021120 | +0.016853 ± 0.011597 | 70% | 100% | 0% | 0.046208 | 0.045126 | +0.001081 ± 0.002107 |
| 0.1 | 0.7 | 0.108315 | 0.061847 | 0.048136 | +0.013710 ± 0.014464 | 86% | 100% | 0% | 0.046912 | 0.046675 | +0.000238 ± 0.001764 |
| 0.1 | 0.8 | 0.124502 | 0.092685 | 0.079506 | +0.013179 ± 0.015876 | 98% | 100% | 0% | 0.047981 | 0.048757 | -0.000776 ± 0.003028 |
| 0.2 | 0.6 | 0.116469 | 0.070936 | 0.043578 | +0.027358 ± 0.017635 | 78% | 100% | 0% | 0.081848 | 0.079517 | +0.002331 ± 0.002851 |
| 0.2 | 0.7 | 0.131257 | 0.116709 | 0.097413 | +0.019296 ± 0.021688 | 90% | 100% | 0% | 0.080963 | 0.080720 | +0.000243 ± 0.003181 |
| 0.2 | 0.8 | 0.149778 | 0.176041 | 0.160184 | +0.015858 ± 0.024883 | 96% | 100% | 0% | 0.083110 | 0.081241 | +0.001869 ± 0.003012 |
| 0.3 | 0.6 | 0.179065 | 0.239160 | 0.083721 | +0.155439 ± 0.058880 | 94% | 100% | 2% | 0.114401 | 0.113957 | +0.000444 ± 0.002190 |
| 0.3 | 0.7 | 0.197816 | 0.307777 | 0.162399 | +0.145378 ± 0.059961 | 96% | 100% | 2% | 0.114886 | 0.114649 | +0.000238 ± 0.002409 |
| 0.3 | 0.8 | 0.253762 | 0.488571 | 0.269723 | +0.218848 ± 0.078279 | 98% | 100% | 4% | 0.117466 | 0.112124 | +0.005342 ± 0.004687 |
| 0.4 | 0.6 | 0.398347 | 0.667180 | 0.232280 | +0.434901 ± 0.202555 | 90% | 100% | 12% | 0.145158 | 0.207013 | -0.061855 ± 0.067434 |
| 0.4 | 0.7 | 0.454656 | 0.726858 | 0.350439 | +0.376419 ± 0.211619 | 90% | 100% | 14% | 0.144150 | 0.223174 | -0.079025 ± 0.072898 |
| 0.4 | 0.8 | 0.522068 | 0.761323 | 0.478015 | +0.283308 ± 0.212133 | 90% | 100% | 16% | 0.151577 | 0.222384 | -0.070806 ± 0.072407 |
| 0.5 | 0.6 | 1.105272 | 0.413008 | 0.873683 | -0.460676 ± 0.344203 | 52% | 100% | 50% | 0.177363 | 0.446306 | -0.268942 ± 0.115049 |
| 0.5 | 0.7 | 1.151105 | 0.413008 | 0.968606 | -0.555598 ± 0.327130 | 52% | 100% | 52% | 0.177363 | 0.450048 | -0.272684 ± 0.114129 |
| 0.5 | 0.8 | 1.236978 | 0.413008 | 1.093755 | -0.680747 ± 0.305259 | 52% | 100% | 56% | 0.176305 | 0.452558 | -0.276253 ± 0.113317 |


## 5.3 p 扫描数值复核与证据限制

已复核 MTCNN 500 个 NPZ、KDD/ContRS 各 750 个目标证书点，以及 VAE/CLIP 1500 个 ContRS 目标证书点；CP 下界和 R 的重算误差为 0。该数值核验不消除数据来源、精度和统计解释限制。

以下来源表已完整内嵌：[p_sweep/summary/verification.md](p_sweep/summary/verification.md)。

#### Safe-region p 扫描复核记录

##### 数值完整性

- MTCNN 共检查 500 个原始 NPZ：KDD 250、ContRS 250；每个 sigma 均为相同 50 个唯一图片 ID。
- 对 KDD 750 个目标点和 ContRS 750 个目标点，独立重算 Clopper–Pearson 下界及 R；p_L 与 R 的最大绝对误差均为 0。
- 15 个 KDD safe_r 全精度组均值与 mtcnn_s_values.json 位级一致；ContRS 的 750 个目标半径均精确唯一命中。
- 独立重算的 MTCNN 15 行均值、标准差、认证率、配对 t 区间与原始 TSV 的全部派生数值一致。
- VAE/CLIP 的 1,500 个图像-p ContRS 证书点此前也已独立复核，p_L 与 R 最大误差均为 0。

##### VAE/CLIP 外部汇总精度与出处

- 当前 30 行只精确评估用户提供的 mean_safe_r 数值。本地旧 VAE/CLIP KDD 驱动均以 %.4f 写逐图 safe radius，且 30/30 个外部均值都落在 2e-6=0.0001/50 格点上，符合“50 个四位小数逐图值求均值”的特征。
- 如果外部汇总确实来自这些四位小数 TSV，则每个 mean_safe_r 的舍入不确定度可达 ±5e-5，而不是六位显示值本身的 ±5e-7；VAE sigma=0.1 的小半径尤其敏感。必须取得原始全精度输出，或明确做 ±5e-5 敏感性分析后，才能把 30/30 的方向扩展为原始未舍入协议下的结论。
- 本地 ContRS 图片清单是按给定 seed 规则在本机数据池重建并核对的；没有外部逐图 ID 和像素哈希，不能据此证明外部 KDD 使用完全相同的数据。

##### MTCNN 必须保留的边界

1. 当前结果只对应 mtcnn50_selection_check.json 的 selected 50。该清单与 legacy50 交集为 0，外部实际文件名尚未确认；不能称为另一会话或历史 50 图的直接复现。
2. KDD safe_r=1 的数量（每档 50 图）：
   - sigma=0.1：0/0/0；sigma=0.2：0/0/0；
   - sigma=0.3：1/1/2；sigma=0.4：6/7/8；sigma=0.5：25/26/28。
   p 顺序均为 0.6/0.7/0.8。
3. 当前 Jaccard 实现把非法框相关距离置为 1，并用 distance<=safe_r 认证。因此 safe_r=1 覆盖整个有界输出空间，产生 kcert=10000 和最大认证半径。这是程序定义一致的边界饱和，不是汇总错误，但不能解释为稳定、有意义的人脸框证书。
4. sigma=0.5 的 ContRS 三档 mean R 都是 0.41300785434322707、认证率均为 52%。原始逐图字段证实 50/50 图在三档阈值上的 kselect/kcert/pL/R 相同，49/50 图中心也相同；唯一换中心的图仍计数并列。这是三个离散阈值上的经验计数平台，不是表格误复制，也不能外推为连续总体曲线恒定。
5. sigma=0.5 的映射输出容差 r=2s 为 1.105272/1.151105/1.236978，均超过 Jaccard 上界 1，作为 prediction-change 输出界是平凡的。
6. 逐行 95% CI 未做多重比较校正；三个 p 共用 P1/P2，同一图片跨组重复，15 行不是独立复现。共同的 s_bar 也由同一批 50 个 KDD 图片估计，固定 s_bar 的普通配对 t 区间没有传播这部分不确定性，也不使 50 个 ContRS 结果相互独立。因此这些区间只作描述性诊断，未校正时 Delta R 显著正向 8 行、负向 3 行、不显著 4 行；不应把这一计数写成无条件总体优越性。
7. 双方为同图、独立噪声池、组均值尺度匹配；不含共同 clean 前向时 KDD 每图 15,000 次、split ContRS 每图 20,000 次，计入 clean 则分别为 15,001/20,001。按图配对 CI 不消除 Monte Carlo 独立性和预算差异。有限样本 ContRS 的直接理论表述是扰动平滑输出相对部署时样本中心 z_hat 的变化界，不能不加说明地把 z_hat 换成总体 g(x)。

##### 报告原则

保留全部 15 行，不事后删除或过滤饱和样本并冒充主结果。正式使用前应确认外部图片清单，并预先定义无脸/非法框和 safe_r=1 的 abstain、过滤或分层协议；safe_r<1 子集只能作为明确标注的诊断分析。


## 5.4 p 扫描机器可读配置

以下来自 [p_sweep/run_config.json](p_sweep/run_config.json)：

```json
{
  "experiment": "safe-region target-p sweep, group-level mean-scale comparison with ContRS",
  "p_values": [0.6, 0.7, 0.8],
  "sigmas": [0.1, 0.2, 0.3, 0.4, 0.5],
  "images_per_group": 50,
  "external_kdd": {
    "seed": 42,
    "n_pred": 5000,
    "n_cert": 10000,
    "batch_size": 32,
    "alpha": 0.005,
    "source": "user-provided six-decimal aggregates; raw per-image outputs not present locally. Legacy drivers write per-image safe radii with four decimals and all 30 means lie on a 2e-6 grid; if aggregates came from those TSV files, radius uncertainty is up to ±5e-5 and requires raw outputs or sensitivity analysis."
  },
  "contrs": {
    "seed": 42,
    "seed_rule": "SHA256(seed:model:sigma:image), first 8 little-endian bytes mod 2^31",
    "n1": 5000,
    "n_select": 5000,
    "n2": 10000,
    "alpha": 0.005,
    "batch_by_model": {"cifar_vae": 2500, "clip_l14": 500, "mtcnn_kdd50": 256},
    "noise_pools": "independent from external KDD; P1/Psel/P2 independent within ContRS",
    "s_rule": "one exact supplied mean_safe_r per p, added with --extra-s; r=2*kappa*s"
  },
  "distance": {"cifar_vae": "LPIPS", "clip_l14": "cosine", "mtcnn_kdd50": "Jaccard"},
  "kappa": {"cifar_vae": 2, "clip_l14": 2, "mtcnn_kdd50": 1},
  "manifest": "input_manifest.json; MTCNN uses separate mtcnn50_selection_check.json and requires external filename confirmation",
  "mtcnn_local_parallel_kdd": {
    "status": "independent local rerun, not a continuation of the other session RNG stream",
    "seed_rule": "SHA256(42:kdd-p-sweep:mtcnn_kdd50:sigma:image), first 8 little-endian bytes mod 2^31",
    "noise_pools": "independent from ContRS; P1/P2 independent within KDD; three p values share P1/P2",
    "n_pred": 5000,
    "n_cert": 10000,
    "batch_size": 32,
    "parallelism": "one image per queue job, up to 32 GPU workers",
    "s_rule": "full-precision float64 mean of 50 per-image safe_r values per (sigma,p); no decimal rounding"
  }
}
```


---

# 第六部分：两个例外设置的独立新种子复核（300 图）

seed=20261109；只重跑 MTCNN σ=0.5 的 safe-region 比较，以及 CLIP σ=0.25 的 center smoothing 和新增 baseline。原始 seed=42 结果保持不变，本部分是独立复核而非替换或全套重跑。


## 6.1 新种子 paper-format 结果

以下来源表已完整内嵌：[ext300/recheck/summary/paper_format.md](ext300/recheck/summary/paper_format.md)。

#### Paper-format tables (Fig.1 protocol)

##### vs center smoothing: ContRS certified R is used as center smoothing's ε_in; compare ContRS's r with center smoothing's ε_out, and both smoothing errors

ε_out uses center smoothing's own constant (3 for Jaccard, γ(1+2γ)=10 for LPIPS/cosine).


**clip_l14_300** (Transformer (ViT-L/14), σ=0.25)

| r (ContRS) | ContRS R = ε_in | ε_out CS | smooth err ContRS | smooth err CS |
|---|---|---|---|---|
| 0.08 | 0.097 | 0.2649 | 0.2342 | 0.2352 |
| 0.088 | 0.135 | 0.2780 | 0.2344 | 0.2352 |
| 0.1 | 0.199 | 0.3027 | 0.2351 | 0.2352 |
| 0.108 | 0.241 | 0.3222 | 0.2352 | 0.2352 |
| 0.12 | 0.303 | 0.3551 | 0.2359 | 0.2352 |

##### vs safe region: run safe region (p=0.55) first, set ContRS r = 2κ·mean(r_SR) at each σ; compare R and smoothing errors

(* = evaluated at the nearest lower grid point; replaced automatically by the exact rerun once it finishes)


**mtcnn300**

| σ | r = 2κ·mean r_SR | ContRS R | safe region R | smooth err ContRS | smooth err safe region |
|---|---|---|---|---|---|
| 0.5 | 1.1800 | 0.3202 | 0.9098 | 0.1900 | 0.5415 |


## 6.2 新种子 CLIP baseline 结果

以下来源表已完整内嵌：[ext300/recheck/summary/baseline_tables.md](ext300/recheck/summary/baseline_tables.md)。

#### 新 baseline 对比（论文原任务与原模型）


#### 检索（cosine）：clip_l14_300（σ=0.25，300 张图）

**表 A：对比中位数平滑**（中位数平滑的 ε_in = ContRS 的平均 R）

| r（ContRS） | ContRS R = 中位数平滑 ε_in | 中位数平滑 ε_out | 平滑误差 ContRS | 平滑误差 中位数平滑 | 中位数平滑在同一 r 下的 R |
|---|---|---|---|---|---|
| 0.0800 | 0.097 | 0.0038 | 0.2342 | 0.2301 | 0.4857 |
| 0.0880 | 0.135 | 0.0068 | 0.2344 | 0.2301 | 0.5055 |
| 0.1000 | 0.199 | 0.0139 | 0.2351 | 0.2301 | 0.5323 |
| 0.1080 | 0.241 | 0.0203 | 0.2352 | 0.2301 | 0.5484 |
| 0.1200 | 0.303 | 0.0320 | 0.2359 | 0.2301 | 0.5703 |

**表 A'：同一容差 r 下的认证半径**（ContRS-M = 候选中心加入逐坐标中位数/均值，证书不变）

| r | ContRS R | ContRS-M R | 中位数平滑 R | 平滑误差 ContRS | 平滑误差 ContRS-M | 平滑误差 中位数平滑 |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0973 | 0.2205 | 0.4857 | 0.2342 | 0.2301 | 0.2301 |
| 0.0880 | 0.1354 | 0.2705 | 0.5055 | 0.2344 | 0.2302 | 0.2301 |
| 0.1000 | 0.1987 | 0.3415 | 0.5323 | 0.2351 | 0.2302 | 0.2301 |
| 0.1080 | 0.2412 | 0.3847 | 0.5484 | 0.2352 | 0.2303 | 0.2301 |
| 0.1200 | 0.3028 | 0.4422 | 0.5703 | 0.2359 | 0.2304 | 0.2301 |

**表 C：对比均值平滑**（ε_in = ContRS 的平均 R；检测/重建为逐坐标期望平滑 Salman et al. 2019，检索为 Smoothed Embeddings, Pautov et al. 2022）

| r（ContRS） | ContRS R = 均值平滑 ε_in | 均值平滑 ε_out | 平滑误差 ContRS | 平滑误差 均值平滑 | 均值平滑在同一 r 下的 R | ContRS R |
|---|---|---|---|---|---|---|
| 0.0800 | 0.097 | 0.0669 | 0.2342 | 0.2302 | 0.1073 | 0.0973 |
| 0.0880 | 0.135 | 0.1245 | 0.2344 | 0.2302 | 0.1129 | 0.1354 |
| 0.1000 | 0.199 | 0.2750 | 0.2351 | 0.2302 | 0.1208 | 0.1987 |
| 0.1080 | 0.241 | 0.4377 | 0.2352 | 0.2302 | 0.1258 | 0.2412 |
| 0.1200 | 0.303 | 2.0000 | 0.2359 | 0.2302 | 0.1329 | 0.3028 |

**表 B：对比 α-平滑**（同一容差 r；α=0.35、P=0.9、n=10 为官方默认，置信度与 ContRS 一致）

| r | ContRS R | α-平滑 R | 平滑误差 ContRS | 平滑误差 α-平滑 |
|---|---|---|---|---|
| 0.0800 | 0.0973 | 0.0000 | 0.2342 | 0.2318 |
| 0.0880 | 0.1354 | 0.0000 | 0.2344 | 0.2318 |
| 0.1000 | 0.1987 | 0.0000 | 0.2351 | 0.2318 |
| 0.1080 | 0.2412 | 0.0000 | 0.2352 | 0.2318 |
| 0.1200 | 0.3028 | 0.0000 | 0.2359 | 0.2318 |


---

# 第七部分：全部 95% CI 详细表

以下区间都是逐行、未做多重比较校正的区间。300 图与 50 图分区保留；独立新种子 CI 另列。p 扫描 MTCNN 的 CI 已在 5.2 内嵌，其特殊相关结构限制见 5.3。


## 7.1 300 图主实验与 baseline 的 95% CI

以下来源表已完整内嵌：[ext300/summary/confidence_intervals.md](ext300/summary/confidence_intervals.md)。

#### 95% confidence intervals

图像为独立样本。均值区间为双侧 95% Student-t CI：mean ± t(0.975,n−1)·sd/√n。配对差值先在每张图上相减，再计算 CI。ΔR = ContRS − baseline（越大越有利于 ContRS）；Δerr = ContRS − baseline（越小越有利于 ContRS）。区间不做多重比较校正。

Center smoothing 的 ε_out 仅在未弃权图像上统计，弃权率另用 Wilson 95% CI；safe region 的 ContRS 数值全部取平均 r_SR 处的精确补跑结果。


##### Center smoothing：认证输出界

所有列均为 mean [95% CI]。r 是预设容差，本身没有抽样 CI；ε_out−r 和 ε_out/r 大于 0/1 表示 ContRS 更紧。

| model | r | ContRS R = ε_in | CS ε_out | CS ε_out − r | CS ε_out / r | CS valid n | CS abstain [Wilson 95% CI] |
|---|---:|---|---|---|---|---:|---|
| mtcnn300 | 0.08 | 0.0168 [0.0136, 0.0200] | 0.1570 [0.1496, 0.1643] | 0.0770 [0.0696, 0.0843] | 1.962 [1.870, 2.054] | 297/300 | 1.00% [0.34%, 2.90%] |
| mtcnn300 | 0.1 | 0.0476 [0.0426, 0.0525] | 0.1879 [0.1783, 0.1976] | 0.0879 [0.0783, 0.0976] | 1.879 [1.783, 1.976] | 297/300 | 1.00% [0.34%, 2.90%] |
| mtcnn300 | 0.12 | 0.0821 [0.0760, 0.0881] | 0.2248 [0.2134, 0.2363] | 0.1048 [0.0934, 0.1163] | 1.874 [1.778, 1.969] | 297/300 | 1.00% [0.34%, 2.90%] |
| mtcnn300 | 0.14 | 0.1168 [0.1101, 0.1236] | 0.2679 [0.2541, 0.2817] | 0.1279 [0.1141, 0.1417] | 1.914 [1.815, 2.012] | 297/300 | 1.00% [0.34%, 2.90%] |
| mtcnn300 | 0.16 | 0.1501 [0.1429, 0.1574] | 0.3226 [0.2993, 0.3458] | 0.1626 [0.1393, 0.1858] | 2.016 [1.871, 2.161] | 297/300 | 1.00% [0.34%, 2.90%] |
| cifar_vae300 | 0.012 | 0.1465 [0.1253, 0.1678] | 0.0394 [0.0374, 0.0414] | 0.0274 [0.0254, 0.0294] | 3.285 [3.120, 3.449] | 298/300 | 0.67% [0.18%, 2.40%] |
| cifar_vae300 | 0.014 | 0.2091 [0.1851, 0.2331] | 0.0438 [0.0416, 0.0460] | 0.0298 [0.0276, 0.0320] | 3.130 [2.972, 3.288] | 298/300 | 0.67% [0.18%, 2.40%] |
| cifar_vae300 | 0.016 | 0.2721 [0.2465, 0.2978] | 0.0490 [0.0465, 0.0515] | 0.0330 [0.0305, 0.0355] | 3.065 [2.908, 3.221] | 298/300 | 0.67% [0.18%, 2.40%] |
| cifar_vae300 | 0.018 | 0.3324 [0.3060, 0.3589] | 0.0551 [0.0522, 0.0579] | 0.0371 [0.0342, 0.0399] | 3.059 [2.901, 3.218] | 298/300 | 0.67% [0.18%, 2.40%] |
| cifar_vae300 | 0.02 | 0.3864 [0.3598, 0.4130] | 0.0619 [0.0587, 0.0652] | 0.0419 [0.0387, 0.0452] | 3.096 [2.934, 3.259] | 298/300 | 0.67% [0.18%, 2.40%] |
| clip_l14_300 | 0.08 | 0.0961 [0.0793, 0.1128] | 0.2643 [0.2531, 0.2755] | 0.1843 [0.1731, 0.1955] | 3.304 [3.164, 3.443] | 299/300 | 0.33% [0.06%, 1.86%] |
| clip_l14_300 | 0.088 | 0.1344 [0.1146, 0.1541] | 0.2772 [0.2655, 0.2890] | 0.1892 [0.1775, 0.2010] | 3.150 [3.017, 3.284] | 299/300 | 0.33% [0.06%, 1.86%] |
| clip_l14_300 | 0.1 | 0.1967 [0.1735, 0.2199] | 0.3012 [0.2883, 0.3141] | 0.2012 [0.1883, 0.2141] | 3.012 [2.883, 3.141] | 299/300 | 0.33% [0.06%, 1.86%] |
| clip_l14_300 | 0.108 | 0.2397 [0.2149, 0.2646] | 0.3205 [0.3065, 0.3345] | 0.2125 [0.1985, 0.2265] | 2.968 [2.838, 3.098] | 299/300 | 0.33% [0.06%, 1.86%] |
| clip_l14_300 | 0.12 | 0.3034 [0.2769, 0.3298] | 0.3541 [0.3380, 0.3703] | 0.2341 [0.2180, 0.2503] | 2.951 [2.816, 3.086] | 299/300 | 0.33% [0.06%, 1.86%] |
| cifar_vae300_linf | 0.2852 | 0.0895 [0.0724, 0.1066] | 0.4677 [0.4594, 0.4760] | 0.1825 [0.1742, 0.1908] | 1.640 [1.611, 1.669] | 296/300 | 1.33% [0.52%, 3.38%] |
| cifar_vae300_linf | 0.3097 | 0.1517 [0.1316, 0.1718] | 0.4893 [0.4807, 0.4979] | 0.1796 [0.1710, 0.1882] | 1.580 [1.552, 1.608] | 296/300 | 1.33% [0.52%, 3.38%] |
| cifar_vae300_linf | 0.3364 | 0.2381 [0.2161, 0.2600] | 0.5231 [0.5141, 0.5321] | 0.1867 [0.1778, 0.1957] | 1.555 [1.528, 1.582] | 296/300 | 1.33% [0.52%, 3.38%] |
| cifar_vae300_linf | 0.3653 | 0.3409 [0.3196, 0.3622] | 0.5714 [0.5619, 0.5810] | 0.2061 [0.1966, 0.2157] | 1.564 [1.538, 1.590] | 296/300 | 1.33% [0.52%, 3.38%] |
| cifar_vae300_linf | 0.3967 | 0.4440 [0.4250, 0.4631] | 0.6446 [0.6341, 0.6550] | 0.2478 [0.2373, 0.2583] | 1.625 [1.598, 1.651] | 296/300 | 1.33% [0.52%, 3.38%] |

##### Center smoothing：平滑误差与配对差值

| model | r | ContRS err（全部图） | CS err（未弃权图） | Δerr = ContRS − CS（同一批未弃权图） | paired n |
|---|---:|---|---|---|---:|
| mtcnn300 | 0.08 | 0.0546 [0.0496, 0.0596] | 0.0546 [0.0498, 0.0594] | 0.0003 [-0.0013, 0.0018] | 297 |
| mtcnn300 | 0.1 | 0.0547 [0.0499, 0.0595] | 0.0546 [0.0498, 0.0594] | 0.0004 [-0.0011, 0.0018] | 297 |
| mtcnn300 | 0.12 | 0.0549 [0.0501, 0.0596] | 0.0546 [0.0498, 0.0594] | 0.0005 [-0.0010, 0.0021] | 297 |
| mtcnn300 | 0.14 | 0.0562 [0.0516, 0.0608] | 0.0546 [0.0498, 0.0594] | 0.0019 [0.0002, 0.0037] | 297 |
| mtcnn300 | 0.16 | 0.0577 [0.0532, 0.0621] | 0.0546 [0.0498, 0.0594] | 0.0033 [0.0013, 0.0053] | 297 |
| cifar_vae300 | 0.012 | 0.0014 [0.0013, 0.0015] | 0.0014 [0.0013, 0.0015] | 0.0000 [-0.0000, 0.0001] | 298 |
| cifar_vae300 | 0.014 | 0.0014 [0.0013, 0.0015] | 0.0014 [0.0013, 0.0015] | 0.0000 [-0.0000, 0.0001] | 298 |
| cifar_vae300 | 0.016 | 0.0014 [0.0013, 0.0015] | 0.0014 [0.0013, 0.0015] | 0.0000 [-0.0000, 0.0001] | 298 |
| cifar_vae300 | 0.018 | 0.0015 [0.0014, 0.0016] | 0.0014 [0.0013, 0.0015] | 0.0000 [-0.0000, 0.0001] | 298 |
| cifar_vae300 | 0.02 | 0.0015 [0.0014, 0.0016] | 0.0014 [0.0013, 0.0015] | 0.0001 [0.0000, 0.0001] | 298 |
| clip_l14_300 | 0.08 | 0.2356 [0.2281, 0.2430] | 0.2355 [0.2281, 0.2430] | 0.0001 [-0.0012, 0.0013] | 299 |
| clip_l14_300 | 0.088 | 0.2350 [0.2277, 0.2423] | 0.2355 [0.2281, 0.2430] | -0.0005 [-0.0016, 0.0006] | 299 |
| clip_l14_300 | 0.1 | 0.2349 [0.2275, 0.2422] | 0.2355 [0.2281, 0.2430] | -0.0006 [-0.0017, 0.0004] | 299 |
| clip_l14_300 | 0.108 | 0.2355 [0.2281, 0.2429] | 0.2355 [0.2281, 0.2430] | -0.0000 [-0.0011, 0.0010] | 299 |
| clip_l14_300 | 0.12 | 0.2356 [0.2282, 0.2429] | 0.2355 [0.2281, 0.2430] | 0.0000 [-0.0011, 0.0012] | 299 |
| cifar_vae300_linf | 0.2852 | 0.0880 [0.0857, 0.0903] | 0.0871 [0.0846, 0.0895] | 0.0009 [-0.0000, 0.0019] | 296 |
| cifar_vae300_linf | 0.3097 | 0.0883 [0.0861, 0.0906] | 0.0871 [0.0846, 0.0895] | 0.0013 [0.0003, 0.0023] | 296 |
| cifar_vae300_linf | 0.3364 | 0.0897 [0.0875, 0.0920] | 0.0871 [0.0846, 0.0895] | 0.0027 [0.0016, 0.0038] | 296 |
| cifar_vae300_linf | 0.3653 | 0.0910 [0.0888, 0.0931] | 0.0871 [0.0846, 0.0895] | 0.0039 [0.0027, 0.0052] | 296 |
| cifar_vae300_linf | 0.3967 | 0.0928 [0.0906, 0.0950] | 0.0871 [0.0846, 0.0895] | 0.0057 [0.0043, 0.0072] | 296 |

##### Safe region：平均 r_SR 口径，R 与平滑误差

| model | σ | r = 2κ·mean r_SR [95% CI] | ContRS R | SR R | ΔR = ContRS − SR | ContRS err | SR err | Δerr = ContRS − SR |
|---|---:|---|---|---|---|---|---|---|
| mtcnn300 | 0.1 | 0.0913 [0.0878, 0.0947] | 0.0330 [0.0288, 0.0373] | 0.0086 [0.0084, 0.0089] | 0.0244 [0.0202, 0.0287] | 0.0545 [0.0496, 0.0594] | 0.0538 [0.0492, 0.0583] | 0.0008 [-0.0004, 0.0020] |
| mtcnn300 | 0.2 | 0.1218 [0.1040, 0.1397] | 0.0947 [0.0863, 0.1031] | 0.0214 [0.0154, 0.0273] | 0.0734 [0.0626, 0.0842] | 0.0758 [0.0708, 0.0808] | 0.0776 [0.0699, 0.0853] | -0.0018 [-0.0068, 0.0032] |
| mtcnn300 | 0.3 | 0.2234 [0.1785, 0.2683] | 0.3783 [0.3573, 0.3994] | 0.0710 [0.0481, 0.0940] | 0.3073 [0.2698, 0.3448] | 0.1060 [0.1005, 0.1114] | 0.1261 [0.1089, 0.1433] | -0.0201 [-0.0356, -0.0047] |
| mtcnn300 | 0.4 | 0.5226 [0.4410, 0.6042] | 0.6655 [0.6115, 0.7196] | 0.2760 [0.2190, 0.3330] | 0.3895 [0.2878, 0.4912] | 0.1518 [0.1405, 0.1631] | 0.2517 [0.2165, 0.2869] | -0.0999 [-0.1339, -0.0660] |
| mtcnn300 | 0.5 | 1.1747 [1.0728, 1.2765] | 0.3196 [0.2689, 0.3704] | 0.9040 [0.8136, 0.9944] | -0.5843 [-0.7178, -0.4508] | 0.1866 [0.1703, 0.2029] | 0.5219 [0.4731, 0.5706] | -0.3353 [-0.3828, -0.2877] |
| cifar_vae300 | 0.1 | 0.0021 [0.0020, 0.0022] | 0.0621 [0.0534, 0.0709] | 0.0093 [0.0090, 0.0095] | 0.0528 [0.0441, 0.0616] | 0.0001 [0.0001, 0.0002] | 0.0001 [0.0001, 0.0002] | 0.0000 [-0.0000, 0.0000] |
| cifar_vae300 | 0.2 | 0.0080 [0.0076, 0.0084] | 0.1238 [0.1065, 0.1411] | 0.0181 [0.0175, 0.0186] | 0.1057 [0.0884, 0.1230] | 0.0008 [0.0008, 0.0009] | 0.0008 [0.0008, 0.0009] | 0.0000 [-0.0000, 0.0000] |
| cifar_vae300 | 0.3 | 0.0172 [0.0164, 0.0180] | 0.1840 [0.1583, 0.2097] | 0.0270 [0.0263, 0.0278] | 0.1570 [0.1312, 0.1827] | 0.0023 [0.0021, 0.0024] | 0.0023 [0.0021, 0.0024] | -0.0000 [-0.0001, 0.0001] |
| cifar_vae300 | 0.4 | 0.0290 [0.0276, 0.0304] | 0.2448 [0.2106, 0.2791] | 0.0370 [0.0359, 0.0380] | 0.2079 [0.1736, 0.2422] | 0.0045 [0.0041, 0.0048] | 0.0046 [0.0042, 0.0049] | -0.0001 [-0.0002, 0.0000] |
| cifar_vae300 | 0.5 | 0.0428 [0.0408, 0.0448] | 0.2973 [0.2554, 0.3392] | 0.0455 [0.0442, 0.0468] | 0.2518 [0.2098, 0.2937] | 0.0075 [0.0070, 0.0081] | 0.0078 [0.0072, 0.0083] | -0.0002 [-0.0005, 0.0000] |
| clip_l14_300 | 0.1 | 0.0563 [0.0536, 0.0590] | 0.0731 [0.0633, 0.0829] | 0.0090 [0.0088, 0.0093] | 0.0641 [0.0543, 0.0739] | 0.0914 [0.0872, 0.0956] | 0.0913 [0.0872, 0.0955] | 0.0001 [-0.0005, 0.0007] |
| clip_l14_300 | 0.2 | 0.0858 [0.0822, 0.0893] | 0.1251 [0.1081, 0.1421] | 0.0178 [0.0173, 0.0183] | 0.1073 [0.0903, 0.1242] | 0.1889 [0.1821, 0.1958] | 0.1888 [0.1820, 0.1956] | 0.0001 [-0.0008, 0.0010] |
| clip_l14_300 | 0.3 | 0.0997 [0.0954, 0.1039] | 0.1962 [0.1705, 0.2218] | 0.0268 [0.0261, 0.0276] | 0.1693 [0.1436, 0.1950] | 0.2726 [0.2645, 0.2806] | 0.2727 [0.2647, 0.2808] | -0.0002 [-0.0014, 0.0010] |
| clip_l14_300 | 0.4 | 0.1085 [0.1046, 0.1125] | 0.2326 [0.1996, 0.2657] | 0.0366 [0.0357, 0.0375] | 0.1960 [0.1629, 0.2291] | 0.3306 [0.3215, 0.3397] | 0.3310 [0.3219, 0.3402] | -0.0004 [-0.0015, 0.0007] |
| clip_l14_300 | 0.5 | 0.1132 [0.1094, 0.1170] | 0.2508 [0.2134, 0.2882] | 0.0446 [0.0433, 0.0459] | 0.2062 [0.1689, 0.2435] | 0.3785 [0.3688, 0.3883] | 0.3785 [0.3688, 0.3883] | 0.0000 [-0.0010, 0.0010] |

###### mtcnn300 vs median smoothing（σ=0.1）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0168 [0.0136, 0.0200] | 0.1234 [0.1193, 0.1276] | -0.1066 [-0.1094, -0.1038] | 0.0138 [0.0130, 0.0147] | -0.0662 [-0.0670, -0.0653] | 300/300 | 0.0546 [0.0496, 0.0596] | 0.0531 [0.0490, 0.0573] | 0.0014 [-0.0004, 0.0033] |
| 0.1000 | 0.0476 [0.0426, 0.0525] | 0.1529 [0.1481, 0.1576] | -0.1053 [-0.1080, -0.1026] | 0.0351 [0.0330, 0.0372] | -0.0649 [-0.0670, -0.0628] | 300/300 | 0.0547 [0.0499, 0.0595] | 0.0531 [0.0490, 0.0573] | 0.0016 [-0.0001, 0.0033] |
| 0.1200 | 0.0821 [0.0760, 0.0881] | 0.1816 [0.1763, 0.1869] | -0.0995 [-0.1023, -0.0967] | 0.0586 [0.0557, 0.0616] | -0.0614 [-0.0643, -0.0584] | 300/300 | 0.0549 [0.0501, 0.0596] | 0.0531 [0.0490, 0.0573] | 0.0017 [0.0001, 0.0034] |
| 0.1400 | 0.1168 [0.1101, 0.1236] | 0.2092 [0.2035, 0.2149] | -0.0923 [-0.0953, -0.0893] | 0.0823 [0.0786, 0.0859] | -0.0577 [-0.0614, -0.0541] | 300/300 | 0.0562 [0.0516, 0.0608] | 0.0531 [0.0490, 0.0573] | 0.0031 [0.0015, 0.0047] |
| 0.1600 | 0.1501 [0.1429, 0.1574] | 0.2341 [0.2282, 0.2399] | -0.0839 [-0.0873, -0.0806] | 0.1054 [0.1009, 0.1099] | -0.0546 [-0.0591, -0.0501] | 300/300 | 0.0577 [0.0532, 0.0621] | 0.0531 [0.0490, 0.0573] | 0.0045 [0.0028, 0.0062] |

###### mtcnn300 vs mean smoothing（σ=0.1）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0168 [0.0136, 0.0200] | 0.0025 [0.0025, 0.0025] | 0.0143 [0.0111, 0.0175] | 0.4139 [0.4098, 0.4179] | 0.3339 [0.3298, 0.3379] | 300/300 | 0.0546 [0.0496, 0.0596] | 0.0537 [0.0496, 0.0577] | 0.0009 [-0.0020, 0.0039] |
| 0.1000 | 0.0476 [0.0426, 0.0525] | 0.0033 [0.0033, 0.0034] | 0.0442 [0.0393, 0.0492] | 0.9041 [0.8992, 0.9090] | 0.8041 [0.7992, 0.8090] | 300/300 | 0.0547 [0.0499, 0.0595] | 0.0537 [0.0496, 0.0577] | 0.0010 [-0.0018, 0.0039] |
| 0.1200 | 0.0821 [0.0760, 0.0881] | 0.0041 [0.0041, 0.0042] | 0.0779 [0.0719, 0.0840] | 0.9996 [0.9990, 1.0001] | 0.8796 [0.8790, 0.8801] | 300/300 | 0.0549 [0.0501, 0.0596] | 0.0537 [0.0496, 0.0577] | 0.0012 [-0.0015, 0.0039] |
| 0.1400 | 0.1168 [0.1101, 0.1236] | 0.0050 [0.0049, 0.0050] | 0.1119 [0.1052, 0.1186] | 1.0000 [1.0000, 1.0000] | 0.8600 [0.8600, 0.8600] | 300/300 | 0.0562 [0.0516, 0.0608] | 0.0537 [0.0496, 0.0577] | 0.0026 [-0.0001, 0.0052] |
| 0.1600 | 0.1501 [0.1429, 0.1574] | 0.0058 [0.0057, 0.0059] | 0.1444 [0.1371, 0.1516] | 1.0000 [1.0000, 1.0000] | 0.8400 [0.8400, 0.8400] | 300/300 | 0.0577 [0.0532, 0.0621] | 0.0537 [0.0496, 0.0577] | 0.0040 [0.0014, 0.0066] |

###### mtcnn300 vs alpha smoothing（σ=0.1）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0168 [0.0136, 0.0200] | 0.0000 [0.0000, 0.0000] | 0.0168 [0.0136, 0.0200] | 0.0546 [0.0496, 0.0596] | 0.0554 [0.0516, 0.0593] | -0.0009 [-0.0029, 0.0012] |
| 0.1000 | 0.0476 [0.0426, 0.0525] | 0.0000 [0.0000, 0.0000] | 0.0476 [0.0426, 0.0525] | 0.0547 [0.0499, 0.0595] | 0.0554 [0.0516, 0.0593] | -0.0007 [-0.0026, 0.0011] |
| 0.1200 | 0.0821 [0.0760, 0.0881] | 0.0001 [-0.0001, 0.0004] | 0.0820 [0.0760, 0.0879] | 0.0549 [0.0501, 0.0596] | 0.0554 [0.0516, 0.0593] | -0.0006 [-0.0024, 0.0012] |
| 0.1400 | 0.1168 [0.1101, 0.1236] | 0.0005 [-0.0002, 0.0011] | 0.1164 [0.1098, 0.1230] | 0.0562 [0.0516, 0.0608] | 0.0554 [0.0516, 0.0593] | 0.0008 [-0.0010, 0.0025] |
| 0.1600 | 0.1501 [0.1429, 0.1574] | 0.0009 [-0.0001, 0.0020] | 0.1492 [0.1421, 0.1563] | 0.0577 [0.0532, 0.0621] | 0.0554 [0.0516, 0.0593] | 0.0022 [0.0005, 0.0039] |

###### mtcnn300：ContRS-M vs median smoothing

| r | ContRS-M R | median R | ΔR = ContRS-M − median | ContRS-M err | median err | Δerr = ContRS-M − median |
|---|---:|---|---|---|---|---|
| 0.0800 | 0.0170 [0.0138, 0.0202] | 0.1234 [0.1193, 0.1276] | -0.1065 [-0.1093, -0.1037] | 0.0547 [0.0498, 0.0597] | 0.0531 [0.0490, 0.0573] | 0.0016 [-0.0003, 0.0034] |
| 0.1000 | 0.0477 [0.0428, 0.0526] | 0.1529 [0.1481, 0.1576] | -0.1052 [-0.1078, -0.1025] | 0.0546 [0.0498, 0.0595] | 0.0531 [0.0490, 0.0573] | 0.0015 [-0.0002, 0.0032] |
| 0.1200 | 0.0822 [0.0762, 0.0883] | 0.1816 [0.1763, 0.1869] | -0.0994 [-0.1022, -0.0966] | 0.0549 [0.0502, 0.0596] | 0.0531 [0.0490, 0.0573] | 0.0018 [0.0001, 0.0034] |
| 0.1400 | 0.1169 [0.1102, 0.1236] | 0.2092 [0.2035, 0.2149] | -0.0923 [-0.0952, -0.0893] | 0.0562 [0.0516, 0.0608] | 0.0531 [0.0490, 0.0573] | 0.0030 [0.0014, 0.0047] |
| 0.1600 | 0.1502 [0.1430, 0.1574] | 0.2341 [0.2282, 0.2399] | -0.0839 [-0.0872, -0.0806] | 0.0578 [0.0533, 0.0622] | 0.0531 [0.0490, 0.0573] | 0.0046 [0.0029, 0.0063] |

###### clip_l14_300 vs median smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0961 [0.0793, 0.1128] | 0.4858 [0.4767, 0.4949] | -0.3897 [-0.4000, -0.3795] | 0.0037 [0.0035, 0.0039] | -0.0763 [-0.0765, -0.0761] | 300/300 | 0.2356 [0.2281, 0.2430] | 0.2301 [0.2229, 0.2374] | 0.0054 [0.0042, 0.0067] |
| 0.0880 | 0.1344 [0.1146, 0.1541] | 0.5055 [0.4963, 0.5147] | -0.3712 [-0.3834, -0.3589] | 0.0067 [0.0064, 0.0070] | -0.0813 [-0.0816, -0.0810] | 300/300 | 0.2350 [0.2277, 0.2423] | 0.2301 [0.2229, 0.2374] | 0.0049 [0.0038, 0.0060] |
| 0.1000 | 0.1967 [0.1735, 0.2199] | 0.5323 [0.5229, 0.5416] | -0.3356 [-0.3504, -0.3207] | 0.0137 [0.0131, 0.0143] | -0.0863 [-0.0869, -0.0857] | 300/300 | 0.2349 [0.2275, 0.2422] | 0.2301 [0.2229, 0.2374] | 0.0047 [0.0038, 0.0057] |
| 0.1080 | 0.2397 [0.2149, 0.2646] | 0.5484 [0.5390, 0.5578] | -0.3087 [-0.3249, -0.2925] | 0.0201 [0.0192, 0.0210] | -0.0879 [-0.0888, -0.0870] | 300/300 | 0.2355 [0.2281, 0.2429] | 0.2301 [0.2229, 0.2374] | 0.0054 [0.0044, 0.0063] |
| 0.1200 | 0.3034 [0.2769, 0.3298] | 0.5703 [0.5610, 0.5797] | -0.2670 [-0.2845, -0.2494] | 0.0322 [0.0307, 0.0336] | -0.0878 [-0.0893, -0.0864] | 300/300 | 0.2356 [0.2282, 0.2429] | 0.2301 [0.2229, 0.2374] | 0.0055 [0.0044, 0.0065] |

###### clip_l14_300 vs mean smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0961 [0.0793, 0.1128] | 0.1073 [0.1072, 0.1074] | -0.0113 [-0.0279, 0.0054] | 0.0653 [0.0652, 0.0654] | -0.0147 [-0.0148, -0.0146] | 300/300 | 0.2356 [0.2281, 0.2430] | 0.2302 [0.2230, 0.2375] | 0.0054 [0.0041, 0.0067] |
| 0.0880 | 0.1344 [0.1146, 0.1541] | 0.1129 [0.1128, 0.1130] | 0.0214 [0.0018, 0.0411] | 0.1227 [0.1225, 0.1229] | 0.0347 [0.0345, 0.0349] | 300/300 | 0.2350 [0.2277, 0.2423] | 0.2302 [0.2230, 0.2375] | 0.0048 [0.0037, 0.0059] |
| 0.1000 | 0.1967 [0.1735, 0.2199] | 0.1208 [0.1207, 0.1209] | 0.0759 [0.0528, 0.0990] | 0.2690 [0.2685, 0.2695] | 0.1690 [0.1685, 0.1695] | 300/300 | 0.2349 [0.2275, 0.2422] | 0.2302 [0.2230, 0.2375] | 0.0046 [0.0037, 0.0056] |
| 0.1080 | 0.2397 [0.2149, 0.2646] | 0.1258 [0.1257, 0.1259] | 0.1139 [0.0891, 0.1387] | 0.4308 [0.4299, 0.4317] | 0.3228 [0.3219, 0.3237] | 300/300 | 0.2355 [0.2281, 0.2429] | 0.2302 [0.2230, 0.2375] | 0.0053 [0.0043, 0.0062] |
| 0.1200 | 0.3034 [0.2769, 0.3298] | 0.1329 [0.1328, 0.1330] | 0.1705 [0.1442, 0.1969] | 2.0000 [2.0000, 2.0000] | 1.8800 [1.8800, 1.8800] | 300/300 | 0.2356 [0.2282, 0.2429] | 0.2302 [0.2230, 0.2375] | 0.0054 [0.0043, 0.0064] |

###### clip_l14_300 vs alpha smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0961 [0.0793, 0.1128] | 0.0000 [0.0000, 0.0000] | 0.0961 [0.0793, 0.1128] | 0.2356 [0.2281, 0.2430] | 0.2318 [0.2245, 0.2391] | 0.0038 [0.0025, 0.0050] |
| 0.0880 | 0.1344 [0.1146, 0.1541] | 0.0000 [0.0000, 0.0000] | 0.1344 [0.1146, 0.1541] | 0.2350 [0.2277, 0.2423] | 0.2318 [0.2245, 0.2391] | 0.0032 [0.0021, 0.0043] |
| 0.1000 | 0.1967 [0.1735, 0.2199] | 0.0000 [0.0000, 0.0000] | 0.1967 [0.1735, 0.2199] | 0.2349 [0.2275, 0.2422] | 0.2318 [0.2245, 0.2391] | 0.0031 [0.0021, 0.0040] |
| 0.1080 | 0.2397 [0.2149, 0.2646] | 0.0000 [0.0000, 0.0000] | 0.2397 [0.2149, 0.2646] | 0.2355 [0.2281, 0.2429] | 0.2318 [0.2245, 0.2391] | 0.0037 [0.0027, 0.0046] |
| 0.1200 | 0.3034 [0.2769, 0.3298] | 0.0000 [0.0000, 0.0000] | 0.3034 [0.2769, 0.3298] | 0.2356 [0.2282, 0.2429] | 0.2318 [0.2245, 0.2391] | 0.0038 [0.0027, 0.0048] |

###### clip_l14_300：ContRS-M vs median smoothing

| r | ContRS-M R | median R | ΔR = ContRS-M − median | ContRS-M err | median err | Δerr = ContRS-M − median |
|---|---:|---|---|---|---|---|
| 0.0800 | 0.2203 [0.1983, 0.2423] | 0.4858 [0.4767, 0.4949] | -0.2655 [-0.2790, -0.2520] | 0.2301 [0.2229, 0.2374] | 0.2301 [0.2229, 0.2374] | 0.0000 [-0.0000, 0.0001] |
| 0.0880 | 0.2704 [0.2473, 0.2936] | 0.5055 [0.4963, 0.5147] | -0.2351 [-0.2495, -0.2208] | 0.2302 [0.2229, 0.2374] | 0.2301 [0.2229, 0.2374] | 0.0000 [-0.0000, 0.0001] |
| 0.1000 | 0.3404 [0.3167, 0.3641] | 0.5323 [0.5229, 0.5416] | -0.1919 [-0.2066, -0.1772] | 0.2302 [0.2229, 0.2374] | 0.2301 [0.2229, 0.2374] | 0.0001 [-0.0001, 0.0002] |
| 0.1080 | 0.3841 [0.3606, 0.4076] | 0.5484 [0.5390, 0.5578] | -0.1644 [-0.1788, -0.1499] | 0.2303 [0.2231, 0.2375] | 0.2301 [0.2229, 0.2374] | 0.0002 [0.0000, 0.0004] |
| 0.1200 | 0.4426 [0.4198, 0.4653] | 0.5703 [0.5610, 0.5797] | -0.1278 [-0.1416, -0.1140] | 0.2304 [0.2232, 0.2377] | 0.2301 [0.2229, 0.2374] | 0.0003 [0.0001, 0.0006] |

###### cifar_vae300_linf vs median smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.2852 | 0.0895 [0.0724, 0.1066] | 0.7282 [0.7272, 0.7292] | -0.6387 [-0.6555, -0.6219] | 0.0273 [0.0268, 0.0277] | -0.2579 [-0.2584, -0.2574] | 300/300 | 0.0880 [0.0857, 0.0903] | 0.0468 [0.0442, 0.0494] | 0.0412 [0.0393, 0.0431] |
| 0.3097 | 0.1517 [0.1316, 0.1718] | 0.7314 [0.7309, 0.7319] | -0.5797 [-0.5997, -0.5597] | 0.0435 [0.0427, 0.0442] | -0.2662 [-0.2670, -0.2655] | 300/300 | 0.0883 [0.0861, 0.0906] | 0.0468 [0.0442, 0.0494] | 0.0415 [0.0396, 0.0435] |
| 0.3364 | 0.2381 [0.2161, 0.2600] | 0.7326 [0.7324, 0.7327] | -0.4945 [-0.5164, -0.4725] | 0.0663 [0.0651, 0.0674] | -0.2701 [-0.2712, -0.2689] | 300/300 | 0.0897 [0.0875, 0.0920] | 0.0468 [0.0442, 0.0494] | 0.0429 [0.0411, 0.0448] |
| 0.3653 | 0.3409 [0.3196, 0.3622] | 0.7327 [0.7326, 0.7328] | -0.3918 [-0.4131, -0.3705] | 0.0941 [0.0925, 0.0957] | -0.2712 [-0.2728, -0.2696] | 300/300 | 0.0910 [0.0888, 0.0931] | 0.0468 [0.0442, 0.0494] | 0.0442 [0.0423, 0.0461] |
| 0.3967 | 0.4440 [0.4250, 0.4631] | 0.7328 [0.7328, 0.7328] | -0.2887 [-0.3078, -0.2697] | 0.1232 [0.1211, 0.1253] | -0.2735 [-0.2756, -0.2714] | 300/300 | 0.0928 [0.0906, 0.0950] | 0.0468 [0.0442, 0.0494] | 0.0460 [0.0440, 0.0480] |

###### cifar_vae300_linf vs mean smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.2852 | 0.0895 [0.0724, 0.1066] | 0.0866 [0.0866, 0.0867] | 0.0029 [-0.0142, 0.0199] | 0.2942 [0.2941, 0.2943] | 0.0090 [0.0089, 0.0091] | 300/300 | 0.0880 [0.0857, 0.0903] | 0.0472 [0.0446, 0.0498] | 0.0408 [0.0389, 0.0427] |
| 0.3097 | 0.1517 [0.1316, 0.1718] | 0.0945 [0.0944, 0.0945] | 0.0572 [0.0371, 0.0774] | 0.4868 [0.4867, 0.4868] | 0.1770 [0.1770, 0.1771] | 300/300 | 0.0883 [0.0861, 0.0906] | 0.0472 [0.0446, 0.0498] | 0.0411 [0.0393, 0.0430] |
| 0.3364 | 0.2381 [0.2161, 0.2600] | 0.1030 [0.1029, 0.1030] | 0.1351 [0.1132, 0.1571] | 0.7417 [0.7417, 0.7418] | 0.4054 [0.4053, 0.4054] | 300/300 | 0.0897 [0.0875, 0.0920] | 0.0472 [0.0446, 0.0498] | 0.0426 [0.0407, 0.0444] |
| 0.3653 | 0.3409 [0.3196, 0.3622] | 0.1123 [0.1122, 0.1123] | 0.2287 [0.2074, 0.2500] | 1.0184 [1.0181, 1.0186] | 0.6531 [0.6528, 0.6533] | 300/300 | 0.0910 [0.0888, 0.0931] | 0.0472 [0.0446, 0.0498] | 0.0438 [0.0419, 0.0456] |
| 0.3967 | 0.4440 [0.4250, 0.4631] | 0.1224 [0.1224, 0.1224] | 0.3217 [0.3026, 0.3407] | 1.2585 [1.2575, 1.2594] | 0.8617 [0.8608, 0.8627] | 300/300 | 0.0928 [0.0906, 0.0950] | 0.0472 [0.0446, 0.0498] | 0.0456 [0.0436, 0.0476] |

###### cifar_vae300_linf vs alpha smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---|
| 0.2852 | 0.0895 [0.0724, 0.1066] | 0.0535 [0.0425, 0.0645] | 0.0360 [0.0280, 0.0440] | 0.0880 [0.0857, 0.0903] | 0.0646 [0.0627, 0.0666] | 0.0234 [0.0219, 0.0248] |
| 0.3097 | 0.1517 [0.1316, 0.1718] | 0.1023 [0.0882, 0.1163] | 0.0494 [0.0410, 0.0579] | 0.0883 [0.0861, 0.0906] | 0.0646 [0.0627, 0.0666] | 0.0237 [0.0223, 0.0252] |
| 0.3364 | 0.2381 [0.2161, 0.2600] | 0.1711 [0.1547, 0.1874] | 0.0670 [0.0582, 0.0758] | 0.0897 [0.0875, 0.0920] | 0.0646 [0.0627, 0.0666] | 0.0251 [0.0237, 0.0266] |
| 0.3653 | 0.3409 [0.3196, 0.3622] | 0.2563 [0.2393, 0.2732] | 0.0847 [0.0760, 0.0933] | 0.0910 [0.0888, 0.0931] | 0.0646 [0.0627, 0.0666] | 0.0263 [0.0249, 0.0278] |
| 0.3967 | 0.4440 [0.4250, 0.4631] | 0.3461 [0.3300, 0.3622] | 0.0979 [0.0903, 0.1056] | 0.0928 [0.0906, 0.0950] | 0.0646 [0.0627, 0.0666] | 0.0282 [0.0266, 0.0297] |

###### cifar_vae300_linf：ContRS-M vs median smoothing

| r | ContRS-M R | median R | ΔR = ContRS-M − median | ContRS-M err | median err | Δerr = ContRS-M − median |
|---|---:|---|---|---|---|---|
| 0.2852 | 0.1875 [0.1690, 0.2061] | 0.7282 [0.7272, 0.7292] | -0.5407 [-0.5589, -0.5225] | 0.0478 [0.0452, 0.0504] | 0.0468 [0.0442, 0.0494] | 0.0010 [0.0005, 0.0015] |
| 0.3097 | 0.2805 [0.2622, 0.2989] | 0.7314 [0.7309, 0.7319] | -0.4509 [-0.4691, -0.4327] | 0.0479 [0.0453, 0.0506] | 0.0468 [0.0442, 0.0494] | 0.0011 [0.0006, 0.0017] |
| 0.3364 | 0.3773 [0.3600, 0.3946] | 0.7326 [0.7324, 0.7327] | -0.3553 [-0.3726, -0.3380] | 0.0487 [0.0461, 0.0514] | 0.0468 [0.0442, 0.0494] | 0.0019 [0.0011, 0.0028] |
| 0.3653 | 0.4683 [0.4525, 0.4841] | 0.7327 [0.7326, 0.7328] | -0.2644 [-0.2802, -0.2486] | 0.0493 [0.0466, 0.0520] | 0.0468 [0.0442, 0.0494] | 0.0025 [0.0015, 0.0035] |
| 0.3967 | 0.5524 [0.5385, 0.5662] | 0.7328 [0.7328, 0.7328] | -0.1804 [-0.1942, -0.1666] | 0.0509 [0.0481, 0.0537] | 0.0468 [0.0442, 0.0494] | 0.0041 [0.0027, 0.0055] |

##### 配对差值 CI 概览（未做多重比较校正）

- Safe region ΔR：14/15 行 CI 全为正；1/15 行 CI 全为负；其余跨 0。
- Center smoothing ε_out−r：20/20 行 CI 全为正。
- ContRS vs median 的同容差 ΔR：0/15 行 CI 全为正；15/15 行 CI 全为负；其余跨 0。
- ContRS vs mean 的同容差 ΔR：13/15 行 CI 全为正；0/15 行 CI 全为负；其余跨 0。
- ContRS vs alpha 的同容差 ΔR：15/15 行 CI 全为正；0/15 行 CI 全为负；其余跨 0。


## 7.2 50 图多架构与早期 baseline 的 95% CI

本文件同时含多架构和早期 baseline 的区间；具体 n 以每行 `valid n`、`paired n` 和第三部分完成度表为准。

以下来源表已完整内嵌：[rebuttal/summary/confidence_intervals.md](rebuttal/summary/confidence_intervals.md)。

#### 95% confidence intervals

图像为独立样本。均值区间为双侧 95% Student-t CI：mean ± t(0.975,n−1)·sd/√n。配对差值先在每张图上相减，再计算 CI。ΔR = ContRS − baseline（越大越有利于 ContRS）；Δerr = ContRS − baseline（越小越有利于 ContRS）。区间不做多重比较校正。

Center smoothing 的 ε_out 仅在未弃权图像上统计，弃权率另用 Wilson 95% CI；safe region 的 ContRS 数值全部取平均 r_SR 处的精确补跑结果。


##### Center smoothing：认证输出界

所有列均为 mean [95% CI]。r 是预设容差，本身没有抽样 CI；ε_out−r 和 ε_out/r 大于 0/1 表示 ContRS 更紧。

| model | r | ContRS R = ε_in | CS ε_out | CS ε_out − r | CS ε_out / r | CS valid n | CS abstain [Wilson 95% CI] |
|---|---:|---|---|---|---|---:|---|
| mtcnn | 0.08 | 0.0543 [0.0351, 0.0735] | 0.1669 [0.1446, 0.1893] | 0.0869 [0.0646, 0.1093] | 2.087 [1.808, 2.366] | 48/50 | 4.00% [1.10%, 13.46%] |
| mtcnn | 0.1 | 0.0862 [0.0649, 0.1075] | 0.2092 [0.1779, 0.2405] | 0.1092 [0.0779, 0.1405] | 2.092 [1.779, 2.405] | 48/50 | 4.00% [1.10%, 13.46%] |
| mtcnn | 0.12 | 0.1240 [0.1011, 0.1470] | 0.2585 [0.2216, 0.2955] | 0.1385 [0.1016, 0.1755] | 2.155 [1.846, 2.463] | 48/50 | 4.00% [1.10%, 13.46%] |
| mtcnn | 0.14 | 0.1630 [0.1385, 0.1876] | 0.3215 [0.2795, 0.3635] | 0.1815 [0.1395, 0.2235] | 2.296 [1.996, 2.596] | 48/50 | 4.00% [1.10%, 13.46%] |
| mtcnn | 0.16 | 0.1944 [0.1701, 0.2187] | 0.3982 [0.3447, 0.4517] | 0.2382 [0.1847, 0.2917] | 2.489 [2.154, 2.823] | 48/50 | 4.00% [1.10%, 13.46%] |
| retinaface | 0.02975 | 0.0375 [0.0246, 0.0503] | 0.0631 [0.0500, 0.0761] | 0.0333 [0.0203, 0.0464] | 2.120 [1.681, 2.559] | 49/50 | 2.00% [0.35%, 10.50%] |
| retinaface | 0.03439 | 0.0638 [0.0470, 0.0806] | 0.0702 [0.0555, 0.0848] | 0.0358 [0.0211, 0.0504] | 2.040 [1.615, 2.466] | 49/50 | 2.00% [0.35%, 10.50%] |
| retinaface | 0.03976 | 0.0957 [0.0749, 0.1165] | 0.0796 [0.0628, 0.0963] | 0.0398 [0.0230, 0.0566] | 2.001 [1.580, 2.423] | 49/50 | 2.00% [0.35%, 10.50%] |
| retinaface | 0.04596 | 0.1317 [0.1081, 0.1554] | 0.0921 [0.0726, 0.1116] | 0.0461 [0.0266, 0.0656] | 2.004 [1.580, 2.428] | 49/50 | 2.00% [0.35%, 10.50%] |
| retinaface | 0.05313 | 0.1681 [0.1424, 0.1939] | 0.1097 [0.0848, 0.1346] | 0.0566 [0.0317, 0.0814] | 2.065 [1.597, 2.533] | 49/50 | 2.00% [0.35%, 10.50%] |
| detr | 0.0134 | 0.0307 [0.0080, 0.0534] | 0.4629 [0.1829, 0.7428] | 0.4495 [0.1695, 0.7294] | 34.540 [13.651, 55.428] | 50/50 | 0.00% [0.00%, 7.13%] |
| detr | 0.0207 | 0.0542 [0.0266, 0.0818] | 0.5956 [0.2831, 0.9080] | 0.5748 [0.2624, 0.8873] | 28.766 [13.672, 43.859] | 50/50 | 0.00% [0.00%, 7.13%] |
| detr | 0.03199 | 0.0981 [0.0657, 0.1305] | 0.8203 [0.4689, 1.1717] | 0.7883 [0.4369, 1.1397] | 25.646 [14.659, 36.633] | 50/50 | 0.00% [0.00%, 7.13%] |
| detr | 0.04596 | 0.1395 [0.1046, 0.1743] | 0.8619 [0.5081, 1.2157] | 0.8160 [0.4621, 1.1698] | 18.754 [11.055, 26.452] | 50/50 | 0.00% [0.00%, 7.13%] |
| detr | 0.071 | 0.1793 [0.1424, 0.2162] | 0.9709 [0.6052, 1.3365] | 0.8999 [0.5342, 1.2655] | 13.673 [8.524, 18.823] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae | 0.012 | 0.1985 [0.1374, 0.2595] | 0.0394 [0.0341, 0.0447] | 0.0274 [0.0221, 0.0327] | 3.286 [2.843, 3.728] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae | 0.014 | 0.2640 [0.1975, 0.3305] | 0.0443 [0.0382, 0.0504] | 0.0303 [0.0242, 0.0364] | 3.166 [2.732, 3.601] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae | 0.016 | 0.3222 [0.2523, 0.3920] | 0.0496 [0.0426, 0.0565] | 0.0336 [0.0266, 0.0405] | 3.099 [2.664, 3.534] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae | 0.018 | 0.3757 [0.3052, 0.4462] | 0.0556 [0.0477, 0.0635] | 0.0376 [0.0297, 0.0455] | 3.089 [2.648, 3.530] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae | 0.02 | 0.4331 [0.3628, 0.5034] | 0.0645 [0.0551, 0.0740] | 0.0445 [0.0351, 0.0540] | 3.226 [2.754, 3.699] | 50/50 | 0.00% [0.00%, 7.13%] |
| sd_vae | 0.0595 | 0.0892 [0.0418, 0.1366] | 0.2222 [0.1911, 0.2532] | 0.1627 [0.1316, 0.1937] | 3.734 [3.213, 4.256] | 50/50 | 0.00% [0.00%, 7.13%] |
| sd_vae | 0.06878 | 0.1428 [0.0860, 0.1996] | 0.2360 [0.2031, 0.2689] | 0.1672 [0.1344, 0.2001] | 3.431 [2.954, 3.909] | 50/50 | 0.00% [0.00%, 7.13%] |
| sd_vae | 0.08549 | 0.2487 [0.1775, 0.3199] | 0.2691 [0.2318, 0.3063] | 0.1836 [0.1463, 0.2209] | 3.147 [2.711, 3.583] | 50/50 | 0.00% [0.00%, 7.13%] |
| sd_vae | 0.09883 | 0.3335 [0.2576, 0.4093] | 0.3037 [0.2618, 0.3456] | 0.2048 [0.1629, 0.2468] | 3.073 [2.649, 3.497] | 50/50 | 0.00% [0.00%, 7.13%] |
| sd_vae | 0.1143 | 0.4214 [0.3437, 0.4991] | 0.3567 [0.3076, 0.4058] | 0.2425 [0.1934, 0.2916] | 3.122 [2.693, 3.552] | 50/50 | 0.00% [0.00%, 7.13%] |
| mae | 0.0595 | 0.0906 [0.0509, 0.1304] | 0.1927 [0.1723, 0.2131] | 0.1332 [0.1128, 0.1536] | 3.239 [2.896, 3.582] | 50/50 | 0.00% [0.00%, 7.13%] |
| mae | 0.06878 | 0.1631 [0.1105, 0.2157] | 0.2071 [0.1856, 0.2287] | 0.1384 [0.1168, 0.1599] | 3.012 [2.699, 3.325] | 50/50 | 0.00% [0.00%, 7.13%] |
| mae | 0.07395 | 0.2024 [0.1431, 0.2616] | 0.2159 [0.1937, 0.2381] | 0.1420 [0.1197, 0.1642] | 2.920 [2.619, 3.220] | 50/50 | 0.00% [0.00%, 7.13%] |
| mae | 0.08549 | 0.3057 [0.2411, 0.3702] | 0.2437 [0.2193, 0.2680] | 0.1582 [0.1338, 0.1826] | 2.850 [2.565, 3.135] | 50/50 | 0.00% [0.00%, 7.13%] |
| mae | 0.09883 | 0.4158 [0.3484, 0.4832] | 0.2884 [0.2603, 0.3165] | 0.1896 [0.1614, 0.2177] | 2.918 [2.634, 3.203] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_l14 | 0.08 | 0.0796 [0.0425, 0.1166] | 0.2853 [0.2551, 0.3156] | 0.2053 [0.1751, 0.2356] | 3.567 [3.189, 3.944] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_l14 | 0.088 | 0.1088 [0.0621, 0.1554] | 0.2954 [0.2642, 0.3266] | 0.2074 [0.1762, 0.2386] | 3.357 [3.002, 3.711] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_l14 | 0.1 | 0.1555 [0.0982, 0.2128] | 0.3128 [0.2800, 0.3456] | 0.2128 [0.1800, 0.2456] | 3.128 [2.800, 3.456] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_l14 | 0.108 | 0.1889 [0.1250, 0.2528] | 0.3261 [0.2920, 0.3601] | 0.2181 [0.1840, 0.2521] | 3.019 [2.704, 3.335] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_l14 | 0.12 | 0.2425 [0.1731, 0.3119] | 0.3499 [0.3137, 0.3861] | 0.2299 [0.1937, 0.2661] | 2.916 [2.614, 3.218] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_rn50x16 | 0.06397 | 0.0789 [0.0364, 0.1215] | 0.2467 [0.2185, 0.2748] | 0.1827 [0.1546, 0.2108] | 3.856 [3.416, 4.295] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_rn50x16 | 0.08 | 0.1493 [0.0890, 0.2097] | 0.2691 [0.2375, 0.3007] | 0.1891 [0.1575, 0.2207] | 3.363 [2.968, 3.758] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_rn50x16 | 0.09883 | 0.2422 [0.1681, 0.3163] | 0.3066 [0.2672, 0.3460] | 0.2078 [0.1684, 0.2472] | 3.103 [2.704, 3.501] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_rn50x16 | 0.1143 | 0.3215 [0.2445, 0.3985] | 0.3472 [0.2997, 0.3947] | 0.2330 [0.1855, 0.2805] | 3.039 [2.623, 3.455] | 50/50 | 0.00% [0.00%, 7.13%] |
| clip_rn50x16 | 0.142 | 0.4425 [0.3666, 0.5185] | 0.4433 [0.3829, 0.5038] | 0.3013 [0.2409, 0.3618] | 3.122 [2.696, 3.548] | 50/50 | 0.00% [0.00%, 7.13%] |
| siglip | 0.07951 | 0.0774 [0.0343, 0.1205] | 0.2638 [0.2414, 0.2861] | 0.1842 [0.1619, 0.2066] | 3.317 [3.036, 3.598] | 50/50 | 0.00% [0.00%, 7.13%] |
| siglip | 0.088 | 0.1145 [0.0627, 0.1662] | 0.2745 [0.2512, 0.2978] | 0.1865 [0.1632, 0.2098] | 3.119 [2.854, 3.384] | 50/50 | 0.00% [0.00%, 7.13%] |
| siglip | 0.1 | 0.1705 [0.1101, 0.2309] | 0.2925 [0.2676, 0.3175] | 0.1925 [0.1676, 0.2175] | 2.925 [2.676, 3.175] | 50/50 | 0.00% [0.00%, 7.13%] |
| siglip | 0.12 | 0.2750 [0.2069, 0.3430] | 0.3328 [0.3043, 0.3612] | 0.2128 [0.1843, 0.2412] | 2.773 [2.536, 3.010] | 50/50 | 0.00% [0.00%, 7.13%] |
| siglip | 0.142 | 0.4015 [0.3363, 0.4668] | 0.4067 [0.3713, 0.4421] | 0.2646 [0.2292, 0.3000] | 2.864 [2.614, 3.113] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae_linf | 0.2852 | 0.0891 [0.0413, 0.1370] | 0.4712 [0.4486, 0.4938] | 0.1860 [0.1634, 0.2086] | 1.652 [1.573, 1.732] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae_linf | 0.3097 | 0.1480 [0.0926, 0.2033] | 0.4916 [0.4682, 0.5150] | 0.1819 [0.1585, 0.2053] | 1.587 [1.512, 1.663] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae_linf | 0.3364 | 0.2255 [0.1673, 0.2837] | 0.5214 [0.4969, 0.5458] | 0.1850 [0.1606, 0.2095] | 1.550 [1.477, 1.623] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae_linf | 0.3653 | 0.3275 [0.2720, 0.3829] | 0.5685 [0.5422, 0.5947] | 0.2032 [0.1769, 0.2294] | 1.556 [1.484, 1.628] | 50/50 | 0.00% [0.00%, 7.13%] |
| cifar_vae_linf | 0.3967 | 0.4314 [0.3818, 0.4809] | 0.6350 [0.6062, 0.6638] | 0.2382 [0.2094, 0.2671] | 1.601 [1.528, 1.673] | 50/50 | 0.00% [0.00%, 7.13%] |

##### Center smoothing：平滑误差与配对差值

| model | r | ContRS err（全部图） | CS err（未弃权图） | Δerr = ContRS − CS（同一批未弃权图） | paired n |
|---|---:|---|---|---|---:|
| mtcnn | 0.08 | 0.0503 [0.0364, 0.0642] | 0.0504 [0.0357, 0.0651] | 0.0007 [-0.0026, 0.0039] | 48 |
| mtcnn | 0.1 | 0.0532 [0.0399, 0.0664] | 0.0504 [0.0357, 0.0651] | 0.0038 [0.0010, 0.0066] | 48 |
| mtcnn | 0.12 | 0.0534 [0.0403, 0.0664] | 0.0504 [0.0357, 0.0651] | 0.0041 [-0.0003, 0.0086] | 48 |
| mtcnn | 0.14 | 0.0540 [0.0413, 0.0668] | 0.0504 [0.0357, 0.0651] | 0.0048 [-0.0000, 0.0097] | 48 |
| mtcnn | 0.16 | 0.0528 [0.0402, 0.0654] | 0.0504 [0.0357, 0.0651] | 0.0034 [-0.0019, 0.0086] | 48 |
| retinaface | 0.02975 | 0.0446 [0.0308, 0.0584] | 0.0443 [0.0306, 0.0580] | 0.0005 [-0.0000, 0.0011] | 49 |
| retinaface | 0.03439 | 0.0446 [0.0308, 0.0584] | 0.0443 [0.0306, 0.0580] | 0.0005 [-0.0000, 0.0011] | 49 |
| retinaface | 0.03976 | 0.0448 [0.0309, 0.0586] | 0.0443 [0.0306, 0.0580] | 0.0007 [0.0001, 0.0013] | 49 |
| retinaface | 0.04596 | 0.0449 [0.0310, 0.0587] | 0.0443 [0.0306, 0.0580] | 0.0008 [0.0001, 0.0015] | 49 |
| retinaface | 0.05313 | 0.0447 [0.0308, 0.0586] | 0.0443 [0.0306, 0.0580] | 0.0006 [-0.0001, 0.0013] | 49 |
| detr | 0.0134 | 0.1631 [0.0732, 0.2530] | 0.1717 [0.0756, 0.2678] | -0.0087 [-0.0649, 0.0476] | 50 |
| detr | 0.0207 | 0.1460 [0.0630, 0.2290] | 0.1717 [0.0756, 0.2678] | -0.0257 [-0.0688, 0.0173] | 50 |
| detr | 0.03199 | 0.1413 [0.0582, 0.2244] | 0.1717 [0.0756, 0.2678] | -0.0304 [-0.0722, 0.0113] | 50 |
| detr | 0.04596 | 0.1415 [0.0584, 0.2246] | 0.1717 [0.0756, 0.2678] | -0.0302 [-0.0717, 0.0112] | 50 |
| detr | 0.071 | 0.1417 [0.0587, 0.2246] | 0.1717 [0.0756, 0.2678] | -0.0301 [-0.0717, 0.0115] | 50 |
| cifar_vae | 0.012 | 0.0013 [0.0011, 0.0016] | 0.0013 [0.0011, 0.0016] | 0.0000 [-0.0001, 0.0001] | 50 |
| cifar_vae | 0.014 | 0.0013 [0.0011, 0.0015] | 0.0013 [0.0011, 0.0016] | -0.0000 [-0.0001, 0.0001] | 50 |
| cifar_vae | 0.016 | 0.0013 [0.0011, 0.0016] | 0.0013 [0.0011, 0.0016] | -0.0000 [-0.0001, 0.0001] | 50 |
| cifar_vae | 0.018 | 0.0013 [0.0011, 0.0015] | 0.0013 [0.0011, 0.0016] | -0.0000 [-0.0001, 0.0000] | 50 |
| cifar_vae | 0.02 | 0.0014 [0.0011, 0.0016] | 0.0013 [0.0011, 0.0016] | 0.0000 [-0.0000, 0.0001] | 50 |
| sd_vae | 0.0595 | 0.0264 [0.0202, 0.0326] | 0.0253 [0.0202, 0.0303] | 0.0011 [-0.0010, 0.0032] | 50 |
| sd_vae | 0.06878 | 0.0252 [0.0199, 0.0305] | 0.0253 [0.0202, 0.0303] | -0.0001 [-0.0016, 0.0014] | 50 |
| sd_vae | 0.08549 | 0.0270 [0.0213, 0.0326] | 0.0253 [0.0202, 0.0303] | 0.0017 [0.0003, 0.0032] | 50 |
| sd_vae | 0.09883 | 0.0270 [0.0216, 0.0324] | 0.0253 [0.0202, 0.0303] | 0.0017 [0.0005, 0.0030] | 50 |
| sd_vae | 0.1143 | 0.0274 [0.0219, 0.0328] | 0.0253 [0.0202, 0.0303] | 0.0021 [0.0009, 0.0033] | 50 |
| mae | 0.0595 | 0.0294 [0.0228, 0.0360] | 0.0284 [0.0228, 0.0339] | 0.0010 [-0.0008, 0.0029] | 50 |
| mae | 0.06878 | 0.0295 [0.0234, 0.0357] | 0.0284 [0.0228, 0.0339] | 0.0012 [-0.0001, 0.0024] | 50 |
| mae | 0.07395 | 0.0293 [0.0232, 0.0354] | 0.0284 [0.0228, 0.0339] | 0.0009 [-0.0003, 0.0021] | 50 |
| mae | 0.08549 | 0.0300 [0.0240, 0.0360] | 0.0284 [0.0228, 0.0339] | 0.0016 [0.0003, 0.0029] | 50 |
| mae | 0.09883 | 0.0303 [0.0249, 0.0357] | 0.0284 [0.0228, 0.0339] | 0.0019 [0.0009, 0.0030] | 50 |
| clip_l14 | 0.08 | 0.2491 [0.2279, 0.2703] | 0.2464 [0.2252, 0.2676] | 0.0027 [-0.0015, 0.0069] | 50 |
| clip_l14 | 0.088 | 0.2472 [0.2263, 0.2682] | 0.2464 [0.2252, 0.2676] | 0.0008 [-0.0015, 0.0032] | 50 |
| clip_l14 | 0.1 | 0.2462 [0.2254, 0.2671] | 0.2464 [0.2252, 0.2676] | -0.0002 [-0.0026, 0.0023] | 50 |
| clip_l14 | 0.108 | 0.2462 [0.2256, 0.2668] | 0.2464 [0.2252, 0.2676] | -0.0002 [-0.0025, 0.0022] | 50 |
| clip_l14 | 0.12 | 0.2465 [0.2259, 0.2672] | 0.2464 [0.2252, 0.2676] | 0.0001 [-0.0021, 0.0023] | 50 |
| clip_rn50x16 | 0.06397 | 0.2257 [0.2071, 0.2443] | 0.2257 [0.2072, 0.2441] | 0.0000 [-0.0032, 0.0033] | 50 |
| clip_rn50x16 | 0.08 | 0.2244 [0.2061, 0.2427] | 0.2257 [0.2072, 0.2441] | -0.0012 [-0.0039, 0.0014] | 50 |
| clip_rn50x16 | 0.09883 | 0.2260 [0.2076, 0.2445] | 0.2257 [0.2072, 0.2441] | 0.0004 [-0.0022, 0.0029] | 50 |
| clip_rn50x16 | 0.1143 | 0.2268 [0.2084, 0.2451] | 0.2257 [0.2072, 0.2441] | 0.0011 [-0.0014, 0.0036] | 50 |
| clip_rn50x16 | 0.142 | 0.2278 [0.2104, 0.2453] | 0.2257 [0.2072, 0.2441] | 0.0022 [-0.0009, 0.0053] | 50 |
| siglip | 0.07951 | 0.2603 [0.2396, 0.2811] | 0.2625 [0.2412, 0.2838] | -0.0021 [-0.0047, 0.0004] | 50 |
| siglip | 0.088 | 0.2618 [0.2400, 0.2835] | 0.2625 [0.2412, 0.2838] | -0.0007 [-0.0035, 0.0021] | 50 |
| siglip | 0.1 | 0.2626 [0.2411, 0.2841] | 0.2625 [0.2412, 0.2838] | 0.0001 [-0.0020, 0.0022] | 50 |
| siglip | 0.12 | 0.2630 [0.2415, 0.2846] | 0.2625 [0.2412, 0.2838] | 0.0006 [-0.0018, 0.0029] | 50 |
| siglip | 0.142 | 0.2641 [0.2427, 0.2854] | 0.2625 [0.2412, 0.2838] | 0.0016 [-0.0013, 0.0045] | 50 |
| cifar_vae_linf | 0.2852 | 0.0904 [0.0838, 0.0970] | 0.0897 [0.0834, 0.0960] | 0.0007 [-0.0019, 0.0033] | 50 |
| cifar_vae_linf | 0.3097 | 0.0900 [0.0831, 0.0968] | 0.0897 [0.0834, 0.0960] | 0.0002 [-0.0025, 0.0029] | 50 |
| cifar_vae_linf | 0.3364 | 0.0920 [0.0854, 0.0986] | 0.0897 [0.0834, 0.0960] | 0.0023 [-0.0010, 0.0055] | 50 |
| cifar_vae_linf | 0.3653 | 0.0951 [0.0881, 0.1021] | 0.0897 [0.0834, 0.0960] | 0.0054 [0.0019, 0.0089] | 50 |
| cifar_vae_linf | 0.3967 | 0.0955 [0.0892, 0.1018] | 0.0897 [0.0834, 0.0960] | 0.0058 [0.0015, 0.0100] | 50 |

##### Safe region：平均 r_SR 口径，R 与平滑误差

| model | σ | r = 2κ·mean r_SR [95% CI] | ContRS R | SR R | ΔR = ContRS − SR | ContRS err | SR err | Δerr = ContRS − SR |
|---|---:|---|---|---|---|---|---|---|
| mtcnn | 0.1 | 0.0724 [0.0628, 0.0820] | 0.0447 [0.0259, 0.0636] | 0.0089 [0.0083, 0.0095] | 0.0359 [0.0170, 0.0547] | 0.0501 [0.0361, 0.0642] | 0.0485 [0.0346, 0.0625] | 0.0016 [-0.0010, 0.0042] |
| mtcnn | 0.2 | 0.0923 [0.0802, 0.1043] | 0.0884 [0.0583, 0.1185] | 0.0181 [0.0167, 0.0194] | 0.0703 [0.0403, 0.1003] | 0.0595 [0.0477, 0.0713] | 0.0577 [0.0457, 0.0696] | 0.0018 [-0.0003, 0.0040] |
| mtcnn | 0.3 | 0.1828 [0.0756, 0.2899] | 0.4483 [0.3747, 0.5219] | 0.0642 [0.0103, 0.1180] | 0.3842 [0.2789, 0.4894] | 0.0763 [0.0619, 0.0908] | 0.1054 [0.0518, 0.1589] | -0.0290 [-0.0764, 0.0183] |
| mtcnn | 0.4 | 0.3418 [0.1657, 0.5179] | 0.9364 [0.8114, 1.0614] | 0.1867 [0.0677, 0.3058] | 0.7497 [0.5182, 0.9811] | 0.0915 [0.0790, 0.1039] | 0.1735 [0.0935, 0.2535] | -0.0820 [-0.1587, -0.0054] |
| mtcnn | 0.5 | 0.5048 [0.2897, 0.7198] | 1.1223 [0.9313, 1.3134] | 0.3607 [0.1775, 0.5439] | 0.7616 [0.4023, 1.1209] | 0.1286 [0.0893, 0.1680] | 0.2747 [0.1699, 0.3795] | -0.1461 [-0.2424, -0.0497] |
| retinaface | 0.1 | 0.0338 [0.0273, 0.0403] | 0.0604 [0.0440, 0.0767] | 0.0091 [0.0086, 0.0095] | 0.0513 [0.0349, 0.0677] | 0.0446 [0.0308, 0.0584] | 0.0445 [0.0309, 0.0580] | 0.0002 [-0.0003, 0.0006] |
| retinaface | 0.2 | 0.0512 [0.0423, 0.0602] | 0.1166 [0.0836, 0.1497] | 0.0175 [0.0161, 0.0189] | 0.0992 [0.0662, 0.1321] | 0.0706 [0.0516, 0.0896] | 0.0700 [0.0513, 0.0887] | 0.0006 [-0.0015, 0.0026] |
| retinaface | 0.3 | 0.1800 [0.0479, 0.3122] | 0.7824 [0.7038, 0.8611] | 0.0827 [0.0174, 0.1480] | 0.6997 [0.5663, 0.8332] | 0.1055 [0.0755, 0.1354] | 0.1383 [0.0750, 0.2016] | -0.0328 [-0.0749, 0.0092] |
| retinaface | 0.4 | 0.3145 [0.1356, 0.4933] | 1.0116 [0.8793, 1.1439] | 0.1861 [0.0670, 0.3053] | 0.8255 [0.5861, 1.0648] | 0.1843 [0.1145, 0.2541] | 0.1930 [0.1152, 0.2709] | -0.0088 [-0.0461, 0.0286] |
| retinaface | 0.5 | 0.5628 [0.3307, 0.7949] | 1.0639 [0.8635, 1.2643] | 0.4263 [0.2310, 0.6217] | 0.6376 [0.2563, 1.0188] | 0.2291 [0.1538, 0.3045] | 0.3394 [0.2319, 0.4469] | -0.1103 [-0.1961, -0.0244] |
| detr | 0.1 | 0.1918 [0.0385, 0.3451] | 0.2116 [0.1753, 0.2478] | 0.0341 [0.0093, 0.0589] | 0.1775 [0.1244, 0.2306] | 0.1521 [0.0660, 0.2382] | 0.1718 [0.0757, 0.2679] | -0.0197 [-0.0585, 0.0190] |
| detr | 0.2 | 0.2972 [0.1233, 0.4712] | 0.3416 [0.2683, 0.4149] | 0.0806 [0.0257, 0.1356] | 0.2610 [0.1514, 0.3705] | 0.2139 [0.1233, 0.3046] | 0.2572 [0.1492, 0.3651] | -0.0432 [-0.1041, 0.0176] |
| detr | 0.3 | 0.4951 [0.2874, 0.7027] | 0.4104 [0.3062, 0.5146] | 0.1789 [0.0782, 0.2796] | 0.2315 [0.0547, 0.4082] | 0.3276 [0.2226, 0.4326] | 0.3921 [0.2753, 0.5089] | -0.0645 [-0.1332, 0.0042] |
| detr | 0.4 | 0.6173 [0.3880, 0.8465] | 0.4395 [0.3171, 0.5620] | 0.3150 [0.1633, 0.4667] | 0.1245 [-0.1170, 0.3661] | 0.3937 [0.2909, 0.4966] | 0.4575 [0.3396, 0.5753] | -0.0637 [-0.1457, 0.0182] |
| detr | 0.5 | 0.9256 [0.6772, 1.1740] | 0.4154 [0.2717, 0.5590] | 0.6185 [0.3992, 0.8378] | -0.2031 [-0.5317, 0.1254] | 0.5353 [0.4339, 0.6366] | 0.5993 [0.4843, 0.7142] | -0.0640 [-0.1402, 0.0122] |
| cifar_vae | 0.1 | 0.0020 [0.0017, 0.0022] | 0.0688 [0.0456, 0.0919] | 0.0092 [0.0087, 0.0098] | 0.0595 [0.0364, 0.0827] | 0.0001 [0.0001, 0.0002] | 0.0001 [0.0001, 0.0002] | -0.0000 [-0.0000, 0.0000] |
| cifar_vae | 0.2 | 0.0074 [0.0064, 0.0083] | 0.1371 [0.0914, 0.1828] | 0.0191 [0.0180, 0.0202] | 0.1180 [0.0719, 0.1640] | 0.0007 [0.0006, 0.0009] | 0.0007 [0.0006, 0.0009] | 0.0000 [-0.0000, 0.0000] |
| cifar_vae | 0.3 | 0.0158 [0.0138, 0.0178] | 0.2030 [0.1344, 0.2716] | 0.0278 [0.0261, 0.0294] | 0.1752 [0.1068, 0.2437] | 0.0020 [0.0016, 0.0024] | 0.0021 [0.0016, 0.0025] | -0.0001 [-0.0002, 0.0001] |
| cifar_vae | 0.4 | 0.0268 [0.0235, 0.0301] | 0.2669 [0.1775, 0.3563] | 0.0383 [0.0358, 0.0408] | 0.2286 [0.1390, 0.3182] | 0.0039 [0.0033, 0.0046] | 0.0041 [0.0034, 0.0048] | -0.0001 [-0.0005, 0.0002] |
| cifar_vae | 0.5 | 0.0398 [0.0350, 0.0445] | 0.3192 [0.2109, 0.4276] | 0.0470 [0.0436, 0.0504] | 0.2722 [0.1648, 0.3797] | 0.0068 [0.0056, 0.0079] | 0.0071 [0.0058, 0.0085] | -0.0004 [-0.0008, 0.0001] |
| sd_vae | 0.1 | 0.0302 [0.0245, 0.0358] | 0.1033 [0.0738, 0.1328] | 0.0095 [0.0089, 0.0100] | 0.0939 [0.0644, 0.1233] | 0.0077 [0.0053, 0.0102] | 0.0068 [0.0052, 0.0083] | 0.0009 [-0.0003, 0.0022] |
| sd_vae | 0.2 | 0.0640 [0.0538, 0.0741] | 0.1849 [0.1283, 0.2415] | 0.0187 [0.0176, 0.0197] | 0.1662 [0.1093, 0.2231] | 0.0191 [0.0134, 0.0247] | 0.0177 [0.0137, 0.0216] | 0.0014 [-0.0012, 0.0039] |
| sd_vae | 0.3 | 0.0955 [0.0833, 0.1076] | 0.2431 [0.1636, 0.3225] | 0.0277 [0.0254, 0.0300] | 0.2154 [0.1356, 0.2951] | 0.0360 [0.0279, 0.0440] | 0.0343 [0.0270, 0.0417] | 0.0016 [-0.0002, 0.0035] |
| sd_vae | 0.4 | 0.1225 [0.1099, 0.1351] | 0.2837 [0.1879, 0.3795] | 0.0381 [0.0358, 0.0404] | 0.2456 [0.1501, 0.3411] | 0.0581 [0.0471, 0.0690] | 0.0552 [0.0459, 0.0646] | 0.0028 [-0.0000, 0.0057] |
| sd_vae | 0.5 | 0.1465 [0.1339, 0.1590] | 0.3102 [0.2014, 0.4190] | 0.0453 [0.0422, 0.0484] | 0.2649 [0.1566, 0.3732] | 0.0766 [0.0661, 0.0871] | 0.0747 [0.0645, 0.0848] | 0.0019 [-0.0010, 0.0048] |
| mae | 0.1 | 0.0229 [0.0196, 0.0262] | 0.0774 [0.0527, 0.1021] | 0.0088 [0.0081, 0.0095] | 0.0686 [0.0440, 0.0932] | 0.0052 [0.0044, 0.0061] | 0.0051 [0.0043, 0.0059] | 0.0001 [-0.0001, 0.0003] |
| mae | 0.2 | 0.0547 [0.0482, 0.0612] | 0.1468 [0.1009, 0.1928] | 0.0187 [0.0175, 0.0200] | 0.1281 [0.0825, 0.1736] | 0.0194 [0.0153, 0.0234] | 0.0190 [0.0152, 0.0229] | 0.0003 [-0.0006, 0.0013] |
| mae | 0.3 | 0.0853 [0.0768, 0.0938] | 0.2016 [0.1371, 0.2660] | 0.0278 [0.0258, 0.0298] | 0.1738 [0.1090, 0.2386] | 0.0409 [0.0326, 0.0492] | 0.0407 [0.0327, 0.0486] | 0.0002 [-0.0024, 0.0029] |
| mae | 0.4 | 0.1136 [0.1043, 0.1229] | 0.2421 [0.1626, 0.3215] | 0.0359 [0.0334, 0.0385] | 0.2061 [0.1269, 0.2853] | 0.0679 [0.0562, 0.0796] | 0.0677 [0.0568, 0.0785] | 0.0002 [-0.0029, 0.0033] |
| mae | 0.5 | 0.1381 [0.1285, 0.1477] | 0.2743 [0.1830, 0.3656] | 0.0451 [0.0424, 0.0477] | 0.2293 [0.1379, 0.3207] | 0.0941 [0.0819, 0.1063] | 0.0954 [0.0822, 0.1087] | -0.0013 [-0.0033, 0.0006] |
| clip_l14 | 0.1 | 0.0637 [0.0565, 0.0709] | 0.0783 [0.0530, 0.1035] | 0.0091 [0.0085, 0.0097] | 0.0691 [0.0438, 0.0944] | 0.0914 [0.0801, 0.1028] | 0.0922 [0.0804, 0.1039] | -0.0007 [-0.0021, 0.0007] |
| clip_l14 | 0.2 | 0.0967 [0.0862, 0.1072] | 0.1338 [0.0866, 0.1809] | 0.0186 [0.0174, 0.0197] | 0.1152 [0.0683, 0.1621] | 0.1919 [0.1719, 0.2120] | 0.1934 [0.1730, 0.2137] | -0.0014 [-0.0051, 0.0023] |
| clip_l14 | 0.3 | 0.1063 [0.0957, 0.1168] | 0.2053 [0.1356, 0.2750] | 0.0269 [0.0252, 0.0287] | 0.1784 [0.1088, 0.2479] | 0.2907 [0.2683, 0.3131] | 0.2889 [0.2674, 0.3104] | 0.0018 [-0.0015, 0.0051] |
| clip_l14 | 0.4 | 0.1152 [0.1047, 0.1258] | 0.2543 [0.1680, 0.3406] | 0.0374 [0.0351, 0.0396] | 0.2170 [0.1308, 0.3031] | 0.3518 [0.3285, 0.3751] | 0.3517 [0.3286, 0.3748] | 0.0001 [-0.0034, 0.0036] |
| clip_l14 | 0.5 | 0.1191 [0.1099, 0.1283] | 0.2529 [0.1560, 0.3499] | 0.0466 [0.0435, 0.0497] | 0.2064 [0.1096, 0.3031] | 0.4022 [0.3776, 0.4268] | 0.4022 [0.3775, 0.4268] | 0.0000 [-0.0022, 0.0023] |
| clip_rn50x16 | 0.1 | 0.0668 [0.0593, 0.0743] | 0.0826 [0.0521, 0.1130] | 0.0092 [0.0085, 0.0099] | 0.0734 [0.0429, 0.1038] | 0.0971 [0.0859, 0.1083] | 0.0959 [0.0851, 0.1066] | 0.0012 [-0.0000, 0.0025] |
| clip_rn50x16 | 0.2 | 0.0871 [0.0777, 0.0964] | 0.1558 [0.0996, 0.2120] | 0.0181 [0.0169, 0.0194] | 0.1377 [0.0818, 0.1936] | 0.1822 [0.1655, 0.1989] | 0.1798 [0.1628, 0.1968] | 0.0024 [-0.0003, 0.0050] |
| clip_rn50x16 | 0.3 | 0.0891 [0.0785, 0.0996] | 0.2182 [0.1384, 0.2979] | 0.0276 [0.0257, 0.0294] | 0.1906 [0.1113, 0.2699] | 0.2734 [0.2514, 0.2955] | 0.2723 [0.2507, 0.2940] | 0.0011 [-0.0021, 0.0043] |
| clip_rn50x16 | 0.4 | 0.0862 [0.0769, 0.0956] | 0.2649 [0.1684, 0.3614] | 0.0371 [0.0347, 0.0395] | 0.2278 [0.1315, 0.3240] | 0.3512 [0.3280, 0.3744] | 0.3515 [0.3281, 0.3749] | -0.0003 [-0.0030, 0.0025] |
| clip_rn50x16 | 0.5 | 0.0838 [0.0747, 0.0928] | 0.3266 [0.2148, 0.4384] | 0.0450 [0.0420, 0.0480] | 0.2816 [0.1699, 0.3933] | 0.4119 [0.3870, 0.4369] | 0.4139 [0.3888, 0.4390] | -0.0020 [-0.0053, 0.0014] |
| siglip | 0.1 | 0.0629 [0.0557, 0.0702] | 0.0822 [0.0569, 0.1076] | 0.0091 [0.0086, 0.0097] | 0.0731 [0.0478, 0.0984] | 0.0866 [0.0774, 0.0957] | 0.0871 [0.0778, 0.0963] | -0.0005 [-0.0021, 0.0012] |
| siglip | 0.2 | 0.0900 [0.0819, 0.0982] | 0.1300 [0.0817, 0.1783] | 0.0178 [0.0167, 0.0190] | 0.1122 [0.0637, 0.1606] | 0.2058 [0.1869, 0.2247] | 0.2052 [0.1865, 0.2239] | 0.0006 [-0.0017, 0.0028] |
| siglip | 0.3 | 0.0984 [0.0906, 0.1062] | 0.1586 [0.0973, 0.2199] | 0.0274 [0.0256, 0.0292] | 0.1312 [0.0695, 0.1929] | 0.3086 [0.2853, 0.3318] | 0.3076 [0.2838, 0.3313] | 0.0010 [-0.0012, 0.0032] |
| siglip | 0.4 | 0.1047 [0.0972, 0.1121] | 0.1712 [0.0991, 0.2433] | 0.0373 [0.0345, 0.0402] | 0.1339 [0.0614, 0.2063] | 0.3826 [0.3573, 0.4078] | 0.3824 [0.3572, 0.4076] | 0.0002 [-0.0022, 0.0026] |
| siglip | 0.5 | 0.1044 [0.0963, 0.1125] | 0.2375 [0.1536, 0.3213] | 0.0441 [0.0407, 0.0475] | 0.1934 [0.1092, 0.2775] | 0.4411 [0.4165, 0.4658] | 0.4415 [0.4176, 0.4653] | -0.0003 [-0.0036, 0.0030] |

###### mtcnn vs median smoothing（σ=0.1）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0543 [0.0351, 0.0735] | 0.1422 [0.1267, 0.1577] | -0.0879 [-0.0974, -0.0784] | 0.0335 [0.0290, 0.0380] | -0.0465 [-0.0510, -0.0420] | 50/50 | 0.0503 [0.0364, 0.0642] | 0.0473 [0.0343, 0.0603] | 0.0030 [0.0002, 0.0058] |
| 0.1000 | 0.0862 [0.0649, 0.1075] | 0.1751 [0.1589, 0.1913] | -0.0889 [-0.0986, -0.0793] | 0.0537 [0.0470, 0.0605] | -0.0463 [-0.0530, -0.0395] | 50/50 | 0.0532 [0.0399, 0.0664] | 0.0473 [0.0343, 0.0603] | 0.0059 [0.0033, 0.0085] |
| 0.1200 | 0.1240 [0.1011, 0.1470] | 0.2077 [0.1900, 0.2253] | -0.0837 [-0.0946, -0.0727] | 0.0788 [0.0682, 0.0893] | -0.0412 [-0.0518, -0.0307] | 50/50 | 0.0534 [0.0403, 0.0664] | 0.0473 [0.0343, 0.0603] | 0.0061 [0.0029, 0.0092] |
| 0.1400 | 0.1630 [0.1385, 0.1876] | 0.2340 [0.2160, 0.2520] | -0.0709 [-0.0835, -0.0584] | 0.1037 [0.0908, 0.1166] | -0.0363 [-0.0492, -0.0234] | 50/50 | 0.0540 [0.0413, 0.0668] | 0.0473 [0.0343, 0.0603] | 0.0067 [0.0029, 0.0105] |
| 0.1600 | 0.1944 [0.1701, 0.2187] | 0.2535 [0.2363, 0.2708] | -0.0592 [-0.0717, -0.0467] | 0.1235 [0.1092, 0.1378] | -0.0365 [-0.0508, -0.0222] | 50/50 | 0.0528 [0.0402, 0.0654] | 0.0473 [0.0343, 0.0603] | 0.0055 [0.0014, 0.0095] |

###### mtcnn vs alpha smoothing（σ=0.1）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0543 [0.0351, 0.0735] | 0.0002 [-0.0002, 0.0007] | 0.0541 [0.0349, 0.0732] | 0.0503 [0.0364, 0.0642] | 0.0494 [0.0369, 0.0619] | 0.0009 [-0.0023, 0.0041] |
| 0.1000 | 0.0862 [0.0649, 0.1075] | 0.0018 [-0.0008, 0.0043] | 0.0844 [0.0636, 0.1052] | 0.0532 [0.0399, 0.0664] | 0.0494 [0.0369, 0.0619] | 0.0038 [0.0010, 0.0067] |
| 0.1200 | 0.1240 [0.1011, 0.1470] | 0.0033 [-0.0006, 0.0072] | 0.1207 [0.0988, 0.1426] | 0.0534 [0.0403, 0.0664] | 0.0494 [0.0369, 0.0619] | 0.0040 [0.0010, 0.0070] |
| 0.1400 | 0.1630 [0.1385, 0.1876] | 0.0052 [0.0001, 0.0104] | 0.1578 [0.1347, 0.1809] | 0.0540 [0.0413, 0.0668] | 0.0494 [0.0369, 0.0619] | 0.0047 [0.0013, 0.0080] |
| 0.1600 | 0.1944 [0.1701, 0.2187] | 0.0071 [0.0006, 0.0136] | 0.1872 [0.1648, 0.2097] | 0.0528 [0.0402, 0.0654] | 0.0494 [0.0369, 0.0619] | 0.0034 [-0.0002, 0.0071] |

###### mtcnn：ContRS-M vs median smoothing

| r | ContRS-M R | median R | ΔR = ContRS-M − median | ContRS-M err | median err | Δerr = ContRS-M − median |
|---|---:|---|---|---|---|---|
| 0.0800 | 0.0547 [0.0352, 0.0741] | 0.1422 [0.1267, 0.1577] | -0.0875 [-0.0972, -0.0778] | 0.0502 [0.0362, 0.0641] | 0.0473 [0.0343, 0.0603] | 0.0029 [0.0000, 0.0057] |
| 0.1000 | 0.0863 [0.0649, 0.1077] | 0.1751 [0.1589, 0.1913] | -0.0889 [-0.0987, -0.0791] | 0.0531 [0.0398, 0.0664] | 0.0473 [0.0343, 0.0603] | 0.0058 [0.0032, 0.0085] |
| 0.1200 | 0.1236 [0.1008, 0.1464] | 0.2077 [0.1900, 0.2253] | -0.0841 [-0.0950, -0.0732] | 0.0533 [0.0402, 0.0663] | 0.0473 [0.0343, 0.0603] | 0.0060 [0.0030, 0.0090] |
| 0.1400 | 0.1624 [0.1382, 0.1867] | 0.2340 [0.2160, 0.2520] | -0.0715 [-0.0838, -0.0593] | 0.0542 [0.0415, 0.0669] | 0.0473 [0.0343, 0.0603] | 0.0069 [0.0031, 0.0107] |
| 0.1600 | 0.1955 [0.1708, 0.2202] | 0.2535 [0.2363, 0.2708] | -0.0581 [-0.0708, -0.0453] | 0.0531 [0.0406, 0.0657] | 0.0473 [0.0343, 0.0603] | 0.0058 [0.0018, 0.0098] |

###### clip_l14 vs median smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0796 [0.0425, 0.1166] | 0.4681 [0.4454, 0.4908] | -0.3885 [-0.4089, -0.3681] | 0.0029 [0.0026, 0.0033] | -0.0771 [-0.0774, -0.0767] | 50/50 | 0.2491 [0.2279, 0.2703] | 0.2405 [0.2194, 0.2615] | 0.0087 [0.0048, 0.0125] |
| 0.0880 | 0.1088 [0.0621, 0.1554] | 0.4877 [0.4645, 0.5109] | -0.3789 [-0.4067, -0.3511] | 0.0050 [0.0045, 0.0056] | -0.0830 [-0.0835, -0.0824] | 50/50 | 0.2472 [0.2263, 0.2682] | 0.2405 [0.2194, 0.2615] | 0.0068 [0.0040, 0.0095] |
| 0.1000 | 0.1555 [0.0982, 0.2128] | 0.5142 [0.4905, 0.5378] | -0.3587 [-0.3951, -0.3223] | 0.0096 [0.0086, 0.0106] | -0.0904 [-0.0914, -0.0894] | 50/50 | 0.2462 [0.2254, 0.2671] | 0.2405 [0.2194, 0.2615] | 0.0058 [0.0037, 0.0079] |
| 0.1080 | 0.1889 [0.1250, 0.2528] | 0.5302 [0.5064, 0.5540] | -0.3413 [-0.3836, -0.2991] | 0.0138 [0.0123, 0.0153] | -0.0942 [-0.0957, -0.0927] | 50/50 | 0.2462 [0.2256, 0.2668] | 0.2405 [0.2194, 0.2615] | 0.0058 [0.0036, 0.0079] |
| 0.1200 | 0.2425 [0.1731, 0.3119] | 0.5523 [0.5285, 0.5762] | -0.3099 [-0.3568, -0.2629] | 0.0224 [0.0199, 0.0248] | -0.0976 [-0.1001, -0.0952] | 50/50 | 0.2465 [0.2259, 0.2672] | 0.2405 [0.2194, 0.2615] | 0.0061 [0.0038, 0.0083] |

###### clip_l14 vs alpha smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0796 [0.0425, 0.1166] | 0.0000 [0.0000, 0.0000] | 0.0796 [0.0425, 0.1166] | 0.2491 [0.2279, 0.2703] | 0.2422 [0.2212, 0.2633] | 0.0069 [0.0031, 0.0107] |
| 0.0880 | 0.1088 [0.0621, 0.1554] | 0.0000 [0.0000, 0.0000] | 0.1088 [0.0621, 0.1554] | 0.2472 [0.2263, 0.2682] | 0.2422 [0.2212, 0.2633] | 0.0050 [0.0022, 0.0078] |
| 0.1000 | 0.1555 [0.0982, 0.2128] | 0.0000 [0.0000, 0.0000] | 0.1555 [0.0982, 0.2128] | 0.2462 [0.2254, 0.2671] | 0.2422 [0.2212, 0.2633] | 0.0040 [0.0019, 0.0061] |
| 0.1080 | 0.1889 [0.1250, 0.2528] | 0.0000 [0.0000, 0.0000] | 0.1889 [0.1250, 0.2528] | 0.2462 [0.2256, 0.2668] | 0.2422 [0.2212, 0.2633] | 0.0040 [0.0018, 0.0061] |
| 0.1200 | 0.2425 [0.1731, 0.3119] | 0.0000 [0.0000, 0.0000] | 0.2425 [0.1731, 0.3119] | 0.2465 [0.2259, 0.2672] | 0.2422 [0.2212, 0.2633] | 0.0043 [0.0020, 0.0066] |

###### clip_l14：ContRS-M vs median smoothing

| r | ContRS-M R | median R | ΔR = ContRS-M − median | ContRS-M err | median err | Δerr = ContRS-M − median |
|---|---:|---|---|---|---|---|
| 0.0800 | 0.1747 [0.1184, 0.2310] | 0.4681 [0.4454, 0.4908] | -0.2934 [-0.3286, -0.2582] | 0.2404 [0.2195, 0.2614] | 0.2405 [0.2194, 0.2615] | -0.0000 [-0.0002, 0.0001] |
| 0.0880 | 0.2187 [0.1586, 0.2789] | 0.4877 [0.4645, 0.5109] | -0.2689 [-0.3070, -0.2309] | 0.2404 [0.2194, 0.2613] | 0.2405 [0.2194, 0.2615] | -0.0001 [-0.0003, 0.0001] |
| 0.1000 | 0.2871 [0.2238, 0.3504] | 0.5142 [0.4905, 0.5378] | -0.2271 [-0.2676, -0.1865] | 0.2404 [0.2195, 0.2614] | 0.2405 [0.2194, 0.2615] | -0.0000 [-0.0002, 0.0001] |
| 0.1080 | 0.3325 [0.2692, 0.3959] | 0.5302 [0.5064, 0.5540] | -0.1977 [-0.2380, -0.1573] | 0.2404 [0.2194, 0.2614] | 0.2405 [0.2194, 0.2615] | -0.0001 [-0.0003, 0.0001] |
| 0.1200 | 0.3937 [0.3329, 0.4544] | 0.5523 [0.5285, 0.5762] | -0.1587 [-0.1963, -0.1210] | 0.2406 [0.2198, 0.2615] | 0.2405 [0.2194, 0.2615] | 0.0002 [-0.0004, 0.0007] |

###### cifar_vae_linf vs median smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.2852 | 0.0891 [0.0413, 0.1370] | 0.7290 [0.7270, 0.7309] | -0.6398 [-0.6871, -0.5925] | 0.0270 [0.0258, 0.0282] | -0.2582 [-0.2594, -0.2570] | 50/50 | 0.0904 [0.0838, 0.0970] | 0.0491 [0.0422, 0.0560] | 0.0413 [0.0356, 0.0470] |
| 0.3097 | 0.1480 [0.0926, 0.2033] | 0.7323 [0.7317, 0.7329] | -0.5844 [-0.6396, -0.5291] | 0.0422 [0.0403, 0.0442] | -0.2675 [-0.2694, -0.2655] | 50/50 | 0.0900 [0.0831, 0.0968] | 0.0491 [0.0422, 0.0560] | 0.0409 [0.0352, 0.0465] |
| 0.3364 | 0.2255 [0.1673, 0.2837] | 0.7328 [0.7328, 0.7328] | -0.5072 [-0.5654, -0.4490] | 0.0625 [0.0597, 0.0653] | -0.2738 [-0.2766, -0.2710] | 50/50 | 0.0920 [0.0854, 0.0986] | 0.0491 [0.0422, 0.0560] | 0.0429 [0.0370, 0.0488] |
| 0.3653 | 0.3275 [0.2720, 0.3829] | 0.7328 [0.7328, 0.7328] | -0.4053 [-0.4607, -0.3499] | 0.0898 [0.0858, 0.0938] | -0.2755 [-0.2795, -0.2715] | 50/50 | 0.0951 [0.0881, 0.1021] | 0.0491 [0.0422, 0.0560] | 0.0461 [0.0404, 0.0517] |
| 0.3967 | 0.4314 [0.3818, 0.4809] | 0.7328 [0.7328, 0.7328] | -0.3014 [-0.3509, -0.2519] | 0.1189 [0.1137, 0.1242] | -0.2778 [-0.2831, -0.2725] | 50/50 | 0.0955 [0.0892, 0.1018] | 0.0491 [0.0422, 0.0560] | 0.0464 [0.0410, 0.0519] |

###### cifar_vae_linf vs alpha smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---|
| 0.2852 | 0.0891 [0.0413, 0.1370] | 0.0549 [0.0208, 0.0891] | 0.0342 [0.0181, 0.0503] | 0.0904 [0.0838, 0.0970] | 0.0668 [0.0618, 0.0717] | 0.0236 [0.0193, 0.0279] |
| 0.3097 | 0.1480 [0.0926, 0.2033] | 0.0974 [0.0573, 0.1374] | 0.0506 [0.0325, 0.0687] | 0.0900 [0.0831, 0.0968] | 0.0668 [0.0618, 0.0717] | 0.0232 [0.0188, 0.0275] |
| 0.3364 | 0.2255 [0.1673, 0.2837] | 0.1604 [0.1160, 0.2048] | 0.0651 [0.0463, 0.0840] | 0.0920 [0.0854, 0.0986] | 0.0668 [0.0618, 0.0717] | 0.0252 [0.0207, 0.0298] |
| 0.3653 | 0.3275 [0.2720, 0.3829] | 0.2461 [0.2016, 0.2905] | 0.0814 [0.0628, 0.0999] | 0.0951 [0.0881, 0.1021] | 0.0668 [0.0618, 0.0717] | 0.0284 [0.0239, 0.0328] |
| 0.3967 | 0.4314 [0.3818, 0.4809] | 0.3378 [0.2960, 0.3796] | 0.0936 [0.0764, 0.1107] | 0.0955 [0.0892, 0.1018] | 0.0668 [0.0618, 0.0717] | 0.0287 [0.0245, 0.0329] |

###### cifar_vae_linf：ContRS-M vs median smoothing

| r | ContRS-M R | median R | ΔR = ContRS-M − median | ContRS-M err | median err | Δerr = ContRS-M − median |
|---|---:|---|---|---|---|---|
| 0.2852 | 0.1779 [0.1277, 0.2281] | 0.7290 [0.7270, 0.7309] | -0.5511 [-0.6003, -0.5019] | 0.0507 [0.0439, 0.0576] | 0.0491 [0.0422, 0.0560] | 0.0017 [-0.0000, 0.0033] |
| 0.3097 | 0.2718 [0.2217, 0.3220] | 0.7323 [0.7317, 0.7329] | -0.4605 [-0.5105, -0.4105] | 0.0505 [0.0436, 0.0574] | 0.0491 [0.0422, 0.0560] | 0.0014 [-0.0002, 0.0030] |
| 0.3364 | 0.3682 [0.3229, 0.4135] | 0.7328 [0.7328, 0.7328] | -0.3645 [-0.4098, -0.3193] | 0.0516 [0.0446, 0.0587] | 0.0491 [0.0422, 0.0560] | 0.0025 [-0.0002, 0.0053] |
| 0.3653 | 0.4594 [0.4191, 0.4997] | 0.7328 [0.7328, 0.7328] | -0.2734 [-0.3137, -0.2331] | 0.0512 [0.0443, 0.0582] | 0.0491 [0.0422, 0.0560] | 0.0022 [-0.0002, 0.0045] |
| 0.3967 | 0.5461 [0.5095, 0.5826] | 0.7328 [0.7328, 0.7328] | -0.1867 [-0.2232, -0.1502] | 0.0525 [0.0452, 0.0599] | 0.0491 [0.0422, 0.0560] | 0.0034 [0.0001, 0.0068] |

##### 配对差值 CI 概览（未做多重比较校正）

- Safe region ΔR：43/45 行 CI 全为正；0/45 行 CI 全为负；其余跨 0。
- Center smoothing ε_out−r：50/50 行 CI 全为正。
- ContRS vs median 的同容差 ΔR：0/15 行 CI 全为正；15/15 行 CI 全为负；其余跨 0。
- ContRS vs alpha 的同容差 ΔR：15/15 行 CI 全为正；0/15 行 CI 全为负；其余跨 0。


## 7.3 两个例外设置独立新种子复核的 95% CI

以下来源表已完整内嵌：[ext300/recheck/summary/confidence_intervals.md](ext300/recheck/summary/confidence_intervals.md)。

#### 95% confidence intervals

图像为独立样本。均值区间为双侧 95% Student-t CI：mean ± t(0.975,n−1)·sd/√n。配对差值先在每张图上相减，再计算 CI。ΔR = ContRS − baseline（越大越有利于 ContRS）；Δerr = ContRS − baseline（越小越有利于 ContRS）。区间不做多重比较校正。

Center smoothing 的 ε_out 仅在未弃权图像上统计，弃权率另用 Wilson 95% CI；safe region 的 ContRS 数值全部取平均 r_SR 处的精确补跑结果。


##### Center smoothing：认证输出界

所有列均为 mean [95% CI]。r 是预设容差，本身没有抽样 CI；ε_out−r 和 ε_out/r 大于 0/1 表示 ContRS 更紧。

| model | r | ContRS R = ε_in | CS ε_out | CS ε_out − r | CS ε_out / r | CS valid n | CS abstain [Wilson 95% CI] |
|---|---:|---|---|---|---|---:|---|
| clip_l14_300 | 0.08 | 0.0973 [0.0805, 0.1141] | 0.2649 [0.2536, 0.2763] | 0.1849 [0.1736, 0.1963] | 3.311 [3.170, 3.453] | 296/300 | 1.33% [0.52%, 3.38%] |
| clip_l14_300 | 0.088 | 0.1354 [0.1155, 0.1553] | 0.2780 [0.2659, 0.2900] | 0.1900 [0.1779, 0.2020] | 3.159 [3.022, 3.295] | 296/300 | 1.33% [0.52%, 3.38%] |
| clip_l14_300 | 0.1 | 0.1987 [0.1754, 0.2219] | 0.3027 [0.2894, 0.3161] | 0.2027 [0.1894, 0.2161] | 3.027 [2.894, 3.161] | 296/300 | 1.33% [0.52%, 3.38%] |
| clip_l14_300 | 0.108 | 0.2412 [0.2162, 0.2661] | 0.3222 [0.3075, 0.3368] | 0.2142 [0.1995, 0.2288] | 2.983 [2.847, 3.118] | 296/300 | 1.33% [0.52%, 3.38%] |
| clip_l14_300 | 0.12 | 0.3028 [0.2765, 0.3291] | 0.3551 [0.3383, 0.3720] | 0.2351 [0.2183, 0.2520] | 2.959 [2.819, 3.100] | 296/300 | 1.33% [0.52%, 3.38%] |

##### Center smoothing：平滑误差与配对差值

| model | r | ContRS err（全部图） | CS err（未弃权图） | Δerr = ContRS − CS（同一批未弃权图） | paired n |
|---|---:|---|---|---|---:|
| clip_l14_300 | 0.08 | 0.2342 [0.2268, 0.2416] | 0.2352 [0.2277, 0.2427] | -0.0013 [-0.0023, -0.0003] | 296 |
| clip_l14_300 | 0.088 | 0.2344 [0.2270, 0.2418] | 0.2352 [0.2277, 0.2427] | -0.0010 [-0.0020, 0.0001] | 296 |
| clip_l14_300 | 0.1 | 0.2351 [0.2276, 0.2425] | 0.2352 [0.2277, 0.2427] | -0.0003 [-0.0014, 0.0007] | 296 |
| clip_l14_300 | 0.108 | 0.2352 [0.2278, 0.2426] | 0.2352 [0.2277, 0.2427] | -0.0002 [-0.0011, 0.0007] | 296 |
| clip_l14_300 | 0.12 | 0.2359 [0.2285, 0.2433] | 0.2352 [0.2277, 0.2427] | 0.0005 [-0.0004, 0.0014] | 296 |

##### Safe region：平均 r_SR 口径，R 与平滑误差

| model | σ | r = 2κ·mean r_SR [95% CI] | ContRS R | SR R | ΔR = ContRS − SR | ContRS err | SR err | Δerr = ContRS − SR |
|---|---:|---|---|---|---|---|---|---|
| mtcnn300 | 0.5 | 1.1800 [1.0782, 1.2819] | 0.3202 [0.2693, 0.3711] | 0.9098 [0.8195, 1.0001] | -0.5896 [-0.7233, -0.4559] | 0.1900 [0.1723, 0.2077] | 0.5415 [0.4926, 0.5904] | -0.3515 [-0.3998, -0.3033] |

###### clip_l14_300 vs median smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0973 [0.0805, 0.1141] | 0.4857 [0.4767, 0.4948] | -0.3884 [-0.3987, -0.3781] | 0.0038 [0.0036, 0.0040] | -0.0762 [-0.0764, -0.0760] | 300/300 | 0.2342 [0.2268, 0.2416] | 0.2301 [0.2229, 0.2374] | 0.0041 [0.0029, 0.0052] |
| 0.0880 | 0.1354 [0.1155, 0.1553] | 0.5055 [0.4963, 0.5147] | -0.3701 [-0.3825, -0.3577] | 0.0068 [0.0065, 0.0071] | -0.0812 [-0.0815, -0.0809] | 300/300 | 0.2344 [0.2270, 0.2418] | 0.2301 [0.2229, 0.2374] | 0.0043 [0.0032, 0.0054] |
| 0.1000 | 0.1987 [0.1754, 0.2219] | 0.5323 [0.5229, 0.5417] | -0.3336 [-0.3485, -0.3187] | 0.0139 [0.0133, 0.0146] | -0.0861 [-0.0867, -0.0854] | 300/300 | 0.2351 [0.2276, 0.2425] | 0.2301 [0.2229, 0.2374] | 0.0050 [0.0038, 0.0061] |
| 0.1080 | 0.2412 [0.2162, 0.2661] | 0.5484 [0.5390, 0.5578] | -0.3072 [-0.3235, -0.2909] | 0.0203 [0.0194, 0.0212] | -0.0877 [-0.0886, -0.0868] | 300/300 | 0.2352 [0.2278, 0.2426] | 0.2301 [0.2229, 0.2374] | 0.0051 [0.0040, 0.0061] |
| 0.1200 | 0.3028 [0.2765, 0.3291] | 0.5703 [0.5609, 0.5796] | -0.2675 [-0.2849, -0.2501] | 0.0320 [0.0306, 0.0335] | -0.0880 [-0.0894, -0.0865] | 300/300 | 0.2359 [0.2285, 0.2433] | 0.2301 [0.2229, 0.2374] | 0.0058 [0.0048, 0.0067] |

###### clip_l14_300 vs mean smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | baseline ε_out | ε_out − r | ε_out valid n | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---:|---|---|---|
| 0.0800 | 0.0973 [0.0805, 0.1141] | 0.1073 [0.1072, 0.1074] | -0.0100 [-0.0267, 0.0067] | 0.0669 [0.0667, 0.0670] | -0.0131 [-0.0133, -0.0130] | 300/300 | 0.2342 [0.2268, 0.2416] | 0.2302 [0.2230, 0.2375] | 0.0040 [0.0028, 0.0051] |
| 0.0880 | 0.1354 [0.1155, 0.1553] | 0.1129 [0.1128, 0.1130] | 0.0225 [0.0027, 0.0423] | 0.1245 [0.1243, 0.1247] | 0.0365 [0.0363, 0.0367] | 300/300 | 0.2344 [0.2270, 0.2418] | 0.2302 [0.2230, 0.2375] | 0.0042 [0.0031, 0.0054] |
| 0.1000 | 0.1987 [0.1754, 0.2219] | 0.1208 [0.1207, 0.1209] | 0.0778 [0.0546, 0.1010] | 0.2750 [0.2745, 0.2755] | 0.1750 [0.1745, 0.1755] | 300/300 | 0.2351 [0.2276, 0.2425] | 0.2302 [0.2230, 0.2375] | 0.0049 [0.0037, 0.0060] |
| 0.1080 | 0.2412 [0.2162, 0.2661] | 0.1258 [0.1257, 0.1259] | 0.1154 [0.0905, 0.1402] | 0.4377 [0.4367, 0.4386] | 0.3297 [0.3287, 0.3306] | 300/300 | 0.2352 [0.2278, 0.2426] | 0.2302 [0.2230, 0.2375] | 0.0050 [0.0039, 0.0060] |
| 0.1200 | 0.3028 [0.2765, 0.3291] | 0.1329 [0.1328, 0.1330] | 0.1699 [0.1437, 0.1961] | 2.0000 [2.0000, 2.0000] | 1.8800 [1.8800, 1.8800] | 300/300 | 0.2359 [0.2285, 0.2433] | 0.2302 [0.2230, 0.2375] | 0.0057 [0.0047, 0.0067] |

###### clip_l14_300 vs alpha smoothing（σ=0.25）

| r | ContRS R | baseline R | ΔR = ContRS − baseline | ContRS err | baseline err | Δerr = ContRS − baseline |
|---|---|---|---|---|---|---|
| 0.0800 | 0.0973 [0.0805, 0.1141] | 0.0000 [0.0000, 0.0000] | 0.0973 [0.0805, 0.1141] | 0.2342 [0.2268, 0.2416] | 0.2318 [0.2246, 0.2391] | 0.0024 [0.0012, 0.0035] |
| 0.0880 | 0.1354 [0.1155, 0.1553] | 0.0000 [0.0000, 0.0000] | 0.1354 [0.1155, 0.1553] | 0.2344 [0.2270, 0.2418] | 0.2318 [0.2246, 0.2391] | 0.0026 [0.0015, 0.0037] |
| 0.1000 | 0.1987 [0.1754, 0.2219] | 0.0000 [0.0000, 0.0000] | 0.1987 [0.1754, 0.2219] | 0.2351 [0.2276, 0.2425] | 0.2318 [0.2246, 0.2391] | 0.0033 [0.0021, 0.0044] |
| 0.1080 | 0.2412 [0.2162, 0.2661] | 0.0000 [0.0000, 0.0000] | 0.2412 [0.2162, 0.2661] | 0.2352 [0.2278, 0.2426] | 0.2318 [0.2246, 0.2391] | 0.0034 [0.0023, 0.0044] |
| 0.1200 | 0.3028 [0.2765, 0.3291] | 0.0000 [0.0000, 0.0000] | 0.3028 [0.2765, 0.3291] | 0.2359 [0.2285, 0.2433] | 0.2318 [0.2246, 0.2391] | 0.0041 [0.0031, 0.0051] |

###### clip_l14_300：ContRS-M vs median smoothing

| r | ContRS-M R | median R | ΔR = ContRS-M − median | ContRS-M err | median err | Δerr = ContRS-M − median |
|---|---:|---|---|---|---|---|
| 0.0800 | 0.2205 [0.1986, 0.2425] | 0.4857 [0.4767, 0.4948] | -0.2652 [-0.2787, -0.2517] | 0.2301 [0.2229, 0.2374] | 0.2301 [0.2229, 0.2374] | 0.0000 [-0.0000, 0.0001] |
| 0.0880 | 0.2705 [0.2474, 0.2937] | 0.5055 [0.4963, 0.5147] | -0.2350 [-0.2493, -0.2206] | 0.2302 [0.2229, 0.2374] | 0.2301 [0.2229, 0.2374] | 0.0000 [-0.0000, 0.0001] |
| 0.1000 | 0.3415 [0.3177, 0.3654] | 0.5323 [0.5229, 0.5417] | -0.1908 [-0.2056, -0.1759] | 0.2302 [0.2229, 0.2374] | 0.2301 [0.2229, 0.2374] | 0.0000 [-0.0000, 0.0001] |
| 0.1080 | 0.3847 [0.3612, 0.4081] | 0.5484 [0.5390, 0.5578] | -0.1637 [-0.1781, -0.1493] | 0.2303 [0.2231, 0.2375] | 0.2301 [0.2229, 0.2374] | 0.0002 [0.0000, 0.0003] |
| 0.1200 | 0.4422 [0.4195, 0.4649] | 0.5703 [0.5609, 0.5796] | -0.1281 [-0.1418, -0.1144] | 0.2304 [0.2232, 0.2376] | 0.2301 [0.2229, 0.2374] | 0.0003 [0.0000, 0.0005] |

##### 配对差值 CI 概览（未做多重比较校正）

- Safe region ΔR：0/1 行 CI 全为正；1/1 行 CI 全为负；其余跨 0。
- Center smoothing ε_out−r：5/5 行 CI 全为正。
- ContRS vs median 的同容差 ΔR：0/5 行 CI 全为正；5/5 行 CI 全为负；其余跨 0。
- ContRS vs mean 的同容差 ΔR：4/5 行 CI 全为正；0/5 行 CI 全为负；其余跨 0。
- ContRS vs alpha 的同容差 ΔR：5/5 行 CI 全为正；0/5 行 CI 全为负；其余跨 0。


---

# 第八部分：复现文件索引与最终使用边界

## 8.1 主要实现和配置

| 内容 | 文件 |
|---|---|
| 模型、数据、batch、预处理注册表 | [rebuttal/scripts/registry.py](rebuttal/scripts/registry.py) |
| ContRS/safe/center 共用主驱动 | [rebuttal/scripts/run_arch.py](rebuttal/scripts/run_arch.py) |
| 任务距离实现 | [rebuttal/scripts/discrepancy.py](rebuttal/scripts/discrepancy.py) |
| 多架构出表与 center 系数 | [rebuttal/scripts/rebuttal_tables.py](rebuttal/scripts/rebuttal_tables.py) |
| 新增 baseline 驱动 | [rebuttal/scripts/run_baselines.py](rebuttal/scripts/run_baselines.py) |
| 中位数/α/均值证书实现 | [rebuttal/scripts/baselines.py](rebuttal/scripts/baselines.py) |
| CI 计算 | [rebuttal/scripts/confidence_intervals.py](rebuttal/scripts/confidence_intervals.py) |
| 300 图编排与精确补跑 | [ext300/orchestrate.py](ext300/orchestrate.py) |
| p 扫描配置 | [p_sweep/run_config.json](p_sweep/run_config.json) |
| MTCNN p 扫描 KDD 驱动 | [p_sweep/run_mtcnn_kdd_parallel.py](p_sweep/run_mtcnn_kdd_parallel.py) |
| MTCNN p 扫描编排 | [p_sweep/orchestrate_mtcnn.py](p_sweep/orchestrate_mtcnn.py) |
| MTCNN p 扫描出表 | [p_sweep/make_mtcnn_tables.py](p_sweep/make_mtcnn_tables.py) |
| VAE/CLIP p 扫描出表 | [p_sweep/make_tables.py](p_sweep/make_tables.py) |
| 独立新种子复核编排 | [ext300/recheck/orchestrate.py](ext300/recheck/orchestrate.py) |

原始逐图 NPZ 分别位于 [ext300/results/](ext300/results/)、[ext300/results_exact_b/](ext300/results_exact_b/)、[ext300/results_baselines/](ext300/results_baselines/)、[rebuttal/results/](rebuttal/results/)、[rebuttal/results_exact_b/](rebuttal/results_exact_b/)、[p_sweep/results_kdd/](p_sweep/results_kdd/) 和 [p_sweep/results_contrs/](p_sweep/results_contrs/)。

## 8.2 rebuttal 中必须保留的限制

1. **样本量不可混写：**300 图主实验、50 图多架构、50 图 p 扫描、49/50 图早期 baseline 是不同实验包。
2. **不是全组合：**没有完成 300 图 p 扫描，也没有在所有新增架构上完成 median/mean/alpha 的 300 图比较。
3. **safe 的 p 不是认证 p_L：**p=0.55 与 p=0.6/0.7/0.8 是构造球的目标覆盖率。
4. **组均值匹配不是逐图匹配：**论文式 safe 主表使用每组统一平均半径；两种口径结果不能互相替代。
5. **非等查询预算：**split ContRS 为 20000 次加噪前向/图，safe region 为 15000；含 clean 则各加 1。
6. **p 扫描不是共享噪声：**双方同图但噪声独立；三个 p 间又共享 safe-region 样本池，行间不独立。
7. **VAE/CLIP 外部精度未闭合：**六位均值可能来自四位逐图 TSV，潜在 ±5×10⁻⁵ 不确定度；外部逐图 ID 和像素哈希未核实。
8. **MTCNN selected50 未获外部确认：**与 legacy50 交集为 0，不得称为历史同 50 图复现，也不得误用 mtcnn300 清单。
9. **Jaccard 饱和：**safe_r=1 的闭球覆盖整个有界输出空间；r>1 的 prediction-change 输出界平凡。保留所有行，不事后过滤饱和样本冒充主结果。
10. **CI 是描述性逐行区间：**未做多重比较校正，且固定组均值半径未传播其估计不确定性。
11. **center 有弃权：**输出界只在未弃权图上平均，应同时报告有效 n 和弃权率。
12. **LPIPS/cosine 的 center 有效系数是 10：**coef=3 只作投稿旧口径对照。
13. **新增 VAE baseline 是像素 ℓ∞：**不能写成这些逐坐标方法已直接在 LPIPS 上认证。
14. **均值平滑不是独立新论文：**检测/重建关联 Salman 2019，检索关联 Pautov 2022；ContRS-M 也不是外部论文。
15. **如实报告中位数结果：**median 在当前 15 个同容差点均优于 ContRS/ContRS-M 的 R；不能用其他 baseline 的正向结果掩盖这一点。
16. **有限样本语义：**ContRS 直接保证相对部署样本中心的变化，不应无条件改写成相对总体平滑函数的相同保证。

## 8.3 推荐的 rebuttal 取用顺序

- 主文样本扩展：第一部分 + 7.1；
- 主文新增 baseline：第二部分 + 7.1，并明确 VAE 为像素 ℓ∞；
- 架构泛化：第三部分或 3.3 紧凑表 + 7.2；
- safe-region 超参数稳健性：第五部分，同时携带精度、样本出处和 Jaccard 饱和说明；
- 对两个例外结果的回应：第六部分 + 7.3；
- 第四部分只作历史记录，不建议与 300 图主结果并排而不标样本量。

---

**文档结束。所有主要结果表均已内嵌；表内数值未在合并过程中重新计算。**
