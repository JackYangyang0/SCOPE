# SCOPE 论文工作与导师汇报大纲

## 1. 汇报定位

方法名称：SCOPE — Strategy-Guided Program Optimization。

建议题目：面向 FP32 GEMM 的策略图引导与验证闭环代码优化。

一句话概括：通过结构化程序状态、可检查的优化策略及局部补丁，将 LLM 的代码优化从整段生成转化为分阶段构造与可回退的性能搜索。

状态说明：当前已实现方法原型，最近完成了验证时机与策略实现逻辑调整；新版本仍需端到端实验确认。当前 results/check 下没有最新 top_3_terminal_results.json，不能将历史性能数值作为本版本最终成果。

## 2. PPT 大纲与逐页内容

### 第 1 页：研究目标与范围

- 任务：依据 GEMM 需求、硬件信息和优化策略生成正确且高性能的 kernel。
- 主要对象：单设备 SIMT FP32 GEMM，GPU 为论文主线。
- 数学目标：C = alpha × A × B + beta × C。
- 性能目标：在统一精度与测试条件下接近 cuBLAS，并寻找特定形状上的优势。
- CPU C 代码与 OpenBLAS 对比作为扩展，不与 GPU 结论混在一起。

讲述重点：超过 cuBLAS 是实验目标而非已有结论；先回答方法是否能稳定生成正确代码，再回答性能是否具有竞争力。

### 第 2 页：研究问题

直接让 LLM 生成完整 GEMM 存在以下困难：

1. 输出缺少显式程序状态，容易出现参数、索引、共享内存与流水线不一致。
2. 优化之间存在依赖与冲突，例如向量加载与协作加载映射必须匹配。
3. 整文件重写容易覆盖已经有效的优化，并增加输入输出成本。
4. 原始编译与运行报错不能直接定位到策略或 IR 字段。
5. 中间构造尚未完成时运行验证，可能错误淘汰本来可用的策略路径。

研究问题：如何让 LLM 在明确的状态与策略约束下完成组合优化，并通过验证反馈可靠地纠错？

### 第 3 页：方法总体架构

输入：GEMM specification、target hardware、初始代码骨架、策略索引/库及依赖图。

组成：OptIR + Strategy Graph + Patch Protocol + Verification Oracle + Defect Diagnosis + Search。

流程：需求与硬件提取 → Phase 1 基础构造 → 完整验证与修复 → 正确候选排序 → Phase 2 batch unlock → 统一结果排序。

图示建议：主流程横向展开，诊断与修复画成返回代码更新节点的反馈箭头。

### 第 4 页：OptIR 的设计

OptIR 描述当前问题、硬件与 kernel 的优化状态，而不是替代真实代码。

- problem：M/N/K、数据类型、布局及现有模板定义的语义参数。
- hardware：warp size、SM 数量、线程上限、共享内存等与 GEMM 有关的能力。
- tiling/mapping：BM/BN/BK、WM/WN、TM/TN、warp iteration 与线程覆盖关系。
- memory/vectorization/pipeline：共享内存布局、向量宽度、对齐、尾部处理与流水线状态。
- strategy/history：已应用策略、阶段进度、失败次数与路径。
- verification/performance：编译、正确性、运行安全、延迟与 GFLOPS。

问题提取仅填充 problem 中已定义的字段，未提取项采用模板默认值；初始 IR 不被覆盖，新版本单独保存。

技术注意：alignment_guard=true 等声明不能证明实际代码有 guard。代码语义和运行验证必须在完整实现形成后检查声明与实现是否一致。

### 第 5 页：策略库与依赖图

策略索引用于轻量候选检索，策略库提供具体优化内容与约束，依赖图描述可用顺序和组合关系。

候选过滤检查：当前阶段/subphase、requires、conflicts、失败次数/fallback、maturity 与 phase、alias/canonical 去重及资源条件。

约束分离：

- preconditions：应用策略前的 OptIR 谓词。
- postconditions.patch_ir_verification：策略所负责的 IR 更新谓词。
- postconditions.code_verification：完整代码的结构、覆盖与动态性质。

本轮审计确认 GPU index/library 各有 136 个标签且双向一致，但标签一致不等于实现完整。重复寄存器策略已设为别名，无差异化实现的三个 lane-layout 候选已暂停；有效候选数应按实验时冻结版本重新统计。

### 第 6 页：Phase 1 分阶段构造

组织单位：stage → subphase → micro-strategy。

stage 包括 Tiling、Mapping、Layout、Reordering、Vectorization、Pipeline 等；具体顺序与所属关系以当前调用图为准，不按前缀任意推断。

每个策略：pre 检查 → 生成/应用局部 Patch → post 谓词检查，并保留轻量源语法和补丁结构检查。

阶段末只检查阶段负责的 IR 条件。整个 Phase 1 完成后才检查全局硬约束、语义、编译和运行，避免用未完成 kernel 的运行错误否定参数路径。

硬约束包括线程数与共享内存限制、tile 参数完整性、全局访存边界、向量对齐与尾部处理。

没有可用后继时应进入终止候选的完整验证；不完整实现需要被诊断为完整性缺陷，而不是静默丢弃。

### 第 7 页：层次化 Tiling 选择与确定性执行

LLM 选 Top-3 BlockTile → 每条路径过滤 WarpTile → LLM 选 Top-3 WarpTile → 过滤 ThreadTile → LLM 选 Top-3 ThreadTile。

关键过滤关系：

```text
BM % WM == 0
BN % WN == 0
warps_per_block = (BM / WM) × (BN / WN)
threads_per_block = warps_per_block × warp_size
WMITER % TM == 0
WNITER % TN == 0
(WMITER / TM) × (WNITER / TN) == warp_size
```

程序根据可行覆盖推导 WMITER/WNITER，线程数必须符合硬件上限。BlockTile 仅在 BM/BN/BK 三者同时相同时视为重复。

确定性策略：LLM 只选择策略，程序生成 IR 更新和 anchor replacement，不让 LLM 生成代码。Tiling 更新主要作用于 cuda_gemm 的 LAUNCH_CONFIG。

非确定性策略：LLM 接收当前选中策略完整内容、IR 摘要及相关 anchor 区域，返回局部替换。

3×3×3 是候选充足且所有组合合法时的上界，不是保证生成 27 条有效 Tiling 路径。当前保留所选兄弟分支，后续组合仍可能快速增长，应记录总预算。

### 第 8 页：局部 Patch 协议

代码由固定测试 harness 与可修改 kernel 区域组成。main.cpp 不作为优化目标，不发送到 GPU 修复提示词，也不纳入 kernel AST 分析。

相关区域：LAUNCH_CONFIG、SHARED_DECL、INDEX_MAPPING、REGISTER_DECL、GLOBAL_TO_SHARED_LOAD、NEXT_TILE_LOAD、MAIN_LOOP、COMPUTE_INNER、STORE 等。

LLM 输出区域名和 replacement_lines；程序定位 anchor 并替换，不要求每次返回完整 cuda_kernel.cuh。

Patch 记录策略 ID、修改区域、IR 字段更新、替换内容、预期效果和风险。采用替换而非追加，重复 anchor 或不合法响应必须报错，不得叠加变量定义。

优势假设：减少整文件重写引起的优化破坏与 token 成本。是否有效需要消融证明。

### 第 9 页：完整验证与结构化修复

Phase 1 终止候选：硬约束检查 → kernel AST/静态语义检查 → nvcc 编译 → correctness/runtime test → 性能采样 → 诊断与修复。

oracle 返回 compile_status、correctness_status、cuda_error、latency_ms、gflops，成功和失败都写回 IR。

诊断例子：misaligned address → Vectorization.AlignmentViolation → 相关向量宽度/对齐字段及加载代码区域 → 添加实际对齐分支或调整向量加载映射。

修复采用 LLM region edits，默认保留已有 tile、layout、vectorization 和 pipeline。错误来源涉及多个耦合区域时，必须允许共同修复，例如初始与后续 tile 加载都要修改。

目前末端修复允许多个 anchor，尚不是严格的最小修改范围；不应把局部修复描述为形式化语义保证。

### 第 10 页：Phase 2 Size-aware Batch Unlock

入口：Phase 1 完整实现中通过编译、正确性和运行安全的 Top-3 候选。

matrix_profile 同时考虑 ops、CTA 数、CTA/SM、K tile 数、共享内存及寄存器估计，不仅依据 M/N/K。

默认规则：CTA 数低于 SM 数视为低并行度；large_mn 要求 CTA 数至少为 SM 数两倍且 ops ≥ 2^33；large_k 为 K tiles ≥ 128 或 K ≥ 4096。阈值是当前启发式，需验证而非理论最优。

LLM 输入压缩 index、IR/代码摘要、matrix profile、verifier/profile 与失败历史，规划耦合 batch。程序再过滤非法 ID、重复、已应用、requires、conflict、资源及 large-only 策略。

每个 batch：选择 0~N 策略 → 局部 Patch → 应用 → 编译运行 → 诊断修复/重试 → fallback → accept 或 rollback。

默认修复上限 3 次。checker 失败作为诊断输入，不直接阻止编译运行；但 oracle 通过不意味着所有优化声明都已真实实现。

最终统一比较 Phase 1 与 Phase 2 候选，不能只输出 stable 或 unlock 的一方。

### 第 11 页：当前实现与典型问题

已实现：IR 提取、策略图过滤、分阶段控制、层次化 Tiling、确定性代码应用、局部 LLM 补丁、编译运行 verifier、缺陷诊断和修复、batch unlock 与链路记录。

已修正：重复变量/anchor、同名不同实现错配、通用 STORE 冒充 Beta 快路径、部分 mapping 策略覆盖 compute、NoAsyncCopy 错误重置流水线状态、中间阶段运行淘汰。

典型案例 chain.1-1-1-1：Tiling 的 IR 条件通过，运行 misaligned address；已有 float4 加载但映射产生非 4 倍数坐标；第三次修复只修改 COMPUTE_INNER，没有触及出错加载。该例说明验证时机和诊断区域定位都影响路径保留。

局限：静态检查并非完整 CUDA 语义解析；资源估计未必等于实际占用；修复不能保证成功；没有实测性能前不能称候选为更优解。

### 第 12 页：实验设计——环境与工作负载

建议主环境：RTX 4060 Ti、A800、RTX 5090；只报告实际可访问并完成实验的设备。CPU 可独立使用 Intel 平台/OpenBLAS。

记录 GPU 型号、SM 数、显存、驱动、CUDA/nvcc、编译器、编译 flags、目标 arch、OS、LLM 模型/版本/采样参数及超时重试设置。

主负载：512、1024、2048、4096 方阵；补充长宽不对称、large-K 和非 tile 整除尺寸。主比较先固定 FP32、布局、alpha/beta，再增加 beta=0/1/general 的语义覆盖。

正确性输入应包含正负随机数、不同数值范围和边界尺寸，避免只用单一整数模式掩盖索引或舍入问题。

### 第 13 页：实验设计——公平基线与计时

主基线：严格 FP32 配置的 cuBLAS SGEMM，输入、alpha/beta、布局和误差标准与 SCOPE 一致。

辅助基线：QiMeng CUDA 代码及直接 LLM 生成；只在可以统一 harness、语义和编译参数时比较。

TF32/cuBLASLt/Torch matmul 单独列出计算模式，不与 SIMT FP32 混作同一公平基线。不能根据变量是 float 就认定没有 Tensor Core 或降精度路径。

编译一次后执行多次：当前终端测量配置为 warmup 2 次、正式 5 次进程运行。报告平均延迟及方差/标准差，补充中位数。

单次 GEMM 使用 CUDA events、同步和足够的内部重复，计时排除编译、初始化与主机拷贝；固定主频条件或记录温度/功耗，GPU 性能测量串行执行。

若每次工作量相同，主要 GFLOPS = 2MNK / (平均 latency_ms × 10^6)。各次 GFLOPS 的均值与由平均延迟计算的 GFLOPS 不完全相同，论文与排名必须统一口径。

如果 beta 非零且反复写回 C，每次重复需确保测试语义一致；正确性比较和计时不要使用已经被不同次数更新的 C。

### 第 14 页：评价指标与消融

主要指标：正确候选生成率、最终 GFLOPS、相对 cuBLAS speedup、跨形状几何平均 speedup、Top-3 有效结果数量。

过程指标：总生成时间、LLM 调用数/token、编译次数、修复次数、失败类型、各阶段路径数与候选保留率。

建议消融：

1. 直接完整代码生成 vs 局部 Patch。
2. 不使用策略图 vs requires/conflict 过滤。
3. 全部 LLM 代码生成 vs 确定性策略执行。
4. 平坦 Tiling 选择 vs Block/Warp/Thread 层次化剪枝。
5. 中间阶段运行验证 vs Phase 1 完成验证。
6. 原始错误反馈 vs 结构化诊断与局部修复。
7. Phase 1 only vs size-aware batch unlock。

所有组使用相同需求、硬件、模型、预算和 harness，多次独立搜索。不能用本组历史最佳代码反向提示 LLM，再将结果作为无历史条件的主比较。

### 第 15 页：结果展示与下一步

性能表：硬件 × 形状，列出 SCOPE Phase 1、Phase 2、cuBLAS FP32、QiMeng、speedup 和正确性；无结果标记失败或未测，不填 0 GFLOPS 冒充有效结果。

过程图：候选数 → 合法 IR → 完整代码 → 编译通过 → 正确性通过 → unlock → Top-3，展示路径损失与原因。

案例图：同一 kernel 在两个形状的差异，以及修复前后的加载映射；性能原因结合 PTXAS 寄存器/共享内存和实际 profiler，而不是仅看策略名字。

下一步：冻结代码/策略版本 → 小规模端到端回归 → 三种主尺寸验证 → 公平 FP32 基线 → 多硬件扩展 → 消融与论文写作。

## 3. 候选论文贡献表述

1. 提出基于 OptIR 与可检查策略依赖的 GEMM 代码构造框架，协调参数、布局、映射和流水线状态。
2. 提出 LLM 策略决策与确定性/局部代码变换相结合的构造流程，并将完整验证推迟到可执行实现形成后。
3. 提出 verifier/profile 驱动的 size-aware batch 优化及结构化诊断修复，实现可记录、可重试、可回滚的优化搜索。

这些是待通过相关工作与消融支撑的贡献候选，不宣称首次提出。当前搜索更准确地称为策略图引导的分支搜索，不能只凭名称描述为具有标准种群/交叉/变异的进化算法。

## 4. 建议导师汇报结语

目前工作已经从单次 LLM 生成推进到可追踪的策略组合与验证修复框架。最近发现的主要问题不是缺少优化名称，而是策略实现一致性、跨阶段构造依赖和修复区域定位。下一阶段先冻结新流程并确认稳定正确性，再通过统一 FP32 基线和消融判断各模块是否真正提升性能与生成效率。
