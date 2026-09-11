# 复杂任务子任务拆解与大小模型路由：数据集、核心框架与 WSL 实现指导

> 文档日期：2026-08-24
> 本文只确定研究方案与实现边界，不在当前阶段生成核心框架代码。
> 实施纪律：第九章“第一轮”是独立无训练实验，必须单独运行和验收；第十章以后另开任务实施。

## 一、先给结论

本课题最合适的研究问题不是“先固定拆解，再做一次模型路由”，而是：

> 面对一个尚未完成的复合节点，系统应当直接交给小模型、直接交给大模型，还是继续把它拆成更细的子任务？

建议采用暂定名 **AdaSplitRoute（负载感知的自适应拆解与节点路由）**。框架从粗粒度任务图开始，对每个节点反复执行三选一决策：

```text
继续拆解（DECOMPOSE） / 小模型执行（EXECUTE_SMALL） / 大模型执行（EXECUTE_LARGE）
```

研究重点按优先级排列为：

1. 复杂任务怎样形成语义完整、依赖明确、粒度适中的 DAG；
2. 怎样用大小模型在真实子任务上的执行结果定义“难度”；
3. 怎样联合决定“继续拆”还是“选择模型”；
4. 服务负载与关键路径怎样改变拆解粒度和路由阈值；
5. 在最终质量只小幅变化时，是否显著减少大模型调用、服务成本和高分位时延。

明确不纳入本文核心：模型池选择、故障恢复、在线结果验证、多智能体角色设计、复杂结果聚合、GPU 部署搜索。聚合只做必要的结果合成，正确性判断主要用于离线实验评测。

---

## 二、本地论文筛选结果

两个文件夹中的 25 份论文都已检查。真正能够共同支撑本课题的不是某一篇完整方案，而是下列八篇论文分别提供的能力。

### 2.1 必须精读的八篇

| 优先级 | 论文                                                                                                         | 发表情况                            | 对本课题的直接作用                                                   | 不能照搬的部分                                                                              |
| -----: | ------------------------------------------------------------------------------------------------------------ | ----------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
|      1 | [Decomposed Prompting](./子任务拆解/10_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf) | ICLR 2023                           | 拆解器、独立子任务处理器、递归拆解；定义“真正的子任务”而非一段 CoT | 主要是静态/顺序执行，不感知负载，也不做大小模型路由                                         |
|      2 | [RouteGoT](./模型路由/01_RouteGot.pdf)                                                                        | 2026 arXiv 预印本，尚非正式会议论文 | 最接近本课题：推理图节点级路由、继续拆解、成功率预测和预算控制       | 仅做 Token 预算；不感知实时队列和关键路径；拆解策略与模型规模绑定；与本课题存在最大创新冲突 |
|      3 | [Murakkab](./子任务拆解/01_Murakkab_Resource_Efficient_Agentic_Workflow_Orchestration_OSDI2026.pdf)           | OSDI 2026                           | 证明工作流结构、模型选择、硬件状态和负载需要联合考虑                 | 工作流多由开发者给定，不负责从任意复杂问题生成语义任务图                                    |
|      4 | [RouteLLM](./子任务拆解/06_RouteLLM_Learning_to_Route_LLMs_with_Preference_Data.pdf)                          | ICLR 2025                           | 学习型大小模型路由器、路由阈值、成本—质量曲线                       | 只路由完整请求；训练标签不是独立子任务执行结果                                              |
|      5 | [Agentix](./子任务拆解/03_Agentix_Efficient_Serving_Engine_for_LLM_Agents_NSDI2026.pdf)                       | NSDI 2026                           | 动态 DAG、运行时关键路径、负载条件下的程序级调度                     | 不生成任务语义，不选择大小模型；关键路径只代表时延敏感性，不代表节点可省略                  |
|      6 | [Teola/Ayo](./子任务拆解/02_Teola_End_to_End_Optimization_of_LLM_Applications_ASPLOS2025.pdf)                 | ASPLOS 2025                         | 把应用展开成细粒度可执行数据流图，显式暴露依赖和并行性               | 从预定义应用展开，不解决开放问题的语义拆解                                                  |
|      7 | [BOUTE](./模型路由/02_BOute_Cost_Efficient_LLM_Routing_and_Deployment_MLSys2026.pdf)                          | MLSys 2026                          | 把请求难度、队列时延、SLO 和模型负载放入路由代价                     | 完整 GPU 部署与贝叶斯优化过于工程化；只做整题路由                                           |
|      8 | [CALM](./模型路由/07_CALM_Confident_Adaptive_Language_Modeling.pdf)                                           | NeurIPS 2022                        | 在验证集上校准置信阈值，使质量损失有可控含义                         | 是单模型内部的层级早退，不是跨模型、跨子任务路由                                            |

### 2.2 应作为基线或补充的论文

- [Least-to-Most Prompting](./子任务拆解/11_Least_to_Most_Prompting_Enables_Complex_Reasoning.pdf)：最合适的固定顺序拆解基线。
- [Tree of Thoughts](./子任务拆解/12_Tree_of_Thoughts_Deliberate_Problem_Solving_with_LLMs.pdf)：借鉴“展开还是停止”的搜索思想，但它的节点是候选思维状态，不等同于可独立路由的子任务。
- [ReAct](./子任务拆解/13_ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf)：借鉴执行反馈驱动的动态展开，但本课题不把在线恢复作为研究内容。
- [FrugalGPT](./模型路由/03_FrugalGPT_How_to_Use_LLMs_While_Reducing_Cost_and_Improving_Performance.pdf)：适合作为“小模型先做、低置信再升级”的级联基线。
- [HELIOS](./模型路由/04_HELIOS_Adaptive_Model_and_Early_Exit_Selection_MLSys2026.pdf)：只借鉴周期性在线 profiling 和对请求流变化的适配；其核心是模型选择与层级早退，不是语义节点路由。
- [Parrot](./子任务拆解/04_Parrot_Efficient_Serving_of_LLM_Applications_OSDI2024.pdf)：借鉴语义变量和 LLM 调用间的数据依赖表示。
- [AI Metropolis](./子任务拆解/09_AI_Metropolis_Out_of_Order_Multi_Agent_Execution_MLSys2025.pdf)、[FlashAgents](./子任务拆解/08_FlashAgents_Accelerating_Multi_Agent_LLM_Systems_MLSys2026.pdf)：补充乱序、并行和流水执行的系统依据。

### 2.3 不应成为核心框架的论文

- Twill、RollART：优化的是模型/训练阶段到异构硬件的调度，不解决语义子任务拆解。
- LLM-Blender：所有模型都生成候选再融合，与降低调用成本的目标相反。
- Splitwise：拆分 Prefill/Decode 计算阶段，不是按语义难度选择大小模型。
- Speculative Decoding：小模型草拟、大模型验证 Token，不是节点级任务路由。
- PAL、CoT、Self-Consistency、Zero-shot CoT：可以作为推理方法或难度信号，但不生成独立可调度 DAG。

### 2.4 必须规避的创新重合

RouteGoT 已经完成了“推理图节点级路由 + 递归拆解 + 成功率预测 + Token 预算”。进一步检索还发现 [R2-Reasoner / Route-and-Reason（WWW 2026）](https://doi.org/10.1145/3774904.3793038)：它把复杂任务拆成顺序子任务，再在 9 个异构模型之间分配，并联合优化拆解器与分配器。其[官方仓库](https://github.com/tsinghua-fib-lab/R2-Reasoner)还提供部分分解与分配训练数据。它是比一般整题路由器更直接的竞品和必须复现的基线。

因此，“子任务拆解 + 按难度分配不同规模模型”本身已经不是足够的新意。如果本课题只复现 RouteGoT 或 R2-Reasoner 的内容，创新不足。

本课题应坚持以下差异：

1. **拆解不是一次性前处理，而是可继续触发的运行时动作。** 同一个未完成节点在不同能力与负载条件下，可停止拆解并选择模型，也可继续局部细化。
2. **只研究大小模型两级，突出粒度而非模型池搜索。** R2-Reasoner 的重点是 9 模型协作；本文把变量收紧为“节点是否需要继续拆、拆到何处、何时交给小/大模型”。
3. **路由器用真实中间节点日志训练。** 不以主观难度、整题标签或单次 token 置信度代替选定大小模型的节点级实际成败。
4. **拆解粒度随实时服务状态变化。** 代价包含队列等待和关键路径时延，而不仅是 Token 或 API 预算。
5. **研究对象是可执行语义任务 DAG。** 每个节点有明确目标、输入、输出和依赖，允许串并行与局部替换，不把任意“思维片段”当成子任务序列。

---

## 三、重新检索后的数据集选择

### 3.1 选择标准

本课题需要分别测量四种能力，不能把最终答对率当成全部证据：

| 能力             | 数据必须提供什么                                           |
| ---------------- | ---------------------------------------------------------- |
| 是否以及怎样拆解 | 原始复杂任务、子任务和粒度信息                             |
| 图结构质量       | 节点、依赖、串并行或层次关系                               |
| 节点路由         | 每个子任务有标准答案或可执行结果，能实测大小模型成败       |
| 系统收益         | 存在足够的 fan-out、关键路径和真实请求分布，可测成本与时延 |

截至 2026-08-24，没有一个公开顶会数据集同时具备“自然复杂任务、人工 Gold DAG、节点答案、执行反馈、真实负载和大小模型路由标签”。因此采用**开发集、结构集、系统集、最终留出集分工**，而不是寻找一个万能主数据集。

### 3.2 最值得采用的数据集

| 优先级与角色                   | 数据集                   | 发表情况与规模                                                         | 可直接利用的监督                                                                      | 主要限制                                                                            |
| ------------------------------ | ------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **P0 路由开发主集**      | **MuSiQue-Ans**    | TACL 2022；24,814 个可回答的 2–4 跳问题                               | 自然语言子问题、逐步答案、支持段落、依赖和最终答案；划分成熟                          | 图短、问答域较窄、并行度有限                                                        |
| **P0 路由外部集**        | **MEQA**           | NeurIPS 2024 Datasets & Benchmarks；论文 2,243 条，当前仓库称 2,093 条 | 每一步 explanation 都是“子问题 + 子答案”；事件、时间和比较关系增加节点类型          | 论文与仓库计数不一致，必须固定 commit；模板化事件问答可能形成捷径                   |
| **P1 可控复杂任务**      | **TaskCraft**      | ICLR 2026；约 41K 工具密集任务、12.6K 工具轨迹、5K 多跳拆解            | 通过深度扩展增加依赖链、宽度扩展增加并行分支；有任务答案和可复现执行轨迹              | 合成数据；并非通用人工 DAG；HF JSON 字段类型不统一，需自写解析器                    |
| **P1 人工复杂 DAG 评测** | **FanOutQA-dev**   | ACL 2024 Short；1,034 个根问题、7,305 个人工子问题；310 dev 可公开评测 | 嵌套 decomposition、节点答案、`depends_on` 和证据；天然 fan-out 与汇合              | 完整公开标注规模小，724 个 test 的答案隐藏；检索会带来额外噪声                      |
| **P1 系统主集**          | **PARALLELPROMPT** | NeurIPS 2025 Datasets & Benchmarks；超过 37K 个真实聊天请求            | 模板、共享上下文、迭代输入和执行套件；直接比较串行/并行时延与语义保真                 | 官方只发布 train split，需自建冻结评测子集；主要是 map 型结构，没有逐节点 Gold 答案 |
| **P2 最难留出集**        | **MoNaCo**         | TACL 2026；1,315 个真人复杂问题                                        | QDMR、依赖赋值、逐步子问题/答案、证据、算子输出和完整执行轨迹；单题可展开几十至数百步 | HF 需同意访问；无标准划分；作者强调避免基准污染，本文仅作最终留出评测               |
| **P2 最新规模泛化**      | **MINTQA**         | ACL 2026 Long；28,366 个问题                                           | 1–4 跳 Gold 子问题、子答案、事实和新旧/流行长尾知识属性                              | 基本为链式合成问答；知识熟悉度与推理难度混杂；需自建无泄漏划分                      |
| **P2 图生成基准**        | **WorFBench**      | ICLR 2025；18,679 train、2,146 test                                    | Gold 节点、链和 DAG 边；node/chain/graph 指标完整                                     | 没有节点执行答案，不能单独训练大小模型路由器                                        |

官方入口：

- [MuSiQue 官方仓库](https://github.com/stonybrooknlp/musique)
- [MEQA 官方论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e560a0b22e4432003d0dba63ff8dc457-Abstract-Datasets_and_Benchmarks_Track.html)与[仓库](https://github.com/du-nlp-lab/MEQA)
- [TaskCraft ICLR 2026 页面](https://iclr.cc/virtual/2026/poster/10009241)、[仓库](https://github.com/OPPO-PersonalAI/TaskCraft)与[数据](https://huggingface.co/datasets/PersonalAILab/TaskCraft)
- [FanOutQA 论文](https://aclanthology.org/2024.acl-short.2/)与[仓库](https://github.com/zhudotexe/fanoutqa)
- [PARALLELPROMPT 论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6bbefc73a187dd42e0dc065b4e7a0615-Abstract-Datasets_and_Benchmarks_Track.html)与[数据](https://huggingface.co/datasets/forgelab/ParallelPrompt)
- [MoNaCo 论文](https://aclanthology.org/2026.tacl-1.2/)与[受控访问数据](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark)
- [MINTQA 论文](https://aclanthology.org/2026.acl-long.18/)与[仓库](https://github.com/probe2/multi-hop)
- [WorFBench 论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/adbe936993aa7cf41e45054d8b72f183-Abstract-Conference.html)与[仓库](https://github.com/zjunlp/WorFBench)

### 3.3 最终采用顺序

**第一轮是独立的无训练可行性实验**，只使用公开可下载但被固定为评测用途的子集，不训练拆解器或路由器：

```text
MuSiQue-dev：在固定 Gold 图上测大小模型的真实节点能力差异
FanOutQA-dev：零样本生成 DAG，并与人工子任务、答案和依赖比较
PARALLELPROMPT：使用官方 reference schema，比较整题、reference schema 串行和 reference schema 并行的语义保真与时延
```

第二轮以后才加入训练和扩展数据：

```text
MEQA：验证节点能力和路由结论能否迁移到事件、时间和比较问题
TaskCraft：控制拆解深度/宽度，研究动态粒度和路由收益
WorFBench：评测未知任务上的节点、链和 DAG 生成
```

最后才加入 MoNaCo 和 MINTQA。MoNaCo 不训练、不调阈值，只报告严格留出结果；MINTQA 扩大链式路由样本并检查知识新颖性泛化。

### 3.4 顶会拆解论文常用数据为何不能直接当主集

近期方法论文常用 MATH/GSM8K、HumanEval、HotpotQA、SCAN、CSQA、MT-Bench 等验证最终性能。例如 [Select-Then-Decompose（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.278/)使用多种数学、代码和问答任务；与本课题最接近的 [R2-Reasoner（WWW 2026）](https://doi.org/10.1145/3774904.3793038)使用 P3、SCAN、MATH、CHAMP、CSQA 和 MuSiQue。

这些数据大多只有整题答案，没有人工 Gold 子任务、依赖和节点答案。它们适合做“最终能否答对”的方法基线，却不能判断是拆解器、节点路由器还是合成器出了错。因此本文只保留 MuSiQue 作为核心数据源；其他任务至多作为与原论文对齐的附加实验。

### 3.5 只作辅助或暂不采用

- **Researchy Questions（SIGIR 2025）**：96,448 个真实复杂搜索问题，有多面性、推理性、`decompositional_score` 和 GPT-4 分解，适合训练“是否需要拆解/问题类型”辅助分类器；没有节点答案和 Gold 边，不能评测拆解正确性。[数据](https://huggingface.co/datasets/corbyrosset/researchy_questions)
- **WorkflowBench/WorkflowLLM（ICLR 2025）**：106,763 个工作流、1,503 个 API、83 个应用，适合拆解器预训练；控制流隐含在 Python 风格工作流中，没有统一节点答案。[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d7259031023c5aa463187c4a31c95c8-Abstract-Conference.html)
- **TaskLAMA（AAAI 2024）**：人工开放任务 DAG 很有价值，但没有节点执行真值；若需要展示跨域人工拆解，可替代 WorFBench 做小规模外部验证。[论文](https://research.google/pubs/tasklama-probing-the-complex-task-understanding-of-language-models/)
- **CofCA（ICLR 2025）**：反事实多跳问题带中间子问答，适合检查模型是否依靠参数记忆；但数据小、test-only 且长期下载地址稳定性较弱，放在可选鲁棒性实验。[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2628d4d3b054c2d7ad33ab03435204f4-Abstract-Conference.html)
- **TaskBench（NeurIPS 2024 Datasets & Benchmarks）**：保留为可执行工具图备选；PARALLELPROMPT 更贴近本课题的真实请求、拆解并行和系统时延主张。
- **AgentSynth（ICLR 2026）/WorkArena++（NeurIPS 2024 Datasets & Benchmarks）**：有可执行长任务和逐步验证，但环境、GUI和工具噪声太大，只适合后期系统压力测试。
- **AgenticRAGTracer（Findings of ACL 2026）**：顺序/并行结构和逐跳答案很契合，但截至本文日期官方仓库尚未发布论文所述 1,305 条数据，列入 watchlist，不作为当前依赖。
- **BREAK（TACL 2020）**：83,978 个 QDMR 可用于预训练拆解器，但无标准子任务答案，不能作为路由主集。

### 3.6 路由标签与划分纪律

节点“难度”不采用数据集自带的 easy/medium/hard 或知识流行度直接代替，而由选定大小模型的真实重复执行结果产生。第一轮的 MuSiQue 节点能力实验必须给每个节点相同的 Gold 支持上下文、Gold 前驱输出和解码设置，先隔离检索与误差传播：

```text
EITHER：大小模型都稳定正确，成本优先时选小模型
SMALL_ONLY：仅小模型正确，作为异常样本单独分析
LARGE_NEEDED：小模型失败、大模型稳定正确
UNSOLVED：两者都失败，不应被路由准确率掩盖
```

必须按根问题、源文档或实体整体划分，禁止把同一个根问题展开后的节点随机分到 train/test。MINTQA 需按实体或关系去重后再划分；TaskCraft 需按原子 seed 或内容来源分组；MoNaCo 与 FanOutQA test 不参与训练和阈值选择。

### 3.7 WSL 中的数据准备

不要把第三方数据混入 `src/`，也不要提交大型原始数据：

```text
adaptive-subtask-routing/
├── data/
│   ├── raw/          # .gitignore，只读原始数据
│   ├── processed/    # 统一后的 TaskGraphSample JSONL
│   └── manifests/    # 来源、commit、校验和、许可和划分
└── third_party/      # 官方评测脚本或只读仓库
```

第一轮只下载 MuSiQue、FanOutQA 和 PARALLELPROMPT：

```bash
mkdir -p /mnt/e/HiNA/HiNA/adaptive-subtask-routing
cd /mnt/e/HiNA/HiNA/adaptive-subtask-routing
mkdir -p data/raw data/processed data/manifests third_party

git clone https://github.com/StonyBrookNLP/musique.git third_party/musique
(cd third_party/musique && bash download_data.sh)

git clone https://github.com/zhudotexe/fanoutqa.git third_party/fanoutqa

python -m pip install -U huggingface_hub
hf download forgelab/ParallelPrompt --repo-type dataset \
  --local-dir data/raw/parallelprompt
```

第二轮以后再下载：

```bash
hf download PersonalAILab/TaskCraft --repo-type dataset \
  --local-dir data/raw/taskcraft

git clone https://github.com/du-nlp-lab/MEQA.git third_party/MEQA
git clone https://github.com/zjunlp/WorFBench.git third_party/WorFBench
git clone https://github.com/probe2/multi-hop.git third_party/mintqa
```

注意：

1. MuSiQue 只需原始 JSONL 和 evaluator，不安装其完整旧 AllenNLP 环境。
2. TaskCraft 当前 HF viewer 会因同一字段在不同记录中类型不一致而报错；使用 `hf download` 下载原始文件，再逐行容错解析，不直接依赖 `load_dataset()`。
3. 第一轮只使用 FanOutQA 的公开 dev 标注，不使用隐藏 test 答案，也不把 Gold decomposition 放入零样本拆解提示。
4. PARALLELPROMPT 官方只发布 train split；第一轮按 `source/source_index` 去重后冻结自建 evaluation subset，校准 ID 与正式 ID 永久互斥。
5. MEQA 论文与仓库样本数不同，第二轮接入时 manifest 必须保存 commit、各 split 实际行数和 SHA-256。
6. MoNaCo 需在 HF 登录并同意访问条件后单独下载；原始文件不得再分发，也不得进入训练语料。

---

## 四、确定的核心框架

### 4.1 一条关键路径

```text
复杂任务 + 当前服务状态
          │
          ▼
  生成粗粒度初始 DAG
          │
          ▼
 对就绪节点估计：小模型可完成性、可拆性、拆解收益
          │
          ▼
 ┌─────────────────────────────────────┐
 │ 继续拆解 │ 小模型执行 │ 大模型执行 │
 └─────────────────────────────────────┘
          │
          ▼
  更新 DAG、关键路径和就绪队列
          │
          ▼
      拓扑执行并合成最终答案
```

这不是“先一次性拆到底，再路由”，而是**粗到细的在线图构造**：困难复合节点只有在拆开后确实更可能由小模型完成、且额外开销可接受时才继续拆。

### 4.2 统一节点表示

每个子任务节点至少包含：

```json
{
  "id": "n3",
  "objective": "找出人物 A 就读的大学",
  "task_type": "retrieval_qa",
  "input_refs": ["root.question", "n1.output"],
  "depends_on": ["n1"],
  "context_refs": ["paragraph_7"],
  "output_schema": {"type": "short_answer"},
  "splittable": true,
  "depth": 1
}
```

运行时字段与语义字段分开保存：

```json
{
  "p_small_success": 0.82,
  "p_large_success": 0.96,
  "estimated_small_ms": 180,
  "estimated_large_ms": 760,
  "small_queue_ms": 30,
  "large_queue_ms": 900,
  "on_critical_path": false,
  "decision": "EXECUTE_SMALL"
}
```

### 4.3 拆解器只负责什么

拆解器负责：

- 从复杂目标生成少量语义完整的粗粒度节点；
- 写清节点目标、输入引用、输出模式和依赖；
- 对被选择为 `DECOMPOSE` 的节点局部细化，并用子图替换原节点；
- 保证子图的入口能够获得原节点输入，子图出口能够提供原节点承诺的输出。

拆解器不负责：

- 直接指定节点必须由哪个具体模型执行；
- 为了节点数更多而无限细分；
- 生成“理解问题、认真思考、检查答案”一类不可独立执行的伪节点；
- 改变任务语义或遗漏最终答案所需步骤。

### 4.4 停止拆解条件

只要满足以下任一条件，就停止细化并进入大小模型选择：

- 当前节点已是单一、明确、可独立判分的操作；
- 小模型成功率已达到校准阈值；
- 节点内部高度耦合，拆开会丢失全局语义；
- 继续拆解不能减少预期大模型用量；
- 新增调用、重复上下文、通信和合成开销超过节省；
- 会造成过长串行链，显著拉长关键路径；
- 达到最大深度、最大节点数或拆解预算。

### 4.5 节点难度的正确定义

不要把“长问题”“多步骤”直接当成难题，也不要仅让大模型主观打 easy/medium/hard。

本课题中最有用的难度定义是：

> 在给定节点输入、上下文和上游结果的条件下，小模型正确完成该节点的概率。

离线对同一节点分别运行大小模型，得到四类真实结果：

| 小模型 | 大模型 | 标签含义         | 路由训练处理                           |
| ------ | ------ | ---------------- | -------------------------------------- |
| 正确   | 正确   | `EITHER`       | 小模型正例                             |
| 错误   | 正确   | `LARGE_NEEDED` | 大模型正例                             |
| 错误   | 错误   | `UNSOLVED`     | 单独统计；不把它伪装成`LARGE_NEEDED` |
| 正确   | 错误   | `SMALL_ONLY`   | 保留分析，检查采样、提示和评测误差     |

第一轮在固定提示、解码配置和随机种子规则下，对正式样本中的大小模型各独立运行至少 3 次；稳定正确的判定阈值必须在正式运行前写入配置，不能只凭一次输出生成标签。

### 4.6 第二轮：难度估计器的输入

第二轮采用轻量模型即可，不急于微调一个新的 LLM。候选特征：

- 节点文本表示或 embedding；
- 任务类型、答案类型、输入/上下文 Token 数；
- 节点深度、入度、出度、祖先结果长度；
- 是否含比较、组合、约束、计算、代码或多证据；
- 小模型的回答置信信号（仅用于“小模型先执行再升级”基线）；
- 历史上同类节点的小模型成功率。

优先比较 Logistic Regression、XGBoost/LightGBM 或小型 MLP。只有它们明显不足时才增加复杂路由模型。

### 4.7 第四轮：服务负载怎样进入决策

服务负载只改变“怎样完成节点”，不能改变语义依赖或删除节点。

- 大模型队列很长：对非关键路径且可并行的复合节点，更倾向继续拆解并下沉到小模型。
- 小模型队列很长或时延 SLO 很紧：减少过细拆解；必要时把关键节点直接交给大模型。
- 关键路径节点：提高小模型成功率阈值，避免错误传播；同时谨慎增加串行层数。
- 非关键路径节点：允许更积极的细化，只要不会增加最终汇合等待。
- 队列状态只在节点决策边界读取，初版不做连续抢占和迁移。

### 4.8 第四轮：三选一控制器

控制器对每个就绪节点比较三种动作的预期结果：

- `EXECUTE_SMALL`：小模型质量足够，且成本/等待有优势；
- `EXECUTE_LARGE`：小模型成功率不足，或节点耦合强、位于关键路径、继续拆解收益低；
- `DECOMPOSE`：节点仍是复合任务，拆开后多个子节点有望由小模型处理，节省足以覆盖额外开销。

先实现规则策略，得到稳定可解释基线；再用离线日志学习三动作策略。不要一开始上强化学习。这两节都不属于第一轮实现范围。

---

## 五、如何把拆解和路由分别测清楚

端到端正确率下降时，必须判断是“拆错了”还是“路由错了”。因此第一轮先完成三个彼此隔离的无训练实验；训练与联合控制另起后续轮次。

### 第一轮 A：Gold 图 + Gold 上游答案，只测节点能力

在 MuSiQue Gold DAG 上让每个节点只接收允许的支持上下文和 Gold 前驱答案，分别调用大小模型并用节点标准答案判分。这里只生成真实能力标签，并比较全小、全大和后验 Oracle 上界，不训练难度估计器。

### 第一轮 B：预测图 + 固定大模型执行，只测拆解

在 FanOutQA-dev 上零样本生成 DAG，提示中不提供 Gold 分解。所有变体共享去节点映射的根问题级证据池，并比较整题直接回答、预测图大模型执行、Gold 图大模型执行和 Gold 节点输出直接合成，依次隔离图结构、节点执行与合成错误。

### 第一轮 C：相同模型 + 不同调度，只测串并行收益

在 PARALLELPROMPT 冻结子集上直接使用官方 reference schema，保持模型、节点提示、聚合器和生成设置一致，只比较整题、reference schema 串行和 reference schema 并行三种组织方式。这里测量的是参考拆解暴露的并行性能否转化为时延收益；该结构不是人工 Gold，不能称为 Oracle，也不宣称预测拆解收益或引入负载感知路由。

### 第二轮以后：学习与联合评测

第二轮加入 MEQA 并训练节点难度估计器；第三轮加入 TaskCraft/WorFBench 研究动态粒度；第四轮再组合预测图、学习型路由和负载回放，测量真实质量—成本—时延 Pareto 曲线。只有离线结果支持核心假设后，才考虑接入真实 serving 队列与批处理。

---

## 六、实验设计

### 6.1 必须有的基线

1. `All-Large / No-Decomposition`：整题全部交给大模型；
2. `All-Small / No-Decomposition`：整题全部交给小模型；
3. `Query-Level Router`：整题只做一次大小模型路由，RouteLLM 风格；
4. `Fixed DAG / All-Large`；
5. `Fixed DAG / All-Small`；
6. `Fixed DAG / Node Router`：只做节点路由，不动态改变粒度；
7. `Adaptive Decomposition / Static Router`：只动态拆解，路由阈值固定；
8. `AdaSplitRoute`：动态拆解 + 节点路由 + 负载/关键路径；
9. `Oracle Cheapest-Correct`：知道每个节点真实结果的成本下界，仅作上界参考。

第一轮按数据集固定基线，不做泛化扩张：MuSiQue 使用 Gold 图的 `all_small / all_large / oracle_route`；FanOutQA 使用 `direct_large / predicted_graph_all_large / gold_graph_all_large / gold_graph_gold_outputs`；PARALLELPROMPT 使用 `whole_request / reference_schema_serial / reference_schema_parallel`。`Query-Level Router`、学习型 `Node Router`、动态拆解和 `AdaSplitRoute` 全部留到后续轮次。

### 6.2 指标

拆解质量：

- 节点 Precision、Recall、F1：用语义匹配加 Hungarian matching，防止重复节点刷分；
- Edge-F1、DAG 合法率、缺失依赖率；
- 节点数量、最大深度、平均并行宽度、关键路径长度；
- 过拆率：能够由一个节点稳定完成却被继续拆开的比例。

路由学习质量（第二轮以后）：

- 路由准确率、F1、AUROC、校准误差；
- Oracle match 与 routing regret；
- `LARGE_NEEDED` 召回率，避免为了省成本漏掉真正困难节点。

端到端质量：

- 第一轮 MuSiQue 的节点答案 EM/F1 与四类能力标签分布；第二轮再加入 MEQA；
- FanOutQA 的节点答案、依赖边与最终答案指标；
- WorFBench 的 node/chain/graph F1（第三轮以后）；
- TaskCraft 的节点/最终答案、工具轨迹成功率与按深度/宽度分组结果（第三轮以后）；
- PARALLELPROMPT 的 semantic fidelity；该数据没有逐节点 Gold 答案，不报告 accuracy。

系统指标：

- 大模型调用数、调用比例、输入/输出 Token 占比；
- 总模型成本、拆解器成本、通信/合成成本分别统计；
- 第一轮报告 p50/p95 端到端时延和拆解/执行/合成分解；后续轮次再报告吞吐量、SLO 达标率和队列等待；
- PARALLELPROMPT 上串行与并行执行的加速比，以及并行化后的质量变化。

最终图表必须报告质量—成本和质量—p95 时延的 Pareto 曲线，不能只报告“节省百分比”。

### 6.3 消融实验

以下完整消融从第三轮开始，不属于第一轮交付：

- 去掉服务负载；
- 去掉关键路径信号；
- 固定拆解粒度；
- 不允许递归拆解；
- 主观难度标签 vs 真实节点执行标签；
- 规则阈值 vs 学习型难度估计；
- 拆解与路由绑定 vs 解耦；
- 不同最大深度、节点数和质量容忍度。

### 6.4 论文核心假设的判定条件

只有同时观察到以下结果，核心主张才成立：

1. 同一复杂任务内确实同时存在 `EITHER/SMALL_ONLY` 和 `LARGE_NEEDED` 节点；
2. 节点级路由比整题路由减少更多大模型计算，同时质量下降受控；
3. 动态粒度比固定粗图或固定细图有更好的质量—成本/时延权衡；
4. 负载变化时，最优拆解粒度或路由阈值发生可重复的变化；
5. 收益在至少两个不同结构的数据集上成立，而不是 MuSiQue 特例。

---

## 七、独立原型工程原则

本课题新建独立仓库 `adaptive-subtask-routing`，不读取、导入或依赖其他实验项目的代码和数据结构。这样可以避免旧任务类型、旧标签体系和旧运行流程限制新的 DAG 抽象。

独立原型只保留研究所必需的五层：

- 数据层：把不同数据集转换为统一任务图格式；
- 拆解层：生成粗粒度 DAG，并对指定节点局部细化；
- 路由层：预测小模型可完成性，执行三选一策略；
- 执行层：按依赖运行节点，记录负载、成本和时延；
- 评测层：分别测量图结构、节点答案、最终质量和系统指标。

第一轮使用统一的 OpenAI-compatible 模型接口即可，同时支持两个任务执行端点：`small_model` 和 `large_model`；另允许一个只判分、不执行节点且费用单列的 `evaluator_model`。第一轮的路由层只生成能力标签和 Oracle 上界，不训练策略；本地模型、学习型路由和动态控制以后通过适配器加入，不应影响图与实验接口。

所有真实子任务标签都必须来自独立节点的实际执行结果。不能用整段回答中的“规划、计算、检查”等主观阶段标签代替节点真值。

---

## 八、建议代码结构

### 8.1 第一轮实际创建的结构

第一轮只创建下列文件。不要预建学习型路由、动态拆解、负载状态或后续数据集的空壳，避免 WSL Codex 越界实现。

```text
adaptive-subtask-routing/
├── pyproject.toml
├── README.md
├── src/splitroute/
│   ├── decomposition/
│   │   ├── schemas.py             # TaskGraph、TaskNode、TaskGraphSample
│   │   ├── graph_validate.py      # 唯一 ID、边存在、无环、引用可解析
│   │   └── planners/
│   │       ├── base.py
│   │       ├── oracle.py          # 读取数据集金标准图
│   │       └── prompt_planner.py  # 第一轮只生成一次性完整 DAG
│   ├── datasets/
│   │   └── graph_adapters/
│   │       ├── musique.py
│   │       ├── fanoutqa.py
│   │       └── parallelprompt.py
│   ├── routing/
│   │   ├── labels.py              # 四类真实执行标签
│   │   └── oracle_policy.py       # 仅作后验上界，非学习型路由
│   ├── models/
│   │   ├── client.py              # small/large 两个任务执行端点
│   │   ├── evaluator.py           # 只判分，不执行节点
│   │   └── cache.py               # 响应缓存与断点续跑
│   ├── execution/
│   │   ├── dag_executor.py
│   │   ├── node_context.py
│   │   ├── synthesis.py           # 固定合成器，便于隔离合成错误
│   │   └── profiler.py
│   ├── evaluators/
│   │   ├── graph_evaluator.py
│   │   ├── node_answer_evaluator.py
│   │   └── system_evaluator.py
│   └── experiments/
│       └── first_round/
│           ├── data_audit.py
│           ├── run_node_capability.py
│           ├── run_zero_shot_decomposition.py
│           ├── run_parallel_execution.py
│           └── build_report.py
├── configs/
│   ├── first_round.yaml
│   └── models.example.yaml
├── tests/
│   ├── test_graph_schema.py
│   ├── test_graph_validate.py
│   ├── test_musique_adapter.py
│   ├── test_fanoutqa_adapter.py
│   ├── test_parallelprompt_adapter.py
│   ├── test_fanoutqa_evidence_protocol.py
│   ├── test_model_roles.py
│   ├── test_response_cache.py
│   ├── test_node_labels.py
│   ├── test_dag_executor.py
│   ├── test_synthesis.py
│   └── test_parallel_scheduler.py
├── data/
│   ├── raw/                       # 不提交仓库
│   ├── processed/
│   └── manifests/
└── outputs/
    └── first_round/               # 第一轮全部日志、指标、图和报告
```

第二轮以后才按需增加 MEQA/TaskCraft/WorFBench/MINTQA/MoNaCo adapter、`routing/features.py`、`routing/difficulty.py`、三动作策略、局部图替换和负载回放模块；这些文件不得在第一轮创建或实现。

### 8.2 统一样本接口

所有数据集 adapter 输出同一个 `TaskGraphSample`：

```text
sample_id
task_text
context
gold_graph        # 第一轮必须存在；可由官方依赖直接读取或由官方 schema 确定性派生
gold_final_answer # 允许为空
metadata
```

`gold_graph` 内每个节点必须有：

```text
id / objective / depends_on / input_refs / output_schema
gold_output（若数据集提供）/ supporting_context（若提供）
```

第一轮的图来源必须写入 `metadata.graph_source`：MuSiQue 根据官方子问题中的前驱引用确定性构图，FanOutQA 直接读取官方 `depends_on`，PARALLELPROMPT 根据官方 reference schema 构造 `START → parallel branches → AGGREGATE` 扁平图，并记录 `validation_tier`。统一接口允许图为空只用于后续数据集，第一轮需要图的样本必须 100% 构图并通过校验。

---

## 九、第一轮：独立的无训练可行性实验

第一轮不是后续完整框架的“工程准备阶段”，而是一项可以独立运行、独立验收、独立得出结论的实验。它只回答三个最基础的问题：

1. 同一个复杂任务内部，是否同时存在小模型可以完成和必须交给大模型的节点？
2. 不训练拆解器时，现成大模型能否零样本生成有意义的子任务 DAG？
3. 给定参考拆解后，其中的并行分支能否在质量基本不变时缩短端到端时延？

第一轮结束时必须形成完整报告，并根据结果决定是否进入第二轮；不能把“以后训练后可能有效”当作第一轮结论。

### 9.1 第一轮边界

第一轮只使用预训练模型的推理接口、数据集 Gold 标注、确定性规则和后验 Oracle 上界，不进行任何模型训练、微调、蒸馏或强化学习。

本轮明确不做：

- 不训练拆解器或节点难度预测器；
- 不实现动态递归拆解和 `DECOMPOSE / SMALL / LARGE` 三动作控制器；
- 不加入 MEQA、TaskCraft、WorFBench、MINTQA 或 MoNaCo；
- 不搭建在线服务集群，不研究模型池、容错、恢复和 GPU 部署；
- 不以第一轮结果宣称已经得到可上线的路由策略。

### 9.2 第一轮数据和固定规模

| 数据集         |      冒烟/校准集 |                                                       锁定的正式评测集 | 本轮唯一用途                             |
| -------------- | ---------------: | ---------------------------------------------------------------------: | ---------------------------------------- |
| MuSiQue-dev    |      50 个根问题 |               与校准集互斥的 300 个根问题，按 hop 数和组合类型分层抽样 | Gold 节点能力测试和路由收益上界          |
| FanOutQA-dev   | 固定 20 个根问题 |                                               其余 290 个公开 dev 问题 | 零样本 DAG 拆解质量与端到端质量          |
| PARALLELPROMPT |  固定 100 个请求 | 与校准集互斥，按 source 与 validation_tier 去重、分层抽取 1,000 个请求 | Reference schema 串行/并行质量—时延比较 |

抽样以根问题为单位，不能随机抽散节点。冒烟/校准 ID 与正式 ID 必须永久互斥，分别写入 manifest；先固定两份样本清单，再运行模型。提示、语义匹配阈值和质量容忍度可以在校准集上确定，但校准样本永不计入主结果、置信区间或显著性检验。

### 9.3 工作包 A：独立工程与数据审计

1. 新建 `adaptive-subtask-routing` 独立 Python 3.11 项目，使用 `src/` layout；
2. 实现 `TaskNode`、`TaskGraph`、`TaskGraphSample`、DAG 校验和拓扑排序；
3. 只实现 MuSiQue、FanOutQA 和 PARALLELPROMPT 三个 adapter；
4. 为原始数据生成来源、commit、许可、split、SHA-256 和样本统计；
5. 建立统一模型接口，配置两个任务执行端点 `small_model` 与 `large_model`，以及一个只用于语义评测的 `evaluator_model`；评测模型不得参与节点执行或路由，调用与费用单列；
6. 建立响应缓存、断点续跑、调用计数、Token、费用和分段时延日志。

验收条件：所有第一轮样本 ID 唯一；需要图的样本 100% 完成官方/确定性派生构图，`graph_source` 可追溯，图无环且边端点和引用存在；`pytest` 和不调用模型的 dry-run 全部通过。

### 9.4 工作包 B：MuSiQue Gold 节点能力实验

对每个 Gold 节点只提供：该节点所需的 Gold 支持上下文、Gold 前驱输出和统一的回答格式。不要把后续节点、最终答案或其他 Gold 子答案放进提示。

大小模型各独立运行至少 3 次，按稳定正确性生成：

- `EITHER`：大小模型都稳定正确；
- `SMALL_ONLY`：只有小模型稳定正确，单独分析异常原因；
- `LARGE_NEEDED`：小模型失败而大模型稳定正确；
- `UNSOLVED`：两者都未稳定答对。

除全局节点比例外，必须报告根问题级 `mixed_root_rate`：同一 root 内至少包含一个 `EITHER/SMALL_ONLY` 节点且至少包含一个 `LARGE_NEEDED` 节点的比例，并给出 root-level bootstrap 置信区间。否则只能证明“数据集中存在两类节点”，不能证明“同一复杂任务内部存在能力异质性”。

三种只作分析的策略为 `all_small`、`all_large` 和使用真实标签的后验 `oracle_route`。Oracle 规则预先固定为：`EITHER/SMALL_ONLY → small`，`LARGE_NEEDED → large`，`UNSOLVED → 失败并单列`。Oracle 成本是在 Gold-input 隔离条件下，对每个节点只计所选模型一次调用再求和的后验成本下界，不是真实误差传播 rollout；用于产生标签的 3×离线 profiling 成本必须另行报告。`oracle_route` 不能冒充可部署方法。

本工作包的结论是“节点能力是否异质、理论路由空间有多大”，不是“已经学会了路由”。

### 9.5 工作包 C：FanOutQA 零样本拆解实验

为每个根问题先构造一个固定的**根问题级证据池**：取官方必要证据的并集，去掉其与具体 Gold 节点的映射、节点答案和依赖，并按固定种子打乱。`direct`、预测图和 Gold 图执行都只能使用同一个证据池，从而排除检索差异；预测/Gold 节点执行时只额外接收各自前驱输出。第一轮不评测检索能力。

拆解模型只能看到原始复杂问题和上述公共证据池，不能看到 Gold decomposition、`depends_on`、节点答案或最终答案。要求一次输出结构化 JSON DAG；非法图最多进行一次只修格式、不补语义的修复。

至少比较：

- `direct_large`：大模型直接回答整题；
- `predicted_graph_all_large`：零样本预测图，所有节点均由大模型执行；
- `gold_graph_all_large`：Gold 图由大模型逐节点执行；
- `gold_graph_gold_outputs`：直接把 Gold 节点输出送入同一个冻结合成器，只测合成上界。

拆解指标包括有效 DAG 比例、节点数偏差、语义节点 Precision/Recall/F1、完成节点语义匹配后的 Edge-F1、问题覆盖率、过拆率和漏拆率；回答指标使用数据集官方最终答案指标。固定语义匹配器和阈值，并人工复核一个固定小样本，避免评测器本身漂移。

误差归因固定为：预测图对比 Gold 图主要观察结构损失；`gold_graph_all_large` 对比 `gold_graph_gold_outputs` 观察节点执行损失；`gold_graph_gold_outputs` 的剩余错误归入合成器。合成模板、模型和生成设置必须冻结，不能为不同变体单独调优。

### 9.6 工作包 D：PARALLELPROMPT 串并行实验

PARALLELPROMPT 官方发布的是 train split，因此本课题按 `source/source_index` 去重后冻结成自建 evaluation subset，不称其为官方测试集。第一轮使用官方 reference schema，把每个请求确定性转换为 `START → parallel branches → AGGREGATE` 扁平参考图；该 schema 是模型抽取并经规则/分层验证的参考结构，不是人工 Gold。正式结果按 `validation_tier` 分层报告；在校准集上人工核查分支独立性和模板语义等价，并把筛选规则冻结后再用于正式集。它只测参考结构下的系统收益，不测预测拆解质量或通用 DAG 调度。

使用同一个模型、完全相同的节点提示、聚合器和生成设置，只改变执行组织方式：

1. `whole_request`：原请求一次执行；
2. `reference_schema_serial`：reference schema 中的独立分支逐个执行；
3. `reference_schema_parallel`：同一组分支并发执行，最后使用相同聚合器汇合。

该数据没有逐节点 Gold 答案，因此只报告冻结的官方评测套件和固定 `evaluator_model` 给出的 semantic fidelity，不称为 accuracy；评测模型、提示、调用和费用全部单列。分别报告纯节点执行的 speedup，以及包含模板展开、上下文复制和聚合的端到端 speedup，同时给出失败率、总 Token、调用次数和 p50/p95。

串行与并行必须按同一请求做 paired repetitions，记录限流、重试和队列等待。聚合器必须按固定的 schema `branch_id` 顺序接收结果，不能按分支完成顺序拼接。质量比较复用同一组配对输出；时延实验不得命中响应缓存，并需交替运行两种调度顺序以减小服务抖动偏差。第一轮不加入预测 schema 或负载感知策略。

### 9.7 第一轮统一运行入口与产物

建议只暴露以下入口：

```bash
python -m splitroute.cli first-round data-audit --config configs/first_round.yaml
python -m splitroute.cli first-round smoke --config configs/first_round.yaml
python -m splitroute.cli first-round run --config configs/first_round.yaml
python -m splitroute.cli first-round report --config configs/first_round.yaml
```

所有第一轮产物只能写入：

```text
outputs/first_round/
├── manifests/
├── raw_responses/
├── node_capability.jsonl
├── node_capability_summary.csv
├── decomposition_metrics.json
├── parallel_metrics.json
├── figures/
└── first_round_report.md
```

第一轮不应生成训练 checkpoint。`first_round_report.md` 必须能由缓存的原始响应和 manifest 一键重建。

### 9.8 第一轮结束条件

只有完成正式样本、重复运行、数据泄漏检查和报告重建后，第一轮才算结束。报告必须对三个前提分别给出“支持、不支持或证据不足”的结论：

- MuSiQue 的 `mixed_root_rate` 是否达到预注册阈值，且 root-level bootstrap 置信区间下界仍超过该阈值；全局标签比例只作描述；
- FanOutQA 零样本预测图是否达到预注册的节点/边/合法率门槛，且预测图执行的最终质量未超出预设容忍范围；
- PARALLELPROMPT 并行执行是否在预先规定的质量容忍范围内降低**端到端 p95 时延**；execution-only p95 只作诊断。

具体统计阈值应在冒烟测试之后、正式实验之前写入配置和报告模板，不根据正式结果事后修改。任一核心前提不成立时，先分析数据、提示或假设，不直接进入训练阶段掩盖问题。

MuSiQue 和 FanOutQA 是公开基准，闭源预训练模型可能见过相关内容，因此第一轮只提供可行性证据，不宣称无预训练污染的跨分布泛化。所有第一轮正式 ID 永久冻结，后续不得用于训练、提示选择或阈值选择；最终泛化结论由方案冻结后的严格留出集承担。

---

## 十、第二轮及以后：训练与完整框架

本节不是第一轮的实现内容。只有第一轮报告完成并支持继续研究时，才按顺序进入。

### 第二轮：学习节点路由

- 加入 MEQA，扩大事件、时间和比较型节点；
- 冻结第一轮方案后，另在 MuSiQue-train 和 MEQA-train 上生成节点执行日志，训练并校准 `p_small_success`；
- 第一轮 MuSiQue-dev 正式 ID 永不用于训练、提示选择或阈值选择，后续仍保持为冻结评测证据；
- 比较文本特征、图结构特征和二者结合；
- 报告 `LARGE_NEEDED` 召回率、AUROC、F1、ECE 和质量—成本曲线；
- 使用训练集训练、开发集选阈值，测试集只做最终一次评测。

### 第三轮：动态拆解粒度

- 加入 TaskCraft 控制深度和宽度；
- 加入 WorFBench 评测未知任务上的节点、链和 DAG；
- 从一次性粗图扩展为对指定节点执行局部 `DECOMPOSE(node)`；
- 比较不拆、固定粗粒度、固定细粒度和动态粒度。

### 第四轮：负载感知的联合决策

- 实现 `DECOMPOSE / EXECUTE_SMALL / EXECUTE_LARGE` 三动作控制器；
- 加入大小模型队列、估计服务时间和关键路径；
- 构造低、中、高负载回放；
- 报告质量—成本、质量—p95 时延、吞吐和 SLO 违约率。

### 第五轮：严格泛化

- 主结果和阈值全部冻结后，再选择 MINTQA 扩大链式复杂度测试；
- MoNaCo 只作最终严格留出，不训练、不调参、不再次分发原始数据；
- 生成论文表格、消融、Pareto 曲线和按错误来源划分的案例分析。

---

## 十一、实现时的硬约束

1. 第一轮和后续轮次使用不同配置、命令入口和输出目录，禁止混写日志或缓存。
2. 冒烟/校准与正式 ID 永久互斥；第一轮正式 ID 永不进入后续训练、提示选择或阈值选择。
3. 不把 Gold 子问题、子答案或依赖泄漏到端到端提示；Oracle 实验使用独立配置并明确标注上界。
4. 模型响应必须缓存，避免重复 API 花费；缓存键不得包含 API 密钥。
5. 每次运行记录数据版本、模型名/版本、提示版本、温度、Token 上限、随机种子、调用费用和分段时延。
6. 原始数据只读；转换脚本可重复运行并生成 manifest 和校验和。
7. 拆解、排队、节点执行和合成开销分别统计，不能只报总 Token 或总时延。
8. 图校验至少检查唯一 ID、边端点、无环、输入引用和最终 sink。
9. 不使用主观的“规划、计算、检查”等阶段标签作为真实节点难度标签。
10. 第一轮禁止训练、微调、RL、模型池、故障恢复、在线验证或 GPU 部署优化。
11. 后续先实现可复现的规则策略，再训练复杂控制器。
12. 每轮必须独立可审查；上一轮未完成验收时，不实现下一轮功能。
13. 若模型调用会产生费用，先输出模型、调用次数、预计 Token 和费用，不在未授权时自动进行正式批量调用。

---

## 十二、可以直接交给 WSL Codex 的第一轮提示词

```text
请新建并只在 /mnt/e/HiNA/HiNA/adaptive-subtask-routing 中工作。
这是一个独立研究原型，不读取、导入或修改 llm-reasoning-agent-eval、contask 或其他项目。

先完整阅读：
/mnt/e/HiNA/HiNA/大小模型路由papers/复杂任务子任务拆解与大小模型路由_WSL实现指导.md

本次只执行文档第九章“第一轮：独立的无训练可行性实验”。不要实现第十章及以后任何训练或完整框架功能。

第一轮范围：
1. 创建采用 src layout 的 Python 3.11 项目，包名 splitroute；建立 pyproject.toml、README、.gitignore、configs、tests、data 和 outputs；
2. 实现不调用模型的 healthcheck、TaskNode、TaskGraph、TaskGraphSample、DAG 校验和拓扑排序；
3. 只接入 MuSiQue-dev、FanOutQA-dev 和 PARALLELPROMPT；PARALLELPROMPT 从官方 train 发布中冻结自建 evaluation subset。固定官方来源 commit/版本，原始数据只读，生成互斥的校准/正式 ID manifest、SHA-256 和样本审计报告；
4. 建立 OpenAI-compatible 模型接口，本地配置指定 small_model、large_model 和只判分的 evaluator_model；evaluator_model 不得执行节点或参与路由，其调用与费用单列。配置必须被 gitignore，代码中不得硬编码密钥或具体供应商；
5. 实现响应缓存、断点续跑、重试、调用计数、Token、费用以及拆解/排队/执行/合成分段时延日志；
6. 实现 MuSiQue Gold 节点能力实验：严格控制可见上下文和 Gold 前驱输出，大小模型各重复至少 3 次，生成 EITHER、SMALL_ONLY、LARGE_NEEDED、UNSOLVED；报告 root-level mixed_root_rate 及 bootstrap CI、all_small、all_large、后验 oracle_route 上界和单列的 profiling 成本；
7. 实现 FanOutQA 零样本拆解实验：所有变体使用同一个去节点映射的根问题级 Gold evidence 并集；提示中禁止出现 Gold decomposition、depends_on、节点答案和最终答案；输出 JSON DAG，比较 direct_large、predicted_graph_all_large、gold_graph_all_large、gold_graph_gold_outputs；固定同一合成器，计算图、节点、边、过拆/漏拆、节点执行和最终答案指标；
8. 实现 PARALLELPROMPT 的 whole_request、reference_schema_serial、reference_schema_parallel 三种执行方式；使用官方 reference schema，不预测拆解。按 validation_tier 分层并在校准集人工核查分支独立性/模板等价；保持节点提示、模型、聚合器和生成设置相同，按固定 branch_id 聚合并做配对重复，分别报告 semantic fidelity、execution-only 与 end-to-end 的失败率、Token、调用数、p50/p95 和加速比；
9. 提供 first-round data-audit、smoke、run、report 四个 CLI 入口，为 schema、三个 adapter、FanOutQA 证据协议、模型角色隔离、缓存、DAG executor、固定合成器、并行调度和指标写单元测试；
10. 所有第一轮产物只写入 outputs/first_round，并能从缓存和 manifest 一键重建 first_round_report.md；不要生成任何训练 checkpoint；
11. 禁止加入 MEQA、TaskCraft、WorFBench、MINTQA、MoNaCo，禁止训练/微调/RL，禁止实现动态递归拆解、三动作控制器、负载感知策略、模型池、容错和 GPU 部署；
12. 先完成无需模型调用的工程、测试、data-audit 和 dry-run。若数据未下载，列出官方命令、预计大小和许可后再下载；若模型调用产生费用，先汇报模型、预计调用次数、Token 和费用。没有明确授权时只完成 dry-run，不伪造实验结果；
13. 获得调用授权后，先按文档固定规模运行 smoke。smoke 验收通过后再运行正式样本，最后生成 first_round_report.md；
14. 完成后只汇报实际改动、测试证据、数据审计、实际调用规模、第一轮三个问题的结论和未解决问题，不建议或顺带实现第二轮。
```

这段提示词对应一个完整但封闭的第一轮。第一轮报告验收后，应开启新的 WSL Codex 任务并单独下达第二轮要求。

---

## 十三、最后的研究路线

```text
第一轮（无训练）：MuSiQue 节点能力 + FanOutQA 零样本拆解 + PARALLELPROMPT 串并行
                 └─ 结论门：能力是否异质、图是否可用、并行是否有系统收益

第二轮（学习路由）：加入 MEQA，训练并校准节点小模型成功率

第三轮（动态拆解）：加入 TaskCraft/WorFBench，学习何时继续拆、拆到多细

第四轮（联合系统）：加入服务负载与关键路径，联合选择继续拆/小模型/大模型

第五轮（严格泛化）：冻结设计后在 MINTQA 或 MoNaCo 上做留出验证
```

最优先得到的不是一个功能齐全的系统，而是一条逐轮可证伪的学术证据链：

> 第一轮先证明复杂任务内部存在节点级能力异质性、零样本拆解具备可用结构、并行执行能产生系统收益；后续才研究如何学习节点路由、动态控制拆解粒度，并在服务负载下优化质量—成本—时延权衡。
