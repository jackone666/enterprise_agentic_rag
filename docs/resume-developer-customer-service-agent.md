# 开发者智能客服 Agent 平台 - 简历包装

## 推荐项目名称

开发者智能客服 Agent 平台

英文名可写：Developer Intelligent Customer Service Agent Platform

## 一句话介绍

面向开发者官网智能客服入口，设计并实现基于 LangGraph 的多智能体 RAG 问答系统，支持文档问答、错误诊断、代码示例生成、工单工具调用、答案可信校验和人工兜底升级。

## 简历版本

### 精简版

**开发者智能客服 Agent 平台**  
技术栈：Python、FastAPI、LangGraph、LangChain、Milvus、Elasticsearch、Neo4j、Redis、PostgreSQL、OpenTelemetry、Prometheus、React

- 设计多智能体客服工作流，基于 MasterAgent 统一调度意图识别、工具调用、知识检索、代码生成、答案校验和人工兜底节点，覆盖开发者文档咨询、错误诊断、API 示例生成等场景。
- 构建意图感知 RAG 检索链路，按问题类型自动选择 HybridRAG、GraphFirst、ErrorFirst、CodeGeneration 等检索策略，融合向量检索、BM25、知识图谱、外部知识源和 Cross-Encoder 精排。
- 实现 Claim-level 答案校验和冲突证据检测，将回答拆分为原子断言并对照来源文档验证，降低开发者客服场景中的幻觉和过期 API 建议风险。
- 开发智能客服前端 Widget（React + Tailwind），支持全页模式和可嵌入浮动挂件，实现深度思考（CoT Visible）、推荐问题、SSE 流式对话、反馈采集等端到端交互。
- 接入 SSE 流式输出（含 thinking/answer_chunk 增量推送）、Trace 事件、反馈采集和 Eval Gate，支持客服回答链路可观测、badcase 沉淀和回归评估闭环。

### 面试突出版

**开发者智能客服 Agent 平台**  
项目背景：参考开发者官网右下角智能客服形态，面向“文档查找成本高、问题类型复杂、客服回答需要可信来源”的开发者支持场景，搭建一套可追踪、可评估、可兜底的 Agentic RAG 客服系统。

- 工作流编排：基于 LangGraph 搭建 `load_memory -> permission -> deep_intent -> master_agent -> retrieve/tool/code/verify -> fallback` 的状态机流程，MasterAgent 根据意图置信度、工具需求、检索结果和校验状态选择下一步动作。
- 检索增强：设计意图感知检索路由，对概念问答走 HybridRAG，对迁移/兼容问题走 GraphFirst，对错误码排障走 ErrorFirst，对代码示例走 CodeGenerationWorkflow，并用 RRF + Cross-Encoder 进行证据融合与精排。
- 可信回答：实现引用管理、冲突检测、断言级校验和 Self-RAG 质量门控，回答返回来源文档、校验原因、工具结果和 trace_id，便于客服质检与问题复盘。
- 工程闭环：提供 FastAPI `/chat` 与 `/chat/stream` 接口，前端展示聊天、RAG 证据、工具调用、Trace 面板和反馈按钮；通过 OpenTelemetry、Prometheus、JSONL 日志和 Eval Gate 形成线上质量优化闭环。

## 对标”官网智能客服”的功能映射

| 官网智能客服能力 | 本项目对应模块 |
| --- | --- |
| 用户在右下角客服入口提问 | FastAPI `/chat` + `/chat/stream`、Widget 浮动挂件 `FloatingWidget` |
| 欢迎页推荐问题点击 | `/api/suggestions` + `WelcomeScreen` |
| 深度思考（展示 AI 推理过程） | `DeepThinking` + `_generate_thinking_trace` + SSE `thinking` 事件 |
| 识别用户要问文档、报错、代码还是工单 | Deep Intent + MasterAgent |
| 查询官方文档、FAQ、API 示例 | HybridRAG / CodeGenerationWorkflow |
| 排查错误码、构建失败、接口异常 | ErrorFirstWorkflow + ToolAgent |
| 回答附带来源和依据 | CitationList + retrieved_docs + evidence selector |
| 复制/点赞/点踩反馈 | `FeedbackButtons` + `/feedback` API |
| 避免胡说和过期建议 | ClaimVerifier + conflict detector |
| 高风险/低置信度转人工 | human_fallback + permission / recovery policy |
| 客服质量监控和持续优化 | Trace、Metrics、Feedback、Eval Gate |

## 面试讲法

可以这样讲：

> 这个项目我不是做一个普通聊天机器人，而是对标华为开发者官网右下角智能客服，把它抽象成开发者智能客服平台。开发者问题有几个特点：问题类型复杂、答案必须有官方文档依据、很多问题需要代码示例或错误诊断，而且低置信度时不能强答。所以我用 LangGraph 做了一个多 Agent 状态机，先做深度意图识别，再由 MasterAgent 决定走文档检索、工具查询、代码生成、答案校验还是人工兜底。检索层不是单一路径，而是按意图选择 HybridRAG、GraphFirst、ErrorFirst 或 CodeGenerationWorkflow。前端做了两套：开发者控制台（含 Trace/证据/工具面板）和终端用户智能客服 Widget（含深度思考、推荐问题、浮动挂件），通过 SSE 流式推送 thinking 和 answer_chunk 增量。最后通过 citation、claim-level verifier、trace 和 feedback 把客服回答做成可追踪、可评估、可迭代的闭环。

## 高频追问防守

### 为什么不用普通 RAG？

普通 RAG 更适合单轮文档问答，但开发者客服的问题会混合概念解释、错误诊断、代码示例、迁移路径、账号/工单查询等多种任务。项目里用 Deep Intent + MasterAgent 先判断问题类型，再选择不同检索和工具路径，避免所有问题都走同一套 top-k 检索。

### 怎么减少幻觉？

主要做了四层约束：检索证据必须进入上下文；答案生成带 citation；ClaimVerifier 将答案拆成原子断言逐条对照文档；冲突检测会标注版本差异、否定关系、API 废弃和建议冲突。校验失败时进入重试、降级或人工兜底。

### 为什么需要知识图谱？

开发者问题经常涉及 API 依赖、版本迁移、组件关系和错误传播路径。向量检索能找相似文本，但不擅长表达“模块 A 依赖 B”“API X 在版本 Y 废弃”“错误 E 与配置 C 相关”这类关系，所以迁移和兼容类问题走 GraphFirstWorkflow。

### 如何评估客服回答质量？

项目里有 Eval Gate 和反馈闭环。离线用评估集检查路由准确率、召回质量、答案正确性和安全合规；线上通过 thumbs up/down、trace_id、retrieved_docs、node_events 自动沉淀 badcase，再回灌到评估集和 prompt / retrieval 策略。

## 注意事项

- 简历中不要写成“华为官方项目”或“接入华为内部系统”，除非确实有授权和真实数据。
- 推荐表述为“参考开发者官网智能客服形态”“面向开发者门户客服场景”“可承接类似官网智能客服入口的技术支持问答”。
- 面试时重点讲工程设计和可验证能力，不要只堆 LangGraph、RAG、Agent 这些名词。
