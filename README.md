# 基于 LangGraph ReAct 架构的自主推理医疗诊断 Agent

> 原项目为"基于RAG与大模型技术的医疗问答系统"，现已全面升级为基于 LangGraph 的自主推理 Agent。

## 项目简介

本项目构建了一个**能自主推理、自主调用工具、自主校验的医疗诊断 Agent**。用户输入模糊症状（如"头痛发烧"），Agent 会自主完成以下流程：

```
用户输入 → NER 实体抽取 → LLM 推理决策 → Ne4j 知识图谱查询 → 安全校验 → 返回回答
              ↑                              ↑                  ↑
         三合一混合NER               ReAct 循环（思考⇄行动）   双Agent协同
```

核心思路是**用知识图谱约束大模型输出**——LLM 不靠"记忆"回答医学问题，而是先查知识图谱，基于真实数据作答，从源头降低幻觉风险。

## 技术架构

| 层级 | 技术栈 | 核心功能 |
|------|--------|---------|
| 前端 | Streamlit + FastAPI WebSocket | 多窗口问诊、推理步骤实时可视化、生产级流式API |
| Agent核心 | LangGraph StateGraph | ReAct 循环 + 四节点（感知/规划/执行/反思） |
| 工具层 | LangChain @tool × 4 | 实体校验、症状反推、属性查询、关联检索 |
| 记忆层 | FAISS + BGE + JSON | 四级记忆（感知/工作/向量/持久化） |
| 数据层 | Neo4j 知识图谱 | 8类实体、11类关系、约4.4万节点、31万边 |

### Agent 工作流

```
START → preprocess(ONNX NER 三合一) → llm_planner(诊断Agent, qwen-max)
           ↕ (ReAct循环)                    ↓
      tool_executor(Neo4j, 并行执行) ← 有 tool_calls?
                                              ↓ 否
                                        reflection(校验Agent, qwen-plus)
                                        规则护栏 + LLM事实校验
                                              ↓
                                            END
```

## 快速开始

### 环境要求

- Python 3.10+
- Neo4j 5.x（已安装并运行，默认 `bolt://localhost:7687`）
- （可选）Ollama 本地模型

### 安装

```bash
git clone https://github.com/zhengwenyi07-cmyk/medical-agent.git
cd medical-agent
pip install langgraph langchain-core langchain-openai streamlit neo4j onnxruntime transformers py2neo faiss-cpu sentence-transformers fastapi uvicorn httpx
```

### 配置

编辑 `.streamlit/secrets.toml`：

```toml
[neo4j]
uri = "bolt://localhost:7687"
user = "neo4j"
password = "你的密码"

[qwen]
api_key = "你的阿里云百炼API Key"
model = "qwen-max"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[ollama]
base_url = "http://localhost:11434"
```

### 构建知识图谱

```bash
python build_up_graph.py --website http://localhost:7474 --user neo4j --password 你的密码 --dbname neo4j
```

### 启动

```bash
# Streamlit 前端（带登录）
streamlit run login3.py --server.port 8501

# 或 FastAPI 生产服务
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 测试

```bash
# 工具链沙盒测试（12项，无需LLM）
python test_tools.py

# CLI 测试 Agent
python -c "from agent_stream import stream_agent; [print(e) for e in stream_agent('头痛吃什么药')]"
```

## 核心特性

### 双Agent协同校验

诊断 Agent(qwen-max) 生成回答后，由独立的校验 Agent(qwen-plus) 进行事实审核——不是同一个模型自己审自己。审核发现问题时标注风险横幅而非直接拦截。

### 三合一混合 NER

RoBERTa-RNN 深度语义 + AC 自动机词典匹配 + TF-IDF/BGE Hybrid 实体对齐。并行运行、结果合并去重。ONNX 导出后纯 CPU 推理提速约 2 倍。

### 四级记忆体系

| 层级 | 实现 | 生命周期 |
|------|------|---------|
| 感知记忆 | raw_entities | 单次请求 |
| 工作记忆 | knowledge_cache | 单次工具调用间 |
| 向量记忆 | FAISS + BGE (512维) | 跨会话持久化 |
| 持久记忆 | JSON 磁盘 | 跨重启 |

### 三层容错 + 八级降级

- 工具层：指数退避重试 + JSON Schema 校验 + 静态字典降级
- LLM层：qwen-max → qwen-plus → qwen3-vl 系列 → Ollama（8 级）

### 高并发工程

asyncio 原生协程 + 多进程 Worker 池(CPU×2) + Neo4j 微服务拆分 + 工具调用并行化（5×加速比）

## 项目文件

| 文件 | 作用 |
|------|------|
| `agent_graph.py` | AgentState + 四节点 + 图装配 + 降级链 + 护栏 |
| `agent_stream.py` | Streamlit 流式桥接 + 摘要压缩 |
| `agent_async.py` | asyncio 协程替代线程 |
| `tools.py` | 4 个 LangChain @tool |
| `neo4j_client.py` | Neo4j 连接池单例 |
| `ner_model.py` | NER 模型（RoBERTa+AC自动机+TF-IDF） |
| `vector_memory.py` | FAISS + BGE 向量记忆 |
| `conversation_storage.py` | 对话持久化 |
| `server.py` | FastAPI + WebSocket 生产服务 |
| `worker_pool.py` | 多进程 Worker 池 |
| `tool_service.py` | Neo4j HTTP 微服务 |
| `webui3.py` | Streamlit 前端 |
| `login3.py` | 登录/注册 |
| `config.py` | 配置读取 |
| `test_tools.py` | 工具链单元测试 |

## 数据集

本项目使用 [Open-KG](http://data.openkg.cn/dataset/disease-information) 医疗数据集，约 4.4 万实体节点、31 万关系边。

## 联系方式

邮箱：25210980145@m.fudan.edu.cn
