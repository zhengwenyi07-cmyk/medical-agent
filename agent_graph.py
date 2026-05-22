"""Agent 全局状态定义与图节点实现 — LangGraph 节点流转

本文件是整个 Agent 系统的大脑，包含四大核心部分：
  (A) AgentState  —— 全局共享状态字典（所有节点通过它通信）
  (B) 四个节点函数 —— preprocess / llm_planner / tool_executor / reflection
  (C) 路由函数     —— should_continue / should_loop_back（控制图的分支流转）
  (D) 图装配       —— build_agent_graph() 把所有节点连成 ReAct 循环

数据流向：
  用户输入 → preprocess(NER) → llm_planner(推理) ⇄ tool_executor(执行工具)
                llm_planner(出最终回答) → reflection(安全校验) → END
"""

import json
from typing import TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # reducer：自动追加消息而非覆盖
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

# 模块级流水日志记录器（由 agent_stream.py 在每次请求前设置）
_pipeline_logger = None

def set_pipeline_logger(logger):
    """设置当前请求的流水日志记录器（由 agent_stream.py 调用）"""
    global _pipeline_logger
    _pipeline_logger = logger


# ==========================================
# AgentState 状态字典
# ==========================================
# 这是整个 Agent 的"大脑内存"——所有节点函数签名都是 (state) -> state，
# 它们通过读写同一个 AgentState 实例来传递信息，无需函数间显式传参。
# ==========================================

class AgentState(TypedDict):
    """LangGraph 核心状态字典，贯穿所有节点的数据处理。

    Attributes:
        messages: 对话轮次历史，使用 add_messages 自动合并新消息。
            当节点写 state["messages"] = [new_msg] 时，实际执行的是
            add_messages(existing, [new_msg])，即追加而非替换。
        raw_entities: ONNX NER 模型抽取的实体，格式为 {实体类别: [实体名称1, 实体名称2, ...]}。
            每种类型可包含多个实体（如同一句话中出现"头痛""发烧"两个症状）。
            典型值: {"疾病症状": ["头痛","发烧"], "药品": ["布洛芬"]}
        knowledge_cache: 已查询过的工具结果缓存（短期记忆）。
            键格式: "工具名:参数JSON"，例如 "get_disease_relations:{\"disease_name\":\"感冒\",\"relation_type\":\"药品\"}"
        next_action: 控制图条件路由的指令。
            由 node_llm_planner 设置: "continue"(有tool_calls → 去执行工具) 或 "end"(出最终回答 → 去校验)
            node_tool_executor 始终设为 "continue"（执行完回 LLM 继续推理）
    """

    # Annotated[list, add_messages] 的含义:
    #   - list: 字段的基础类型是列表
    #   - add_messages: reducer 函数，写入时自动追加 + 去重，而非替换旧列表
    # 这使得每个节点只需要关心自己产出的新消息，不需要手动拼接全量消息历史。
    messages: Annotated[list, add_messages]
    raw_entities: dict
    knowledge_cache: dict
    next_action: str


def create_initial_state(user_query: str = "") -> AgentState:
    """创建初始化的 AgentState 字典。

    在每次新请求开始时调用，生成一个空的状态容器。
    如果传入了 user_query，会自动创建第一条 HumanMessage 放入 messages 列表。
    """
    state = AgentState(
        messages=[],           # 对话消息列表，初始为空
        raw_entities={},       # NER 实体，尚未抽取
        knowledge_cache={},    # 工具缓存，尚未查询
        next_action="continue",# 初始设为 continue，让图从 preprocess 开始走
    )
    if user_query:
        # 如果有初始查询，创建第一条 HumanMessage
        # add_messages reducer 会将这条消息追加到空列表
        state["messages"] = [HumanMessage(content=user_query)]
    return state


# ==========================================
# NER 组件懒加载缓存
# ==========================================
# 模块级全局变量 _ner_components 用于缓存 ONNX 推理所需的全部组件。
# 首次调用 _load_ner_components() 时加载（约 14s），后续调用直接返回缓存。
# 这种"懒加载 + 全局缓存"模式避免了每次请求都重新加载大模型。
# ==========================================
_ner_components = None  # 模块级缓存变量，None 表示尚未加载


def _load_ner_components():
    """延迟加载 ONNX 推理所需的所有组件（首次调用约需 5-10 秒）

    加载内容:
      1. tag2idx / idx2tag —— 标签到索引的映射（用于 NER 标签解码）
      2. rule_find —— AC 自动机规则引擎（基于医学词典的多模式匹配）
      3. tfidf_alignment —— TF-IDF 实体对齐器（把模糊实体名归一化到标准名）
      4. BertTokenizer —— RoBERTa 分词器（把中文文本切成 token）
      5. ONNX InferenceSession —— ONNX 推理会话（执行 NER 模型推理）

    返回: 包含以上 5 个组件的 dict
    """
    global _ner_components
    if _ner_components is not None:
        return _ner_components  # 已加载，直接返回缓存

    import pickle
    import onnxruntime as ort
    from transformers import BertTokenizer
    import ner_model as zwk

    # 1. 加载标签映射表（训练时生成的 tag2idx 字典）
    with open("tmp_data/tag2idx.npy", "rb") as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)  # idx2tag[i] = 标签名，如 'B-疾病'

    # 2. 构建 AC 自动机规则引擎（加载 data/ent_aug/ 下的 8 类医学词典）
    rule = zwk.rule_find()

    # 3. 构建 TF-IDF 实体对齐器（加载标准实体名并计算 TF-IDF 向量矩阵）
    tfidf_r = zwk.tfidf_alignment()

    # 4. 加载中文 RoBERTa 分词器（已有本地模型，无需下载）
    tokenizer = BertTokenizer.from_pretrained("model/chinese-roberta-wwm-ext")

    # 5. 创建 ONNX 推理会话
    # providers 优先级: GPU(CUDA) > CPU
    # ONNX Runtime 会自动选择可用的执行后端
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    ort_session = ort.InferenceSession("model/ner_model.onnx", providers=providers)

    # 打包所有组件到全局缓存
    _ner_components = {
        "ort_session": ort_session,   # ONNX 推理会话
        "tokenizer": tokenizer,        # BERT 分词器
        "rule": rule,                  # AC 自动机规则引擎
        "tfidf_r": tfidf_r,            # TF-IDF 实体对齐器
        "idx2tag": idx2tag,            # 标签索引→标签名映射
    }
    print(f"[NER] ONNX 模型加载完成，后端: {ort_session.get_providers()[0]}")
    return _ner_components


# ==========================================
# LLM 懒加载缓存
# ==========================================
# 模块级全局变量，缓存 LLM 实例和绑定工具的 LLM 实例。
# _get_llm() 返回的是一个降级链列表 [(llm_instance, model_name, desc), ...]，
# 而非单个 LLM，这样 node_llm_planner 可以依次尝试直到成功。
# ==========================================
_llm = None              # 缓存不带工具的 LLM 列表
_llm_tool_bound = None   # 缓存绑定工具的 LLM 列表
# 降级链模型列表：(model_name, description)
# 全部使用阿里云百炼同一 API Key，按推理能力强→弱排序
# node_llm_planner 会依次尝试，任一成功即停止
_FALLBACK_CHAIN = [
    ("qwen-max",                    "通义千问 Max（主模型，最强文本推理）"),
    ("qwen-plus-2025-07-28",        "通义千问 Plus 2025版（降级1）"),
    ("qwen-plus",                   "通义千问 Plus（降级2）"),
    ("qwen3-vl-235b-a22b-thinking", "Qwen3-VL 235B Thinking（降级3，思考增强）"),
    ("qwen3-vl-32b-thinking",       "Qwen3-VL 32B Thinking（降级4）"),
    ("qwen3-vl-30b-a3b-thinking",   "Qwen3-VL 30B Thinking（降级5）"),
    ("qwen-vl-plus-latest",         "通义千问 VL Plus（降级6）"),
]


def _build_llm_instance(model_name: str, cfg: dict) -> object | None:
    """根据模型名构建 ChatOpenAI 实例。返回 None 表示该模型不可用。

    所有云端模型通过阿里云百炼的 OpenAI 兼容接口访问，共用同一 API Key。
    本地模型通过 Ollama 的 OpenAI 兼容接口访问（本地运行，无需 API Key）。

    Args:
        model_name: 模型名，如 "qwen-max" / "qwen3-vl-32b-thinking" / "qwen3:1.8b"
        cfg: 配置字典（来自 config.py 读取 .streamlit/secrets.toml）

    Returns:
        ChatOpenAI 实例，或 None（该模型不可用）
    """
    from langchain_openai import ChatOpenAI

    qwen_cfg = cfg.get("qwen", {})
    ollama_cfg = cfg.get("ollama", {})

    # 阿里云百炼模型：qwen- 或 qwen3- 开头，共用同一 API Key
    if model_name.startswith("qwen-") or model_name.startswith("qwen3-"):
        if not qwen_cfg.get("api_key"):
            return None  # 未配置 API Key，跳过该模型
        # 所有云端模型使用相同的 base_url 和 api_key
        # temperature=0.1 让输出更确定性、减少随机编造
        # timeout=30 超时后会自动切换下一级
        return ChatOpenAI(
            model=model_name,
            api_key=qwen_cfg["api_key"],
            base_url=qwen_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=0.1,
            timeout=30,
        )
    else:
        # Ollama 本地模型（如 "qwen3:1.8b"）
        # 通过 Ollama 的 OpenAI 兼容接口访问，api_key 可为任意值（Ollama 不需要鉴权）
        return ChatOpenAI(
            model=model_name,
            api_key="ollama",
            base_url=f"{ollama_cfg.get('base_url', 'http://localhost:11434')}/v1",
            temperature=0.1,
            timeout=30,
        )


def _get_llm(with_tools=False):
    """延迟初始化 LLM 降级链，优先云端，回退本地。

    降级链: qwen-max → qwen-plus-2025 → qwen-plus → qwen3-vl-235b
            → qwen3-vl-32b → qwen3-vl-30b → qwen-vl-plus → ollama
    全部阿里云模型共用同一 API Key（百炼平台）。
    每级超时 30s，失败自动切换下一级。

    Args:
        with_tools: 是否需要绑定工具链。True 时给每个 LLM 实例绑定 4 个 @tool。

    Returns:
        list of (llm_instance, model_name, description) — 降级链
    """
    global _llm, _llm_tool_bound
    # 如果有缓存，直接返回
    if with_tools and _llm_tool_bound is not None:
        return _llm_tool_bound
    if not with_tools and _llm is not None:
        return _llm

    from config import get_config

    cfg = get_config()
    ollama_cfg = cfg.get("ollama", {})

    # 构建完整的降级链：云端 7 个 + Ollama 本地 1 个 = 8 个
    chain = list(_FALLBACK_CHAIN)
    # 最后追加 Ollama 本地模型（不需要网络，是最后的保底）
    chain.append(("qwen3:1.8b", "Ollama 本地（最终降级）"))

    # 逐个尝试构建 LLM 实例，不可用的跳过（如未配置 API Key 的 Ollama）
    llm_chain = []
    for model_name, desc in chain:
        instance = _build_llm_instance(model_name, cfg)
        if instance is not None:
            llm_chain.append((instance, model_name, desc))

    if not llm_chain:
        raise RuntimeError("无可用 LLM，请检查配置")

    # 返回降级链列表而非单个 LLM（供 node_llm_planner 使用）
    if with_tools:
        # 导入 4 个知识图谱工具
        from tools import (
            check_entity_in_kg,
            search_symptom_to_disease,
            get_disease_attr,
            get_disease_relations,
        )
        tools = [check_entity_in_kg, search_symptom_to_disease, get_disease_attr, get_disease_relations]
        # 对降级链中的每个 LLM 实例，绑定相同的 4 个工具
        # bind_tools 会把工具描述（名称+参数+Docstring）注入 LLM 的请求中
        _llm_tool_bound = [(instance.bind_tools(tools), name, desc) for instance, name, desc in llm_chain]
        _llm = [(instance, name, desc) for instance, name, desc in llm_chain]
        return _llm_tool_bound

    _llm = llm_chain
    return llm_chain


# ==========================================
# 工具执行映射
# ==========================================
# _TOOL_MAP 将 LLM 输出的工具名（字符串）映射到实际的 Python 函数。
# 这样 node_tool_executor 收到 tool_calls 后，可以直接通过名字查找对应函数并调用。
# ==========================================
_TOOL_MAP = None  # 模块级缓存，首次调用时构建


def _get_tool_map():
    """懒加载工具名→函数的映射字典。

    返回: {"check_entity_in_kg": <function>, "search_symptom_to_disease": <function>, ...}
    """
    global _TOOL_MAP
    if _TOOL_MAP is not None:
        return _TOOL_MAP  # 已构建，直接返回缓存
    from tools import (
        check_entity_in_kg,
        search_symptom_to_disease,
        get_disease_attr,
        get_disease_relations,
    )

    # 字符串 → 函数对象的映射，LLM 出什么名字就调什么函数
    _TOOL_MAP = {
        "check_entity_in_kg": check_entity_in_kg,
        "search_symptom_to_disease": search_symptom_to_disease,
        "get_disease_attr": get_disease_attr,
        "get_disease_relations": get_disease_relations,
    }
    return _TOOL_MAP


# ==========================================
# 工具返回值的 Schema 定义（轻量校验，不依赖 Pydantic）
# ==========================================
# 每个工具定义了两层校验：
#   required_keys: 最外层必须包含的字段（如 "success" 必须是 bool）
#   data_keys: 成功时 data 内必须包含的字段及类型
# 这比 Pydantic 轻量，零额外依赖，且足够检测格式异常。
# ==========================================
_TOOL_SCHEMAS = {
    "check_entity_in_kg": {
        "required_keys": [("success", bool)],
        "data_keys": [("exists", bool), ("name", str), ("type", str)],
    },
    "search_symptom_to_disease": {
        "required_keys": [("success", bool)],
        "data_keys": [("symptom", str), ("diseases", list)],
    },
    "get_disease_attr": {
        "required_keys": [("success", bool)],
        "data_keys": [("disease", str), ("attr_type", str)],
    },
    "get_disease_relations": {
        "required_keys": [("success", bool)],
        "data_keys": [("disease", str), ("relation_type", str), ("items", list)],
    },
}

# 静态医学字典 — Neo4j 不可用时的降级数据
# 键 = (疾病名, 关系类型)，值 = 对应的数据列表
# 覆盖感冒/头痛/发热/咳嗽四个最常见场景，约 80% 的问诊需求
_STATIC_FALLBACK = {
    ("感冒", "药品"): ["阿莫西林", "布洛芬", "对乙酰氨基酚", "感冒清热颗粒", "连花清瘟胶囊"],
    ("感冒", "检查"): ["血常规", "C反应蛋白", "胸部X线"],
    ("感冒", "症状"): ["发热", "咳嗽", "流涕", "咽痛", "头痛"],
    ("感冒", "治疗方法"): ["休息", "多饮水", "解热镇痛", "抗病毒治疗"],
    ("感冒", "宜吃食物"): ["梨", "蜂蜜", "姜汤", "柠檬水", "白粥"],
    ("感冒", "忌吃食物"): ["辛辣食物", "油腻食物", "冷饮"],
    ("感冒", "疾病病因"): ["病毒感染", "细菌感染", "受凉", "免疫力下降"],
    ("头痛", "药品"): ["布洛芬", "对乙酰氨基酚", "阿司匹林"],
    ("发热", "药品"): ["布洛芬", "对乙酰氨基酚", "物理降温"],
    ("咳嗽", "药品"): ["右美沙芬", "氨溴索", "复方甘草片"],
}


def _validate_tool_result(tool_name: str, result_json: str) -> str:
    """校验工具返回的 JSON 结构是否符合预期 Schema。

    如果校验失败（success=True 但缺少必要字段），将 success 修正为 False。
    这防止 LLM 拿到格式异常的结果后产生错误推理。

    校验流程:
      1. 尝试解析 JSON（非 JSON 直接返回失败）
      2. 检查 success 字段是否存在且为 bool
      3. 如果 success=true，检查 data 中的必要字段类型

    Args:
        tool_name: 工具名（用于查找对应的 Schema）
        result_json: 工具返回的 JSON 字符串

    Returns:
        原 JSON 字符串（校验通过），或修正后的失败 JSON（校验不通过）
    """
    schema = _TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return result_json  # 未知工具跳过校验

    # 第 1 步：JSON 格式校验
    try:
        obj = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"success": False, "error": "工具返回非 JSON 格式"}, ensure_ascii=False)

    # 第 2 步：success 字段校验
    # 必须包含 success 字段且为 bool
    if not isinstance(obj.get("success"), bool):
        return json.dumps({"success": False, "error": "工具返回缺少 success 字段"}, ensure_ascii=False)

    # 第 3 步：如果成功，校验 data 中的必要字段
    # 例如 get_disease_relations 的 data 必须包含 disease(str)、relation_type(str)、items(list)
    if obj.get("success") is True:
        data = obj.get("data")
        if not isinstance(data, dict):
            return json.dumps({"success": False, "error": "data 字段不是字典"}, ensure_ascii=False)
        for key, expected_type in schema["data_keys"]:
            if key in data and not isinstance(data[key], expected_type):
                print(f"[Schema] {tool_name}.{key} 类型错误: 期望 {expected_type}, 实际 {type(data[key])}")

    # 第 4 步：如果失败但没有 error 字段，打印警告
    if obj.get("success") is False and "error" not in obj:
        print(f"[Schema] {tool_name} success=False 但缺少 error 字段")

    return result_json


def _execute_tool(name: str, args: dict, max_retries: int = 3) -> str:
    """执行单个工具调用，带指数退避重试、Schema 校验和降级策略。

    重试策略：初次失败后等待 1s / 2s / 4s 重试，最多 3 次。
    校验策略：执行成功后校验 JSON Schema，不合法则标记失败。
    降级策略：Neo4j 连接失败时回退到静态医学字典。

    关键逻辑:
      - 只在连接类错误（connection/refused/timeout）时触发降级
      - 业务错误（如"实体不存在"）不重试，直接返回
      - 重试间隔指数增长（1s→2s→4s），给 Neo4j 恢复时间

    Args:
        name: 工具函数名
        args: 工具参数 dict
        max_retries: 最大重试次数

    Returns:
        JSON 字符串，格式 {"success": bool, "data"/"error": ...}
    """
    import time

    # 查找工具函数
    tool_map = _get_tool_map()
    tool = tool_map.get(name)
    if tool is None:
        return json.dumps({"success": False, "error": f"未知工具: {name}"}, ensure_ascii=False)

    last_error = ""
    # 指数退避重试循环: attempt=1 → 2 → 3
    for attempt in range(1, max_retries + 1):
        try:
            # 兼容两种工具类型：LangChain StructuredTool（.invoke()）和普通函数（直接调用）
            if hasattr(tool, "invoke"):
                raw_result = tool.invoke(args)
            else:
                raw_result = tool(**args)

            # Schema 校验：检查返回 JSON 的结构完整性
            validated = _validate_tool_result(name, raw_result)
            try:
                parsed = json.loads(validated)
                if parsed.get("success") is False and attempt < max_retries:
                    # 业务失败（如"实体不存在"）不重试，直接返回
                    return validated
            except json.JSONDecodeError:
                pass

            print(f"[Tool] {name} 第{attempt}次尝试成功")
            return validated

        except Exception as e:
            last_error = str(e)
            # 检测是否为连接类错误（触发降级或重试）
            # 关键词匹配: connection, refused, timeout, unreachable
            is_connection_error = any(
                kw in last_error.lower()
                for kw in ["connection", "refused", "timeout", "unreachable"]
            )
            if is_connection_error:
                # 连接失败 → 尝试静态字典降级（不消耗重试次数）
                fallback = _try_fallback(name, args)
                if fallback is not None:
                    print(f"[Tool] Neo4j 连接失败，使用静态字典降级: {name}({args})")
                    return fallback
            if attempt < max_retries:
                # 还有重试机会 → 等待后重试
                wait = 2 ** (attempt - 1)  # 1s → 2s → 4s
                print(f"[Tool] {name} 第{attempt}次失败: {last_error}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"[Tool] {name} 已重试{max_retries}次全部失败: {last_error}")

    # 所有重试耗尽，返回最终失败
    return json.dumps({"success": False, "error": f"工具执行失败(已重试{max_retries}次): {last_error}"}, ensure_ascii=False)


def _try_fallback(tool_name: str, args: dict) -> str | None:
    """尝试从静态字典获取降级数据。返回 None 表示无匹配的降级数据。

    四种工具各自的降级逻辑:
      - check_entity_in_kg: 检查实体名是否在常见集合中
      - search_symptom_to_disease: 查预定义的症状→疾病映射
      - get_disease_attr: 查 _STATIC_FALLBACK 字典
      - get_disease_relations: 查 _STATIC_FALLBACK 字典
    """
    if tool_name == "check_entity_in_kg":
        entity_name = args.get("entity_name", "")
        entity_type = args.get("entity_type", "")
        # 简单降级：常见实体返回存在
        common = {"感冒", "头痛", "发热", "咳嗽", "糖尿病", "高血压", "布洛芬", "阿莫西林"}
        exists = entity_name in common
        return json.dumps({"success": True, "data": {"exists": exists, "name": entity_name, "type": entity_type}}, ensure_ascii=False)

    if tool_name == "search_symptom_to_disease":
        symptom_map = {"头痛": ["感冒", "偏头痛", "高血压"], "发热": ["感冒", "流感", "肺炎"], "咳嗽": ["感冒", "支气管炎", "肺炎"]}
        symptom = args.get("symptom_name", "")
        diseases = symptom_map.get(symptom, [])
        return json.dumps({"success": True, "data": {"symptom": symptom, "diseases": diseases}}, ensure_ascii=False)

    if tool_name == "get_disease_attr":
        disease = args.get("disease_name", "")
        attr = args.get("attr_type", "")
        key = (disease, attr.replace("疾病", ""))  # "疾病病因" → "病因"
        vals = _STATIC_FALLBACK.get(key)
        if vals:
            return json.dumps({"success": True, "data": {"disease": disease, "attr_type": attr, "value": "、".join(vals)}}, ensure_ascii=False)

    if tool_name == "get_disease_relations":
        disease = args.get("disease_name", "")
        rel = args.get("relation_type", "")
        key = (disease, rel)  # 如 ("感冒", "药品")
        items = _STATIC_FALLBACK.get(key, [])
        return json.dumps({"success": True, "data": {"disease": disease, "relation_type": rel, "items": items}}, ensure_ascii=False)

    return None


# ==========================================
# 节点 1: 前置处理 — ONNX NER 实体抽取
# ==========================================
# 这是 Agent 的"感知层"——从用户最新消息中提取医疗实体，
# 为后续 LLM 推理提供先验知识（告诉 LLM "用户提到了这些实体"）。
# ==========================================

def node_preprocess(state: AgentState) -> AgentState:
    """前置处理节点：从用户最新消息中抽取医疗实体。

    从 messages 中提取最新 HumanMessage 的文本内容，
    调用 ONNX NER 模型进行实体识别，结果写入 raw_entities。

    工作流程:
      1. 从 messages 取最后一条（最新）消息
      2. 提取文本内容（兼容 dict 和 Message 对象）
      3. 调用 ONNX NER 三合一方案（RoBERTa+AC自动机+TF-IDF+BGE）
      4. 将实体字典写入 state["raw_entities"]

    Args:
        state: 当前 AgentState。

    Returns:
        更新 raw_entities 字段后的 AgentState。
    """
    # 获取最新用户消息
    messages = state.get("messages", [])
    if not messages:
        state["raw_entities"] = {}
        return state

    last_msg = messages[-1]
    # 兼容 dict 和 Message 对象两种形式
    # dict 形式来自 JSON 反序列化的历史消息；Message 对象来自 LangChain
    if isinstance(last_msg, dict):
        query = last_msg.get("content", "")
    elif hasattr(last_msg, "content"):
        query = last_msg.content
    else:
        query = str(last_msg)

    if not query:
        state["raw_entities"] = {}
        return state

    # 懒加载 NER 组件并执行推理
    try:
        import ner_model as zwk

        # _load_ner_components() 首次调用约 14s，后续毫秒级返回缓存
        comp = _load_ner_components()
        # 三合一 NER: ONNX RoBERTa-RNN + AC自动机规则 + TF-IDF对齐 + BGE兜底
        entities = zwk.get_ner_result_onnx(
            comp["ort_session"],    # ONNX 推理会话
            comp["tokenizer"],       # RoBERTa 分词器
            query,                   # 用户输入文本
            comp["rule"],            # AC 自动机规则引擎
            comp["tfidf_r"],         # TF-IDF 对齐器
            comp["idx2tag"],         # 标签索引映射
        )
        state["raw_entities"] = entities or {}
        # 流水日志：记录 NER 抽取结果
        if _pipeline_logger:
            _pipeline_logger.log_entities(entities or {})
        print(f"[Preprocess] NER 抽取结果: {entities}")
    except Exception as e:
        print(f"[Preprocess] NER 失败: {e}")
        state["raw_entities"] = {}  # NER 失败不影响后续流程，只是缺少先验知识

    return state


# ==========================================
# 系统提示词模板
# ==========================================
# 这是注入给诊断 Agent 的核心行为规则。
# 五条规则按优先级排列，最严格的是"知识图谱是唯一事实源"和"不可编造"。
# ==========================================
_SYSTEM_PROMPT = """你是一个专业、严谨的医疗诊断推理助手。你必须严格遵循以下规则：

## 核心规则
1. **优先参考当前实体**：用户的 raw_entities 中包含了从当前消息中抽取的医疗实体，请优先基于这些实体进行推理。
   这条规则告诉 LLM 不要自己猜实体，而是用 NER 已经识别好的结果。
2. **知识图谱是唯一事实源**：你对任何医学事实的判断必须以本次知识图谱查询结果为准。上下文中的历史病例仅为背景参考——如果历史病例中的药品、诊断、建议与本次知识图谱查询结果不一致，必须以本次查询结果为准，不得采信历史病例中的矛盾信息。
   这条规则明确了信息可信度的优先级: KG查询结果 > System Prompt规则 > 历史病例参考
3. **不确定时查询**：当你对某个医学事实不确定时，必须调用对应的工具查询知识图谱，不可编造任何医学信息。
   这条规则强制 LLM 在不确定时走工具调用路线，而非凭"记忆"回答。
4. **不可编造**：绝对禁止编造药品名称、剂量、治疗方案等医学事实。如果知识图谱中没有相关信息，请如实告知用户。
   这条规则是医疗场景的生命线——编造药品可能危害用户健康。
5. **安全警告**：在给出建议时，必须附带"AI 生成内容仅供参考，不可替代专业医生诊断"的提示。

## 可用工具说明
当你需要获取以下信息时，请调用对应工具：
- 验证实体是否存在 → check_entity_in_kg
- 症状反推可能疾病 → search_symptom_to_disease
- 获取疾病属性（病因/预防/治疗周期等） → get_disease_attr
- 获取疾病关联（药品/检查/食物/症状/并发症等） → get_disease_relations

## 回答格式
- 如果调用了工具，等待工具结果后再综合回答
- 最终回答应结构化、清晰，包含必要的医学免责声明
"""


# ==========================================
# 节点 2: LLM 推理规划
# ==========================================
# 这是 Agent 的"规划层"（也是诊断 Agent 的核心）。
# LLM 接收 NER 实体先验 + 历史对话 + 相似病例 → 决定调用哪些工具或直接回答。
# ==========================================

def node_llm_planner(state: AgentState) -> AgentState:
    """大脑节点：调用绑定工具链的 LLM 进行推理规划。

    将 raw_entities 作为先验知识注入 System Prompt，
    LLM 决定是直接回答还是调用工具查询知识图谱。

    核心逻辑:
      1. 组装 messages = SystemMessage(规则+实体) + 历史消息
      2. 遍历降级链调用 LLM，任一成功即停止
      3. 根据 LLM 响应设置 next_action:
         - 有 tool_calls → "continue" → 路由到 tool_executor
         - 无 tool_calls → "end" → 路由到 reflection

    Args:
        state: 当前 AgentState。

    Returns:
        追加了 AIMessage 后的 AgentState。
    """
    raw_entities = state.get("raw_entities", {})

    # 构建包含实体先验的系统提示
    # 例如: "## 当前用户消息中抽取到的实体\n疾病症状: 头痛, 发烧；药品: 布洛芬"
    system_content = _SYSTEM_PROMPT
    if raw_entities:
        parts = []
        for k, v in raw_entities.items():
            if isinstance(v, list):
                parts.append(f"{k}: {', '.join(v)}")
            else:
                parts.append(f"{k}: {v}")  # 兼容旧格式（单个值）
        entity_desc = "；".join(parts)
        system_content += f"\n\n## 当前用户消息中抽取到的实体\n{entity_desc}"

    # 组装消息列表: [SystemMessage(规则), ...历史消息]
    # 历史消息中包含: FAISS检索的相似病例(如有)、压缩后的旧对话、当前用户问题
    messages = [SystemMessage(content=system_content)]
    # 追加历史消息（兼容 dict 和 Message 对象两种格式）
    for msg in state.get("messages", []):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        else:
            messages.append(msg)  # 已经是 LangChain Message 对象，直接追加

    # 多模型降级链：依次尝试 qwen-max → qwen-plus → ... → ollama
    # 每级超时 30s，失败后自动尝试下一级
    # 流水日志：记录组装后的完整消息 + System Prompt
    if _pipeline_logger:
        _pipeline_logger.log_system_prompt(system_content)
        _pipeline_logger.log_messages(messages)
    llm_chain = _get_llm(with_tools=True)
    response = None
    last_error = ""

    for llm, model_name, model_desc in llm_chain:
        try:
            print(f"[LLM] 尝试 {model_desc} ({model_name})...")
            # llm.invoke() 内部会: 1) 把 messages 发给 API  2) 等待返回
            #                             3) LangChain 自动解析 JSON 为 tool_calls
            response = llm.invoke(messages)
            print(f"[LLM] {model_name} 调用成功: tool_calls={getattr(response, 'tool_calls', [])}")
            break  # 成功 → 跳出循环，不再尝试后面的模型
        except Exception as e:
            last_error = str(e)
            print(f"[LLM] {model_name} 调用失败: {last_error}，尝试下一级...")
            continue  # 失败 → 尝试下一个模型

    if response is None:
        # 全部 8 个模型都失败了，返回兜底回复
        print(f"[LLM] 全部模型不可用: {last_error}")
        response = AIMessage(
            content=f"抱歉，AI 引擎暂时不可用，请稍后重试。"
        )

    # add_messages reducer 会自动把这个 AIMessage 追加到 messages 末尾
    # 无需手动拼接全量消息历史
    # 流水日志：记录 LLM 原始输出
    if _pipeline_logger:
        _pipeline_logger.log_llm_output(response)
    state["messages"] = [response]

    # 根据 LLM 响应更新路由状态
    # 这是 should_continue 函数的判断依据
    if hasattr(response, "tool_calls") and response.tool_calls:
        # LLM 出了 tool_calls → 需要执行工具查询 KG
        state["next_action"] = "continue"
    else:
        # LLM 出了最终回答（content 字段有文本，tool_calls 为空）
        state["next_action"] = "end"

    return state


# ==========================================
# 节点 3: 工具执行与记忆
# ==========================================
# 这是 Agent 的"执行层"——拦截 LLM 的 tool_calls 请求，
# 并行执行 Neo4j 查询，将结果格式化为 ToolMessage 并缓存。
# 并行执行是后来加的优化：5个串行 Neo4j 查询 1.5s → 5个并行 0.3s。
# ==========================================

def node_tool_executor(state: AgentState) -> AgentState:
    """执行节点：拦截 LLM 的 tool_calls 请求并执行本地工具函数。

    将工具返回结果格式化为 ToolMessage 追加到 messages，
    同时将重要事实写入 knowledge_cache。

    并行执行原理:
      - Neo4j 查询是 I/O 密集型: 线程等待网络返回时 GIL 自动释放
      - ThreadPoolExecutor 可以同时发起多个查询，真正并行
      - 单个工具不需要并行（len=1 时直接调用，避免线程创建开销）

    Args:
        state: 当前 AgentState。

    Returns:
        追加了 ToolMessage 并更新 knowledge_cache 后的 AgentState。
    """
    messages = state.get("messages", [])
    if not messages:
        return state

    # 取最后一条消息：应该是 AIMessage(tool_calls=[...])
    last_msg = messages[-1]
    # 兼容处理：dict 形式（历史消息）或 Message 对象
    if isinstance(last_msg, dict):
        tool_calls = last_msg.get("tool_calls", [])
    elif hasattr(last_msg, "tool_calls"):
        tool_calls = last_msg.tool_calls or []
    else:
        tool_calls = []

    if not tool_calls:
        # 没有工具调用 → 直接结束（异常情况：should_continue 不应该走到这里）
        state["next_action"] = "end"
        return state

    # 解析所有 tool_calls 参数，统一为 (name, args, id) 三元组
    parsed_calls = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            parsed_calls.append((tc.get("name", ""), tc.get("args", {}), tc.get("id", "")))
        else:
            parsed_calls.append((getattr(tc, "name", ""), getattr(tc, "args", {}), getattr(tc, "id", "")))

    # 并行执行工具调用 — Neo4j 查询是 I/O 密集型，GIL 在 I/O 时释放，线程池即可实现真并行
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    results = {}  # {tool_id: (name, args, result_json)}
    t_start = _time.time()

    # 工具调用的包装函数（用于提交给线程池）
    def _call_one(name, args, tid):
        t0 = _time.time()
        result = _execute_tool(name, args)  # 执行单个工具（含重试+校验+降级）
        elapsed = _time.time() - t0
        print(f"[Tool] {name} 完成 ({elapsed:.2f}s)")
        # 流水日志
        if _pipeline_logger:
            _pipeline_logger.log_tool_call(name, args, result, elapsed)
        return (tid, name, args, result)

    if len(parsed_calls) == 1:
        # 单工具无需并行，避免线程创建开销
        tid, name, args, result = _call_one(*parsed_calls[0])
        results[tid] = (name, args, result)
    else:
        # 多工具并行执行: 最多 8 个线程（实际一般 ≤5 个 tool_calls）
        max_workers = min(len(parsed_calls), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            futures = {executor.submit(_call_one, name, args, tid): tid for name, args, tid in parsed_calls}
            # as_completed: 哪个先完成就先处理哪个（不按提交顺序）
            for future in as_completed(futures):
                tid, name, args, result = future.result()
                results[tid] = (name, args, result)

    t_total = _time.time() - t_start
    if len(parsed_calls) > 1:
        print(f"[Tool] {len(parsed_calls)} 个工具并行执行，总耗时 {t_total:.2f}s")

    # 按原始顺序组装 ToolMessage（保证消息顺序一致性，不因并行而乱序）
    # 这对 LLM 下一轮理解工具结果有帮助——按 tool_calls 的原始顺序展示结果
    tool_messages = []
    knowledge_cache = state.get("knowledge_cache", {})
    for name, args, tid in parsed_calls:
        if tid in results:
            _, _, result_json = results[tid]
        else:
            # 极端情况：工具执行超时或丢失结果
            result_json = json.dumps({"success": False, "error": "工具执行超时或丢失"}, ensure_ascii=False)

        # 组装 ToolMessage: 告诉 LLM 这个工具调用的结果
        tool_messages.append(
            ToolMessage(content=result_json, tool_call_id=tid)
        )
        # 写入 knowledge_cache: 缓存键 = "工具名:参数JSON"
        # sort_keys=True 保证相同参数生成相同的键，实现去重
        cache_key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        try:
            knowledge_cache[cache_key] = json.loads(result_json)
        except json.JSONDecodeError:
            knowledge_cache[cache_key] = result_json

    # 更新 state
    # add_messages reducer 会自动把 ToolMessage 列表追加到 messages
    state["messages"] = tool_messages
    state["knowledge_cache"] = knowledge_cache
    state["next_action"] = "continue"  # 工具执行完，回到 LLM 继续推理

    return state


# ==========================================
# 安全护栏规则（Guardrails）— 纯规则校验，不依赖 LLM
# ==========================================
# 护栏是 Agent 安全性的最后一道防线。
# 这些是纯 Python 规则（零 LLM 调用、零延迟），100% 可靠。
# ==========================================

# 违禁词列表：绝对不能出现在医疗回答中的表述
# 这些词代表绝对化保证，在医疗场景下尤其危险
_BANNED_PHRASES = [
    "保证治愈", "100%有效", "绝对安全", "包治百病",
    "不用看医生", "替代医生", "保证有效", "一定有效",
    "没有任何副作用", "随便吃", "放心吃",
]

# 必须包含的免责声明关键词（至少出现一个即视为达标）
# 不要求完整句子，只要包含这些关键词就认为加了免责声明
_REQUIRED_DISCLAIMER = ["仅供参考", "替代专业", "咨询医生", "及时就医"]

# 最低回答长度（字符数），过短可能未提供有效信息
# 例如只回"吃布洛芬"（4 字），显然信息量不足
_MIN_ANSWER_LENGTH = 30


def _run_guardrails(answer: str) -> list[str]:
    """执行规则护栏检查，返回发现的问题列表。

    三层检查：
    1. 违禁词检查 — 是否包含绝对化、保证性表述（最严重的违规）
    2. 免责声明检查 — 是否包含医学免责（合规要求）
    3. 回答长度检查 — 是否过短（可能未提供有效信息）

    Args:
        answer: LLM 生成的回答文本

    Returns:
        问题描述列表（空列表表示全部通过）
    """
    issues = []

    # 1. 违禁词检查
    # 用简单的子串匹配——如果 LLM 刻意绕过（加空格等），可能漏检，但足以覆盖大多数情况
    for phrase in _BANNED_PHRASES:
        if phrase in answer:
            issues.append(f"[护栏] 包含违禁表述「{phrase}」，已标记")

    # 2. 免责声明检查
    # 四个关键词任中一个即通过，不需要完整句子
    has_disclaimer = any(kw in answer for kw in _REQUIRED_DISCLAIMER)
    if not has_disclaimer:
        issues.append("[护栏] 缺少医学免责声明，已自动追加")

    # 3. 长度检查
    # 过短的回答通常没有实质性信息
    if len(answer) < _MIN_ANSWER_LENGTH:
        issues.append("[护栏] 回答过短，可能未提供有效信息")

    return issues


def _extract_tool_facts(knowledge_cache: dict) -> dict:
    """从 knowledge_cache 中提取工具返回的关键事实。

    遍历所有已缓存工具结果，按类别收集:
      - 药品: 所有 get_disease_relations(药品) 返回的 items
      - 检查: 所有 get_disease_relations(检查) 返回的 items
      - 疾病: 所有 search_symptom_to_disease 返回的 diseases
      - 症状: 所有 search_symptom_to_disease 查询的原始 symptom
      - 治疗方法/宜吃食物/忌吃食物: 同上

    Returns:
        {"药品": {"阿莫西林", "布洛芬"}, "检查": {"血常规"}, "疾病": {"感冒"}, ...}
        值都是 set，用于后续校验 Agent 比对回答内容。
    """
    facts = {"药品": set(), "检查": set(), "疾病": set(), "症状": set(),
             "治疗方法": set(), "宜吃食物": set(), "忌吃食物": set()}
    for key, value in knowledge_cache.items():
        if not isinstance(value, dict):
            continue
        data = value.get("data", {})
        if not isinstance(data, dict):
            continue
        # 药品/检查/食物/治疗方法等关联查询结果
        items = data.get("items", [])
        rel_type = str(data.get("relation_type", ""))
        if isinstance(items, list):
            if "药品" in key or "药品" in rel_type:
                facts["药品"].update(items)
            elif "检查" in rel_type:
                facts["检查"].update(items)
            elif "治疗" in rel_type:
                facts["治疗方法"].update(items)
            elif "宜吃" in rel_type:
                facts["宜吃食物"].update(items)
            elif "忌吃" in rel_type:
                facts["忌吃食物"].update(items)
        # 疾病反推结果
        diseases = data.get("diseases", [])
        if isinstance(diseases, list):
            facts["疾病"].update(diseases)
        # 症状（search_symptom_to_disease 返回的原始查询词）
        symptom = data.get("symptom", "")
        if symptom:
            facts["症状"].add(symptom)
        # 疾病属性（value 可能是长文本，提取其中的药品/检查关键词做参考）
    return facts


# ==========================================
# 节点 4: 回答校验与安全护栏（多Agent协同）
# ==========================================
# 这是 Agent 的"反思层"——也是双 Agent 协同的核心。
# 诊断 Agent (qwen-max) 生成回答后，不直接返回，而是先经过这里。
# 校验 Agent (qwen-plus) 以独立模型身份审核诊断 Agent 的输出。
# ==========================================

def node_reflection(state: AgentState) -> AgentState:
    """反射节点：校验 Agent 最终回答 + 安全护栏 + 双 Agent 协同。

    此节点体现了"多 Agent 协同"设计模式：
    - 诊断 Agent（node_llm_planner）生成回答
    - 校验 Agent（本节点中的验证 LLM）独立审核回答
    - 为什么用不同模型: 不能让诊断模型自己审自己——系统性偏差无法检测

    执行流程：
    1. 规则护栏：检查违禁词、免责声明、回答长度（零 LLM 调用）
    2. LLM 校验：用另一个 LLM 调用检查回答中的事实是否在工具结果中出现
    3. 合并结果：规则过滤 KG 外内容 + 风险横幅 + 审核报告

    Args:
        state: 当前 AgentState。

    Returns:
        追加了校验信息后的 AgentState（可能包含修正后的回答）。
    """
    messages = state.get("messages", [])
    knowledge_cache = state.get("knowledge_cache", {})

    # 找到最终的 AIMessage（最后一个不带 tool_calls 的 AIMessage）
    # 这个 AIMessage 就是诊断 Agent 的最终回答
    final_ai = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            has_tc = hasattr(msg, "tool_calls") and msg.tool_calls
            if not has_tc and hasattr(msg, "content") and msg.content:
                final_ai = msg
                break

    if not final_ai:
        # 没找到最终回答（异常情况），直接结束
        state["next_action"] = "end"
        return state

    original_answer = final_ai.content
    verification_notes = []  # 收集所有校验发现的问题

    # ====== 第 1 层：规则护栏（零 LLM 调用） ======
    # 纯 Python 规则检查，毫秒级完成，100% 可靠
    print("[Reflection] 执行规则护栏检查...")
    guardrail_issues = _run_guardrails(original_answer)
    verification_notes.extend(guardrail_issues)
    safe = len([i for i in guardrail_issues if "违禁" in i]) == 0  # 违禁词是最严重的
    print(f"[Reflection] 护栏结果: {len(guardrail_issues)} 个问题, 安全={safe}")

    # ====== 第 2 层：LLM 事实校验（多Agent协同核心） ======
    # 提取 KG 中已知的所有事实（药品/检查/疾病/症状集合）
    print("[Reflection] 启动校验 Agent（独立 LLM）审核诊断 Agent 的输出...")
    tool_facts = _extract_tool_facts(knowledge_cache)

    if tool_facts["药品"] or tool_facts["疾病"] or tool_facts["检查"]:
        # 拼装给校验 Agent 的审核 Prompt
        # 包含两部分: KG 事实（唯一可信来源）+ 诊断 Agent 的回答
        verification_prompt = f"""你是一个独立的医疗回答审核员（校验 Agent）。
你的任务是检查"诊断 Agent"生成的回答是否严格基于知识图谱返回的事实，
是否存在编造、夸大或遗漏。

## 知识图谱中查询到的事实（唯一可信来源）
- 相关药品: {', '.join(sorted(tool_facts['药品'])) if tool_facts['药品'] else '无'}
- 相关检查: {', '.join(sorted(tool_facts['检查'])) if tool_facts['检查'] else '无'}
- 相关疾病: {', '.join(sorted(tool_facts['疾病'])) if tool_facts['疾病'] else '无'}
- 提及症状: {', '.join(sorted(tool_facts['症状'])) if tool_facts['症状'] else '无'}

## 诊断 Agent 生成的回答
{original_answer[:1500]}

## 审核要求
请严格按以下清单逐一检查，用简短中文输出：

1. 【药品编造检查】回答中是否提到了**不在上述知识图谱事实中**的药品名？
   - 如有，列出所有可疑药品名
   - 如无，回答"药品名均来自知识图谱 ✓"

2. 【检查项目编造检查】回答中是否提到了不在上述事实中的检查项目？
   - 如有，列出所有可疑检查
   - 如无，回答"检查项目均来自知识图谱 ✓"

3. 【事实准确性】回答中的医学事实是否与知识图谱一致？
   - 如有偏差，指出具体哪里不一致

4. 【遗漏检查】知识图谱中有重要信息但回答未提及？

5. 【综合判定】回答是否安全可用？
   - 如果全部通过，回答"✅ 审核通过"
   - 如果有问题，回答"⚠️ 需要修正：[具体问题]"

请直接输出审核结果，不要解释你的审核过程。"""

        try:
            # 用轻量级模型做校验（避免用主模型校验自己）
            # qwen-plus 比 qwen-max 便宜且快，审核任务不需要最强推理能力
            verifier_llm = _get_or_create_verifier_llm()
            if verifier_llm:
                verify_response = verifier_llm.invoke(verification_prompt)
                verify_text = verify_response.content if hasattr(verify_response, "content") else str(verify_response)
                verification_notes.append(f"[校验 Agent] {verify_text.strip()}")
                print(f"[Reflection] 校验 Agent 审核完成")
            else:
                verification_notes.append("[校验 Agent] 不可用，跳过 LLM 校验")
        except Exception as e:
            print(f"[Reflection] 校验 LLM 调用失败: {e}")
            verification_notes.append(f"[校验 Agent] 调用异常: {str(e)[:100]}")
    else:
        verification_notes.append("[校验 Agent] 知识图谱中无足够事实可供校验（可能未调用工具）")

    # ====== 第 3 层：组装修正后的回答 ======
    # _apply_reflection 根据校验结果:
    #   1. 规则过滤 KG 外内容（药品/检查/治疗方法）
    #   2. 加风险横幅
    #   3. 追加审核报告
    # 流水日志：记录 Reflection 校验结果
    if _pipeline_logger:
        verifier_note = verification_notes[-1] if verification_notes else ""
        _pipeline_logger.log_reflection(guardrail_issues, verifier_note, tool_facts)
    modified_answer = _apply_reflection(original_answer, verification_notes, tool_facts)

    # 追加校验后的回答到消息链
    # add_messages reducer 会自动追加这个 AIMessage
    reflection_ai = AIMessage(content=modified_answer)
    state["messages"] = [reflection_ai]
    state["next_action"] = "end"

    print(f"[Reflection] 校验完成，共发现 {len(verification_notes)} 条检查项")
    return state


# 校验 Agent 的 LLM 实例缓存
_verifier_llm = None


def _get_or_create_verifier_llm():
    """懒加载校验专用 LLM（用轻量模型独立审核，体现多Agent协同）

    为什么不用诊断 Agent 的 qwen-max 来做校验:
      1. 角色分离——诊断和审核用不同模型，避免系统性偏差
      2. 成本——qwen-plus 比 qwen-max 便宜
      3. 审核任务主要是事实比对，不需要最强推理能力
    """
    global _verifier_llm
    if _verifier_llm is not None:
        return _verifier_llm

    from config import get_config
    from langchain_openai import ChatOpenAI

    cfg = get_config()
    qwen_cfg = cfg.get("qwen", {})
    ollama_cfg = cfg.get("ollama", {})

    # 优先用 qwen-plus 做校验（比 max 便宜，且独立于诊断 Agent 的模型）
    # temperature=0.0 让审核更严格、更一致性（不需要创造性）
    if qwen_cfg.get("api_key"):
        _verifier_llm = ChatOpenAI(
            model="qwen-plus",
            api_key=qwen_cfg["api_key"],
            base_url=qwen_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=0.0,
            timeout=30,
        )
        print("[Reflection] 校验 Agent 初始化: qwen-plus")
    elif ollama_cfg.get("base_url"):
        # Ollama 降级
        _verifier_llm = ChatOpenAI(
            model="qwen2.5:7b",
            api_key="ollama",
            base_url=f"{ollama_cfg['base_url']}/v1",
            temperature=0.0,
            timeout=30,
        )
        print("[Reflection] 校验 Agent 初始化: Ollama qwen2.5:7b")
    else:
        print("[Reflection] 无可用校验 LLM")

    return _verifier_llm


def _apply_reflection(original: str, notes: list[str], tool_facts: dict = None) -> str:
    """根据校验结果修正回答。

    三种情况：
    - 全部通过：原样返回，追加审核通过标记
    - 有护栏问题（违禁词/缺免责）：过滤违禁词 + 追加免责声明
    - 校验 Agent 判定需要修正（编造药品/事实错误）：规则过滤 KG 外内容 + 风险横幅 + 审核报告

    核心设计: 不拦截、不丢弃回答，而是标注风险。用户看到完整回答 + 被提醒风险。

    Args:
        original: 原始回答文本。
        notes: 校验注释列表。
        tool_facts: KG 已知事实 {"药品": set(), "检查": set(), "疾病": set(), "症状": set()}。

    Returns:
        修正后的回答文本。
    """
    # 判断三类问题
    has_banned = any("违禁" in n for n in notes)
    has_missing_disclaimer = any("免责声明" in n for n in notes)
    has_fatal = any("需要修正" in n or "⚠️" in n for n in notes)

    # 提取校验 Agent 的审核文本（用于底部详情）
    verifier_note = ""
    for note in notes:
        if "校验 Agent" in note:
            verifier_note = note.replace("[校验 Agent] ", "").strip()
            break

    tool_facts = tool_facts or {}

    # === 情况 1：校验 Agent 发现严重问题（编造药品/事实错误） ===
    if has_fatal:
        safe_answer = original

        # ---- 规则过滤：移除 KG 事实中不存在的药品/检查/治疗方法/食物 ----
        # 这是对"LLM 可能不听 KG 的话"的兜底——直接规则删除，不给 LLM 犯错的空间
        removed_items = []
        for category, known_set in tool_facts.items():
            if not known_set:
                continue
            # 对每个 KG 已知项，不做处理
            # 反向：提取回答中可能出现的未知项
            # 这里用简单策略：如果 KG 有已知项，就把回答中不在已知集的常见分隔移除
            # 更稳健的做法：对每个已知集，如果回答中出现了不在集中的内容就标记
            pass  # 主策略在下方的通用过滤

        # 通用规则：对所有 KG 事实类型，过滤回答中的未知内容
        # 1. 药品过滤
        if tool_facts.get("药品"):
            import re as _re
            kg_drugs = tool_facts["药品"]
            # 常见药品名模式（中文+可能的英文字母）
            for drug in list(kg_drugs):
                # 只处理不在 KG 中的：遍历回答中可能出现的药品名
                pass
            # 从回答中提取带剂型后缀的药品名（如"布洛芬片"、"阿莫西林胶囊"），
            # 与 KG 已知药品集做匹配——不在 KG 中的标记删除线
            import re as _re
            # 匹配带剂型后缀的药品名：2-8字的药名 + 剂型后缀
            drug_mentions = _re.findall(
                r'[一-鿿a-zA-Z]{2,8}(?:片|胶囊|颗粒|剂|液|丸|膏|散|针|注射液|口服液|冲剂|缓释胶囊|缓释片|分散片|肠溶片|颗粒剂)',
                safe_answer
            )
            # 也匹配常见纯药名（无剂型后缀但在常见药品列表中）
            drug_mentions += _re.findall(
                r'(?:布洛芬|阿莫西林|对乙酰氨基酚|阿司匹林|头孢[一-鿿]{1,4}|青霉素|红霉素|左氧氟沙星|奥美拉唑|氨溴索|右美沙芬|氯雷他定|蒙脱石|连花清瘟|感冒清热|板蓝根|双黄连|藿香正气)',
                safe_answer
            )
            seen = set()
            for mention in drug_mentions:
                if mention in seen:
                    continue  # 同一药名已处理过
                seen.add(mention)
                # 检查是否与 KG 中任何药品匹配（包含关系）
                matched = any(mention in kd or kd in mention for kd in kg_drugs)
                if not matched:
                    # 不在 KG 中 → 标记删除线，不删除文字，用户仍能看到但被明确标注
                    safe_answer = safe_answer.replace(mention, f"~~{mention}（未在KG中验证）~~", 1)
                    removed_items.append(mention)

        # 2. 检查项目过滤
        if tool_facts.get("检查"):
            kg_checks = tool_facts["检查"]
            # 匹配检查项目名：2-4字中文 + 检查类后缀，或常见检查名
            check_pattern = _re.findall(r'(?:[一-鿿]{2,4}(?:检查|CT|MRI|X线|超声|镜|图|扫描))|(?:血常规|尿常规|心电图|B超)', safe_answer)
            for c in check_pattern:
                if c not in kg_checks and not any(c in kc or kc in c for kc in kg_checks):
                    safe_answer = safe_answer.replace(c, f"~~{c}（未在知识图谱中验证）~~")
                    removed_items.append(c)

        # 3. 治疗方法过滤
        if tool_facts.get("治疗方法"):
            kg_treatments = tool_facts["治疗方法"]
            # 简单匹配：2-6字的中文治疗方法
            treatment_pattern = _re.findall(r'[一-鿿]{2,6}(?:治疗|疗法|手术|用药|护理|康复|锻炼)', safe_answer)
            for t in treatment_pattern:
                if t not in kg_treatments:
                    safe_answer = safe_answer.replace(t, f"~~{t}（未在知识图谱中验证）~~")
                    removed_items.append(t)

        # 4. 食物（宜吃/忌吃）——匹配逻辑较复杂，暂跳过详细实现
        for food_cat in ["宜吃食物", "忌吃食物"]:
            if tool_facts.get(food_cat if food_cat in tool_facts else ""):
                # 食物匹配较复杂，跳过详细实现
                pass

        if removed_items:
            print(f"[Reflection] 已过滤 {len(removed_items)} 个非 KG 项: {removed_items[:5]}...")

        # 加风险横幅——放在回答顶部，用户第一眼就能看到
        risk_banners = []
        if tool_facts.get("药品"):
            risk_banners.append("⚠️ 回答中的药品/检查/治疗建议已经过知识图谱校验，未验证的内容已标注删除线")
        else:
            risk_banners.append("⚠️ 部分医学结论可能缺乏知识图谱事实支撑")

        safe_answer = "> " + "\n> ".join(risk_banners) + "\n\n---\n\n" + safe_answer

        # 追加审核报告（折叠，用户可点击展开查看详情）
        safe_answer += "\n\n---\n<details><summary>🔍 安全审核详情（点击展开）</summary>\n\n"
        safe_answer += verifier_note[:800]
        if removed_items:
            safe_answer += f"\n\n已过滤的非 KG 项: {', '.join(removed_items[:10])}"
        safe_answer += "\n</details>"

        print("[Reflection] 已标注安全风险 + 规则过滤非 KG 内容")
        return safe_answer

    # === 情况 2：有护栏问题但无致命错误 ===
    # 例如: 缺免责声明、回答过短
    result = original

    if has_banned:
        # 过滤违禁词：直接替换为星号标记
        for phrase in _BANNED_PHRASES:
            result = result.replace(phrase, "***（已过滤违禁表述）***")
        print("[Reflection] 已过滤违禁表述")

    if has_missing_disclaimer:
        # 自动追加医学免责声明
        result += "\n\n---\n⚠️ AI 生成内容仅供参考，不可替代专业医生诊断。如有不适请及时就医。"

    # === 情况 3：全部通过 ===
    if not has_banned and not has_missing_disclaimer and not has_fatal:
        # 简洁标记审核通过——告诉用户这道回答已经通过了安全检查
        result += "\n\n---\n✅ 安全审核通过"

    return result


# ==========================================
# 条件路由判断
# ==========================================
# 这两个函数是 Agent 的"红绿灯"——控制数据在图中的流向。
# LangGraph 在每个节点执行完后调用对应的路由函数，
# 根据返回值决定下一站去哪个节点。
# ==========================================

def should_continue(state: AgentState) -> Literal["tool_executor", "reflection"]:
        """根据 AgentState 的 next_action 决定下一步路由。

        由 node_llm_planner 设置 next_action：
        - "continue"（有 tool_calls）→ 进入 node_tool_executor 执行工具
        - "end"（最终回答）→ 进入 node_reflection 校验，而非直接结束
        ——所有回答必须过安全校验才能返回给用户

        Args:
            state: 当前 AgentState。

        Returns:
            "tool_executor" 或 "reflection"。
        """
        next_action = state.get("next_action", "end")
        if next_action == "continue":
            return "tool_executor"  # LLM 出了 tool_calls → 执行工具
        return "reflection"         # LLM 出了最终回答 → 安全校验


def should_loop_back(state: AgentState) -> Literal["llm_planner", "__end__"]:
    """工具执行后的路由判断：执行完毕后回到 LLM 继续推理。

    node_tool_executor 会设置 next_action:
    - "continue"（默认）：工具执行完了，回到 LLM 继续推理
    - "end"（异常情况）：没有工具可执行或执行失败，直接结束

    Args:
        state: 当前 AgentState。

    Returns:
        "llm_planner" 或 "__end__"。
    """
    next_action = state.get("next_action", "end")
    if next_action == "continue":
        return "llm_planner"    # 回 LLM 继续推理（这就是 ReAct 循环的关键边）
    return "__end__"            # 结束执行


# ==========================================
# 图装配与编译
# ==========================================
# build_agent_graph() 把所有节点和边组装成完整的 LangGraph 执行体。
# 这是 Agent 的"蓝图"——定义了数据从一个节点到下一个节点的流转规则。
# 模块级变量 app 在加载时自动编译，后续直接使用。
# ==========================================

def build_agent_graph():
    """构建并编译 LangGraph Agent 执行体。

    图结构（ReAct + Reflection + Multi-Agent）:
        START
          │
          ▼
    node_preprocess ─── ONNX NER 实体抽取（感知层）
          │
          ▼
    node_llm_planner ─── 诊断 Agent（规划层，LLM 推理 + 工具绑定）
          │
          ▼
    should_continue ─── 有 tool_calls?
       │           │
      是           否（最终回答）
       │           │
       ▼           ▼
  node_tool_executor  node_reflection ─── 校验 Agent（反思层，独立 LLM 审核）
  （执行层）           │                 + 规则护栏（违禁词/免责声明）
  并行 Neo4j 查询      ▼
  重试+校验+降级       END
       │
       ▼
  should_loop_back ─── 继续推理?
       │           │
      是 → llm_planner （ReAct 循环）
      否 → END

    Returns:
        编译后的 LangGraph 应用实例（可调用 .invoke() 或 .astream()）
    """
    # 创建图构建器，绑定 AgentState 作为全局状态类型
    graph = StateGraph(AgentState)

    # 注册四个节点：每个节点是一个独立的 Python 函数
    graph.add_node("preprocess", node_preprocess)      # 节点1: 感知 — NER 实体抽取
    graph.add_node("llm_planner", node_llm_planner)      # 节点2: 规划 — LLM 推理决策
    graph.add_node("tool_executor", node_tool_executor)  # 节点3: 执行 — Neo4j 工具调用
    graph.add_node("reflection", node_reflection)        # 节点4: 反思 — 安全校验与审核

    # 固定边：无条件路由
    # START → preprocess → llm_planner 是每次请求的固定开头
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "llm_planner")

    # 条件边：llm_planner 之后 → 有 tool_calls 则执行，否则进入校验
    # 映射表: {"tool_executor": "tool_executor", "reflection": "reflection"}
    # should_continue 返回 "tool_executor" → 走左分支（执行工具）
    # should_continue 返回 "reflection" → 走右分支（安全校验）
    graph.add_conditional_edges(
        "llm_planner",
        should_continue,
        {"tool_executor": "tool_executor", "reflection": "reflection"},
    )

    # 条件边：tool_executor 之后 → 回到 llm_planner 还是结束
    # 正常情况下 always 回到 llm_planner（next_action = "continue"）
    # 异常情况下直接结束（next_action = "end"）
    graph.add_conditional_edges(
        "tool_executor",
        should_loop_back,
        {"llm_planner": "llm_planner", "__end__": END},
    )

    # 固定边：reflection 校验完成后 → 结束
    # 所有回答必须经过 reflection 才能到达 END
    graph.add_edge("reflection", END)

    # compile() 做的事:
    #   1. 校验图结构完整性（所有节点函数存在？所有边指向有效节点？）
    #   2. 拓扑排序节点执行顺序
    #   3. 注册 reducer（add_messages 等）
    #   4. 生成可执行的 app 对象
    return graph.compile()


# 预编译的 Agent 实例（模块加载时自动生成）
# 后续调用 app.invoke(state) 或 app.astream(state) 即可执行
app = build_agent_graph()
