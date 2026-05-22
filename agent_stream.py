"""LangGraph Agent 流式桥接 — 将异步执行桥接到同步生成器供 Streamlit 消费"""
import json
from queue import Queue, Empty
from threading import Thread
import asyncio

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from agent_graph import app, create_initial_state
from vector_memory import search_similar, add_case, format_similar_cases, save_index
from pipeline_logger import PipelineLogger
from agent_graph import set_pipeline_logger

# 可分辨的节点名称列表
_NODE_NAMES = {"preprocess", "llm_planner", "tool_executor"}

# 工具名中英文映射 — 前端展示用
_TOOL_CN_NAMES = {
    "check_entity_in_kg":      "验证实体存在性",
    "search_symptom_to_disease": "症状反推疾病",
    "get_disease_attr":        "获取疾病属性",
    "get_disease_relations":   "获取疾病关联信息",
}

# 历史压缩阈值：超过此轮数触发摘要
_MAX_FULL_ROUNDS = 5


def _compress_history(history_messages: list) -> list:
    """长对话摘要压缩：保留最近轮完整内容，旧轮压缩为结构化摘要。

    当历史消息超过 _MAX_FULL_ROUNDS 轮时：
    1. 取最近 N 轮保留完整 Human/AI 消息
    2. 对更早的轮次，提取关键信息（疾病、症状、药品、检查）生成摘要
    3. 摘要作为 SystemMessage 注入，帮助 LLM 了解背景但不占太多 token
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    full_msgs = []
    old_msgs = []

    total_rounds = (len(history_messages) + 1) // 2

    if total_rounds <= _MAX_FULL_ROUNDS:
        for m in history_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                full_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                full_msgs.append(AIMessage(content=content))
        return full_msgs

    keep_from = len(history_messages) - (_MAX_FULL_ROUNDS * 2)
    recent_msgs = history_messages[keep_from:]
    old_msgs = history_messages[:keep_from]

    mentioned_diseases = set()
    mentioned_symptoms = set()
    mentioned_drugs = set()
    mentioned_checks = set()

    for m in old_msgs:
        ent_str = m.get("ent", "{}")
        try:
            ent = eval(ent_str) if ent_str and ent_str != "{}" else {}
            for k, v in ent.items():
                if k == "疾病":
                    mentioned_diseases.add(v)
                elif k in ("疾病症状", "症状"):
                    mentioned_symptoms.add(v)
                elif k == "药品":
                    mentioned_drugs.add(v)
                elif k == "检查项目":
                    mentioned_checks.add(v)
        except Exception:
            pass

    summary_parts = ["[历史对话摘要] 以下为之前讨论的要点："]
    if mentioned_diseases:
        summary_parts.append(f"- 讨论过的疾病: {', '.join(sorted(mentioned_diseases))}")
    if mentioned_symptoms:
        summary_parts.append(f"- 提及的症状: {', '.join(sorted(mentioned_symptoms))}")
    if mentioned_drugs:
        summary_parts.append(f"- 讨论过的药品: {', '.join(sorted(mentioned_drugs))}")
    if mentioned_checks:
        summary_parts.append(f"- 涉及的检查项目: {', '.join(sorted(mentioned_checks))}")
    summary_parts.append(f"- 共 {len(old_msgs) // 2} 轮旧对话已压缩")

    summary_msg = SystemMessage(content="\n".join(summary_parts))

    result = [summary_msg]
    for m in recent_msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))

    print(f"[Compress] {total_rounds} 轮 -> 保留 {_MAX_FULL_ROUNDS} 轮 + 摘要 ({len(old_msgs)//2} 轮压缩)")
    return result




def stream_agent(query: str, memory: dict = None, history_messages: list = None):
    """同步生成器，逐步产出 Agent 执行过程中的结构化事件。

    Args:
        query: 用户当前输入的自然语言问题。
        memory: 可选，前几轮积累的知识缓存 dict，用于跨轮记忆。
        history_messages: 可选，之前的对话历史 [{"role":..., "content":...}, ...]，
            会自动触发长对话摘要压缩（超过5轮时）。

    产出的 dict 类型：
        {"type": "node_completed", "node": str, "data": dict}
        {"type": "done", "final_state": dict}
        {"type": "error", "message": str}
    """
    q = Queue()

    def _run_async():
        async def _stream():
            from langchain_core.messages import HumanMessage, AIMessage

            # 构建初始状态，注入历史消息 + 相似病例
            state = create_initial_state("")  # 空状态
            initial_msgs = []

            # 1. FAISS 向量检索相似历史病例（最优先注入，含实体冲突检测）
            try:
                similar = search_similar(query, top_k=3)
                if similar:
                    # 从最近一轮历史消息中提取实体用于冲突检测
                    recent_entities = {}
                    if history_messages:
                        # 取最后一条用户消息的 ent 字段
                        for m in reversed(history_messages):
                            if m.get("role") == "user" and m.get("ent"):
                                try:
                                    recent_entities = eval(m["ent"]) if isinstance(m["ent"], str) else m["ent"]
                                except Exception:
                                    pass
                                break
                    case_text = format_similar_cases(similar, current_entities=recent_entities)
                    if case_text:
                        initial_msgs.append(SystemMessage(content=case_text))
                        print(f"[Vector] 检索到 {len(similar)} 条相似病例")
            except Exception as e:
                print(f"[Vector] 检索失败（不影响主流程）: {e}")

            # 2. 注入历史消息（自动触发长对话压缩）
            if history_messages:
                initial_msgs.extend(_compress_history(history_messages))

            # 3. 当前问题放最后
            # 流水日志：创建本轮日志记录器 + 记录用户输入
            pipeline_log = PipelineLogger(username="default", window_index=0)
            set_pipeline_logger(pipeline_log)
            pipeline_log.new_round(query)
            initial_msgs.append(HumanMessage(content=query))
            state["messages"] = initial_msgs

            # 注入跨轮记忆
            if memory:
                state["knowledge_cache"] = dict(memory)
            prev_node_set = set()

            try:
                async for full_state in app.astream(state, stream_mode="values"):
                    # full_state 是累积完整状态，通过 messages 变化推断当前节点
                    messages = full_state.get("messages", [])
                    entities = full_state.get("raw_entities", {})
                    cache = full_state.get("knowledge_cache", {})
                    next_action = full_state.get("next_action", "")

                    # 根据消息类型和状态特征推断当前完成的节点
                    current_node = _infer_node(messages, entities, cache, next_action, prev_node_set)

                    if current_node and current_node not in prev_node_set:
                        prev_node_set.add(current_node)
                        q.put({
                            "type": "node_completed",
                            "node": current_node,
                            "data": _extract_node_data(current_node, messages, entities, cache),
                        })
                    else:
                        # 同一节点二次经过（如 llm_planner 循环）
                        q.put({
                            "type": "node_completed",
                            "node": current_node or "llm_planner",
                            "data": _extract_node_data(current_node or "llm_planner", messages, entities, cache),
                        })

                # 将本轮病例存入向量索引（用于后续检索相似病例）
                try:
                    final_answer = ""
                    for msg in reversed(full_state.get("messages", [])):
                        if hasattr(msg, "content") and msg.content:
                            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                                final_answer = msg.content
                                break
                    if final_answer and query:
                        add_case(query, final_answer, full_state.get("raw_entities", {}))
                        save_index()  # 持久化到磁盘
                except Exception as e:
                    print(f"[Vector] 保存病例失败: {e}")

                # 流水日志：记录最终回答并保存
                try:
                    for msg in reversed(full_state.get("messages", [])):
                        if hasattr(msg, "content") and msg.content:
                            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                                pipeline_log.log_final_answer(msg.content)
                                break
                    pipeline_log.save()
                    pipeline_log.clear()
                except Exception:
                    pass
                q.put({"type": "done", "final_state": _serializable_state(full_state)})
            except Exception as e:
                q.put({"type": "error", "message": str(e)})
            q.put(None)

        asyncio.run(_stream())

    thread = Thread(target=_run_async, daemon=True)
    thread.start()

    while True:
        try:
            item = q.get(timeout=0.1)
            if item is None:
                break
            yield item
            if item.get("type") in ("done", "error"):
                break
        except Empty:
            continue


def _infer_node(messages: list, entities: dict, cache: dict, next_action: str, prev: set) -> str:
    """根据状态特征推断最近完成的节点。"""
    if not messages:
        return ""

    last_msg = messages[-1] if messages else None

    # preprocess 完成后 raw_entities 被填充，且还没有 AIMessage
    if entities and "preprocess" not in prev:
        has_ai = any(
            (hasattr(m, "tool_calls") and m.tool_calls) or
            (isinstance(m, dict) and m.get("tool_calls"))
            for m in messages
        )
        if not has_ai:
            return "preprocess"

    # tool_executor 完成后会追加 ToolMessage
    if last_msg and isinstance(last_msg, ToolMessage):
        return "tool_executor"

    # llm_planner 完成后会追加 AIMessage
    if last_msg and isinstance(last_msg, AIMessage):
        return "llm_planner"

    return ""


def _extract_node_data(node_name: str, messages: list, entities: dict, cache: dict) -> dict:
    """从累积状态中提取节点相关的展示数据。"""
    data = {}

    if node_name == "preprocess":
        data["entities"] = dict(entities)

    elif node_name == "llm_planner":
        # 找最新的 AIMessage（可能包含 tool_calls 或最终文本）
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    data["tool_calls"] = [
                        {
                            "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                            "name_cn": _TOOL_CN_NAMES.get(
                                tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                                tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                            ),
                            "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                        }
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "content") and msg.content:
                    data["content"] = msg.content
                break

    elif node_name == "tool_executor":
        results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    results.append(json.loads(msg.content))
                except (json.JSONDecodeError, TypeError):
                    results.append({"raw": msg.content})
        data["tool_results"] = results

    return data


def _serializable_state(state: dict) -> dict:
    """将 AgentState 转为可 JSON 序列化的精简版。"""
    messages = state.get("messages", [])
    serialized_msgs = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            serialized_msgs.append({
                "type": "AIMessage",
                "content": getattr(msg, "content", ""),
                "has_tool_calls": bool(getattr(msg, "tool_calls", [])),
            })
        elif isinstance(msg, ToolMessage):
            serialized_msgs.append({
                "type": "ToolMessage",
                "content": getattr(msg, "content", "")[:200],
            })
        elif hasattr(msg, "content"):
            serialized_msgs.append({
                "type": type(msg).__name__,
                "content": msg.content[:200],
            })

    return {
        "messages": serialized_msgs,
        "raw_entities": state.get("raw_entities", {}),
        "knowledge_cache_keys": list(state.get("knowledge_cache", {}).keys()),
        "next_action": state.get("next_action", ""),
    }
