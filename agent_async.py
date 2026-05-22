"""纯异步 Agent 运行器 — 用 asyncio 原生协程替代 Queue+Thread

对比 agent_stream.py：
- agent_stream.py: Thread + Queue → 每个请求占用一个线程，GIL 限制并发
- agent_async.py:    asyncio 协程 → 每个请求一个协程，单线程高并发

用法：
    async with AsyncAgentRunner() as runner:
        async for event in runner.run("头痛吃什么药"):
            print(event)
"""

import json
import asyncio
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage
from agent_graph import app, create_initial_state
from vector_memory import search_similar, add_case, format_similar_cases, save_index


# 工具名中英文映射（同 agent_stream.py）
_TOOL_CN_NAMES = {
    "check_entity_in_kg": "验证实体存在性",
    "search_symptom_to_disease": "症状反推疾病",
    "get_disease_attr": "获取疾病属性",
    "get_disease_relations": "获取疾病关联信息",
}

_MAX_FULL_ROUNDS = 5  # 压缩阈值


class AsyncAgentRunner:
    """纯异步 Agent 执行器，支持并发请求。

    使用 asyncio 协程替代 threading.Thread，单线程即可处理多个并发请求。
    每个请求是一个独立的协程，通过 asyncio.gather() 并发执行。

    并发性能对比：
    - agent_stream.py: 10 并发 = 10 线程，GIL 竞争 + 上下文切换开销
    - agent_async.py:   10 并发 = 10 协程（1 线程），零 GIL 开销
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query: str, memory: dict = None, history_messages: list = None) -> AsyncIterator[dict]:
        """异步执行 Agent，逐步产出事件。

        Args:
            query: 用户问题。
            memory: 跨轮知识缓存。
            history_messages: 历史对话列表。

        Yields:
            {"type": "node_completed", "node": str, "data": dict}
            {"type": "done", "final_state": dict}
            {"type": "error", "message": str}
        """
        state = create_initial_state("")
        initial_msgs = []

        # 1. FAISS 相似病例检索（不阻塞事件循环，含实体冲突检测）
        try:
            similar = await asyncio.to_thread(search_similar, query, 3)
            if similar:
                # 从最近一轮历史消息中提取实体用于冲突检测
                recent_entities = {}
                if history_messages:
                    for m in reversed(history_messages):
                        if m.get("role") == "user" and m.get("ent"):
                            try:
                                recent_entities = eval(m["ent"]) if isinstance(m["ent"], str) else m["ent"]
                            except Exception:
                                pass
                            break
                case_text = format_similar_cases(similar, current_entities=recent_entities)
                if case_text:
                    from langchain_core.messages import SystemMessage
                    initial_msgs.append(SystemMessage(content=case_text))
        except Exception as e:
            print(f"[Async] 向量检索失败（非致命）: {e}")

        # 2. 历史消息压缩
        if history_messages:
            compressed = await asyncio.to_thread(_compress_history_sync, history_messages)
            initial_msgs.extend(compressed)

        # 3. 当前问题
        initial_msgs.append(HumanMessage(content=query))
        state["messages"] = initial_msgs

        if memory:
            state["knowledge_cache"] = dict(memory)

        prev_nodes = set()

        try:
            async for full_state in app.astream(state, stream_mode="values"):
                messages = full_state.get("messages", [])
                entities = full_state.get("raw_entities", {})
                cache = full_state.get("knowledge_cache", {})
                current_node = _infer_node(messages, entities, prev_nodes)

                if current_node and current_node not in prev_nodes:
                    prev_nodes.add(current_node)
                else:
                    current_node = current_node or "llm_planner"

                yield {
                    "type": "node_completed",
                    "node": current_node,
                    "data": _extract_data(current_node, messages, entities, cache),
                }

            # 保存病例到向量索引（不阻塞）
            final_answer = _get_final_answer(full_state.get("messages", []))
            if final_answer and query:
                await asyncio.to_thread(add_case, query, final_answer, full_state.get("raw_entities", {}))
                await asyncio.to_thread(save_index)

            yield {"type": "done", "final_state": _serializable_state(full_state)}

        except Exception as e:
            yield {"type": "error", "message": str(e)}


def _compress_history_sync(history_messages: list) -> list:
    """同步版历史压缩（供 asyncio.to_thread 调用）"""
    from langchain_core.messages import SystemMessage

    total = (len(history_messages) + 1) // 2
    if total <= _MAX_FULL_ROUNDS:
        result = []
        for m in history_messages:
            if m.get("role") == "user":
                result.append(HumanMessage(content=m.get("content", "")))
            else:
                result.append(AIMessage(content=m.get("content", "")))
        return result

    keep = history_messages[-(_MAX_FULL_ROUNDS * 2):]
    old = history_message[:-(_MAX_FULL_ROUNDS * 2)]

    diseases = set(); drugs = set(); symptoms = set()
    for m in old:
        ent_str = m.get("ent", "{}")
        try:
            ent = eval(ent_str) if ent_str and ent_str != "{}" else {}
            for k, v in ent.items():
                (diseases if k == "疾病" else drugs if k == "药品" else symptoms).add(v)
        except Exception:
            pass

    parts = ["[历史对话摘要]"]
    if diseases: parts.append(f"- 讨论过的疾病: {', '.join(sorted(diseases))}")
    if symptoms: parts.append(f"- 症状: {', '.join(sorted(symptoms))}")
    if drugs: parts.append(f"- 药品: {', '.join(sorted(drugs))}")

    result = [SystemMessage(content="\n".join(parts))]
    for m in keep:
        if m.get("role") == "user":
            result.append(HumanMessage(content=m.get("content", "")))
        else:
            result.append(AIMessage(content=m.get("content", "")))
    return result


def _infer_node(messages, entities, prev):
    if not messages: return ""
    last = messages[-1]
    if entities and "preprocess" not in prev:
        if not any(hasattr(m, "tool_calls") and m.tool_calls for m in messages if hasattr(m, "tool_calls")):
            return "preprocess"
    from langchain_core.messages import ToolMessage
    if isinstance(last, ToolMessage): return "tool_executor"
    if isinstance(last, AIMessage): return "llm_planner"
    return ""


def _extract_data(node, messages, entities, cache):
    data = {}
    if node == "preprocess":
        data["entities"] = dict(entities)
    elif node == "llm_planner":
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    data["tool_calls"] = [
                        {"name": tc.get("name","") if isinstance(tc,dict) else getattr(tc,"name",""),
                         "name_cn": _TOOL_CN_NAMES.get(
                             tc.get("name","") if isinstance(tc,dict) else getattr(tc,"name",""), ""),
                         "args": tc.get("args",{}) if isinstance(tc,dict) else getattr(tc,"args",{})}
                        for tc in msg.tool_calls]
                if hasattr(msg, "content") and msg.content:
                    data["content"] = msg.content
                break
    elif node == "tool_executor":
        from langchain_core.messages import ToolMessage
        results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try: results.append(json.loads(msg.content))
                except: results.append({"raw": msg.content})
        data["tool_results"] = results
    return data


def _get_final_answer(messages):
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                return msg.content
    return ""


def _serializable_state(state):
    msgs = []
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage):
            msgs.append({"type": "AIMessage", "content": getattr(msg,"content",""),
                         "has_tool_calls": bool(getattr(msg,"tool_calls",[]))})
        elif hasattr(msg, "content"):
            msgs.append({"type": type(msg).__name__, "content": msg.content[:200]})
    return {"messages": msgs, "raw_entities": state.get("raw_entities",{}),
            "knowledge_cache_keys": list(state.get("knowledge_cache",{}).keys()),
            "next_action": state.get("next_action","")}


# ==========================================
# 并发运行入口：asyncio.run()
# ==========================================

async def run_concurrent(queries: list[str]) -> list[dict]:
    """并发执行多个 Agent 请求。

    所有请求在同一个事件循环中并发执行，每个请求一个协程。
    """
    async with AsyncAgentRunner() as runner:
        tasks = [_collect(runner.run(q)) for q in queries]
        return await asyncio.gather(*tasks)


async def _collect(stream: AsyncIterator[dict]) -> dict:
    """收集一个流的所有事件，返回最终状态"""
    final = None
    async for event in stream:
        if event["type"] == "done":
            final = event["final_state"]
    return final
