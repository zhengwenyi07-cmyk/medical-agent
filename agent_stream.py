"""LangGraph Agent 流式桥接 — 将异步执行桥接到同步生成器供 Streamlit 消费

核心问题：LangGraph 的 app.astream() 是异步生成器（async for），
        但 Streamlit 的 st.chat_input 处理是同步的。
解决方案：后台线程运行 asyncio 事件循环，通过 Queue 将事件从异步世界
        传递到同步世界，主线程以生成器（yield）方式产出给 Streamlit。

主要组件：
  (A) stream_agent()      —— 主入口，同步生成器
  (B) _compress_history() —— 长对话摘要压缩
  (C) _infer_node()       —— 根据状态变化推断当前完成的节点
  (D) _extract_node_data()—— 从状态中提取前端展示数据
  (E) _serializable_state()—— 将 AgentState 转为可 JSON 序列化的精简版
"""
import json
from queue import Queue, Empty  # Queue: 线程安全的队列  Empty: 队列空时抛出的异常
from threading import Thread    # Thread: 后台线程，运行 asyncio 事件循环
import asyncio                  # asyncio: 异步 I/O 框架，LangGraph 的 astream 依赖它

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from agent_graph import app, create_initial_state  # app: 预编译的 LangGraph 实例
from vector_memory import search_similar, add_case, format_similar_cases, save_index
from pipeline_logger import PipelineLogger  # 全链路流水日志记录器
from agent_graph import set_pipeline_logger  # 将日志记录器注入 agent_graph 的模块级变量
from checkpoint import save_checkpoint, mark_completed  # 检查点机制

# 可分辨的节点名称列表（用于 _infer_node 判断）
_NODE_NAMES = {"preprocess", "llm_planner", "tool_executor"}

# 工具名中英文映射 — 前端展示用
# 前端状态栏和报告卡片优先使用中文名，提升用户体验
_TOOL_CN_NAMES = {
    "check_entity_in_kg":      "验证实体存在性",
    "search_symptom_to_disease": "症状反推疾病",
    "get_disease_attr":        "获取疾病属性",
    "get_disease_relations":   "获取疾病关联信息",
}

# 历史压缩阈值：超过此轮数（10 条消息）触发摘要压缩
# 例如 6 轮对话 = 12 条消息 > 10 → 压缩最旧的 1 轮，保留最近 5 轮
_MAX_FULL_ROUNDS = 5


def _compress_history(history_messages: list) -> list:
    """长对话摘要压缩：保留最近轮完整内容，旧轮压缩为结构化摘要。

    当历史消息超过 _MAX_FULL_ROUNDS 轮时：
    1. 取最近 N 轮保留完整 Human/AI 消息（LLM 需要最近的上下文）
    2. 对更早的轮次，提取关键信息（疾病、症状、药品、检查）生成摘要
    3. 摘要作为 SystemMessage 注入，帮助 LLM 了解背景但不占太多 token

    为什么从 ent 字段提取而非调 LLM 做摘要？
      - 零额外 LLM 调用成本
      - ent 字段已经是结构化数据，提取零成本
      - 实体级别的摘要足够 LLM 理解历史背景

    Args:
        history_messages: 完整对话历史列表，每条 msg 包含 role/content/ent 字段

    Returns:
        LangChain Message 列表：摘要 SystemMessage + 最近 N 轮完整消息
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    full_msgs = []
    old_msgs = []

    # 估算总轮数：每 2 条消息 ≈ 1 轮（user + assistant）
    total_rounds = (len(history_messages) + 1) // 2

    # 未超过阈值：不压缩，原样返回
    if total_rounds <= _MAX_FULL_ROUNDS:
        for m in history_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                full_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                full_msgs.append(AIMessage(content=content))
        return full_msgs

    # 超过阈值：取最近 _MAX_FULL_ROUNDS 轮保持完整（10 条消息）
    keep_from = len(history_messages) - (_MAX_FULL_ROUNDS * 2)
    recent_msgs = history_messages[keep_from:]  # 最近 5 轮完整保留
    old_msgs = history_messages[:keep_from]     # 更早的轮次压缩

    # 从旧消息中收集已讨论过的实体信息
    # 例如收集了 {疾病: {"感冒", "头痛"}, 药品: {"布洛芬"}} → 生成摘要
    mentioned_diseases = set()
    mentioned_symptoms = set()
    mentioned_drugs = set()
    mentioned_checks = set()

    for m in old_msgs:
        # ent 字段存储了每轮 NER 抽取的实体，格式如 "{'疾病': '感冒', '药品': '布洛芬'}"
        ent_str = m.get("ent", "{}")
        try:
            # eval 把字符串形式的 dict 转回 Python dict
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
            pass  # 解析失败不影响主流程

    # 拼装摘要文本
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

    # SystemMessage 类型：告诉 LLM 这是"背景知识"而非用户说的话
    summary_msg = SystemMessage(content="\n".join(summary_parts))

    # 组装最终消息列表：摘要放最前 + 最近完整轮次
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




def stream_agent(query: str, memory: dict = None, history_messages: list = None, log_user: str = "default", log_window: int = 0):
    """同步生成器，逐步产出 Agent 执行过程中的结构化事件。

    这是整个 Agent 系统对外的"大门"——Streamlit/FastAPI 都通过这个函数调用 Agent。

    架构：后台线程运行异步事件循环，主线程通过 Queue 接收事件并 yield。

    Args:
        query: 用户当前输入的自然语言问题。
        memory: 可选，前几轮积累的知识缓存 dict，用于跨轮记忆。
        history_messages: 可选，之前的对话历史 [{"role":..., "content":...}, ...]，
            会自动触发长对话摘要压缩（超过5轮时）。
        log_user: 流水日志用的用户名（默认 "default"）
        log_window: 流水日志用的窗口索引（默认 0）

    产出的 dict 类型：
        {"type": "node_completed", "node": str, "data": dict}
        {"type": "done", "final_state": dict}
        {"type": "error", "message": str}
    """
    # Queue: 线程安全的 FIFO 队列，用于异步→同步桥接
    # 后台线程 put 事件，主线程 get 事件
    q = Queue()

    def _run_async():
        """后台线程的执行体：运行 asyncio 事件循环，执行 LangGraph 异步流"""
        async def _stream():
            """真正的异步逻辑：构建初始状态 → 启动 LangGraph → 逐个产出事件"""
            from langchain_core.messages import HumanMessage, AIMessage

            # ===== 第 0 步：构建初始状态 =====
            # create_initial_state("") 创建一个空的 AgentState
            # 所有后续的 messages/entities/cache 都在这个 state 上累积
            state = create_initial_state("")
            initial_msgs = []  # 消息列表将在下方逐步组装

            # ===== 第 1 步：FAISS 向量检索相似历史病例 =====
            # 这是"最优先注入"——相似病例作为 SystemMessage 放在消息列表最前面
            # LLM 看到的第一条消息就是相关历史病例，帮助它快速定位问题方向
            try:
                similar = search_similar(query, top_k=3)
                if similar:
                    # 从最近一轮历史消息中提取实体用于冲突检测
                    # 例如当前用户说"偏头痛"，但历史病例是"感冒"→检测到实体冲突→降权
                    recent_entities = {}
                    if history_messages:
                        # 取最后一条用户消息的 ent 字段（最近一轮的用户实体）
                        for m in reversed(history_messages):
                            if m.get("role") == "user" and m.get("ent"):
                                try:
                                    recent_entities = eval(m["ent"]) if isinstance(m["ent"], str) else m["ent"]
                                except Exception:
                                    pass
                                break
                    # format_similar_cases 将检索到的病例格式化为 LLM 可读的文本
                    # 同时做冲突检测：如果冲突，标注降权
                    case_text = format_similar_cases(similar, current_entities=recent_entities)
                    if case_text:
                        initial_msgs.append(SystemMessage(content=case_text))
                        print(f"[Vector] 检索到 {len(similar)} 条相似病例")
            except Exception as e:
                # 向量检索失败不影响主流程（只是少了参考背景）
                print(f"[Vector] 检索失败（不影响主流程）: {e}")

            # ===== 第 2 步：注入历史消息（自动触发长对话压缩） =====
            # _compress_history 内部判断是否超过 5 轮阈值
            if history_messages:
                initial_msgs.extend(_compress_history(history_messages))

            # ===== 第 3 步：当前问题放最后 =====
            # 流水日志：创建本轮日志记录器 + 记录用户输入
            pipeline_log = PipelineLogger(username=log_user, window_index=log_window)
            set_pipeline_logger(pipeline_log)
            pipeline_log.new_round(query)
            # 当前问题作为最后一条 HumanMessage（LLM 对末尾信息关注度最高）
            initial_msgs.append(HumanMessage(content=query))
            # 所有消息写入 AgentState（add_messages reducer 自动追加）
            state["messages"] = initial_msgs

            # ===== 第 4 步：注入跨轮记忆 =====
            # memory 来自 st.session_state.agent_memory，跨请求持久化
            # 格式：{cache_key: True, ...}，包含之前所有轮次的知识缓存键
            if memory:
                state["knowledge_cache"] = dict(memory)
            prev_node_set = set()  # 用于跟踪已经报告过的节点（避免重复推送）

            # ===== 第 5 步：启动 LangGraph 异步流 =====
            try:
                # stream_mode="values" 确保每次拿到的是**完整累积状态**
                # （而非每个节点的增量更新），这样 _extract_node_data 能提取完整信息
                async for full_state in app.astream(state, stream_mode="values"):
                    # full_state 是累积完整状态，通过 messages 变化推断当前节点
                    messages = full_state.get("messages", [])
                    entities = full_state.get("raw_entities", {})
                    cache = full_state.get("knowledge_cache", {})
                    next_action = full_state.get("next_action", "")

                    # 根据消息类型和状态特征推断当前完成的节点
                    # 例如：最后一条是 ToolMessage → tool_executor 刚完成
                    #      最后一条是 AIMessage → llm_planner 刚完成
                    current_node = _infer_node(messages, entities, cache, next_action, prev_node_set)

                    if current_node and current_node not in prev_node_set:
                        # 首次经过该节点：加入已记录集合，推送事件
                        prev_node_set.add(current_node)
                        q.put({
                            "type": "node_completed",
                            "node": current_node,
                            "data": _extract_node_data(current_node, messages, entities, cache),
                        })
                        # 检查点：保存节点完成后的状态快照
                        try:
                            save_checkpoint(log_user, log_window, current_node,
                                          _serializable_state(full_state))
                        except Exception:
                            pass
                    else:
                        # 同一节点二次经过（如 llm_planner 在 ReAct 循环中被多次调用）
                        # 仍然推送事件（前端需要更新状态），但不加入 prev_node_set
                        q.put({
                            "type": "node_completed",
                            "node": current_node or "llm_planner",
                            "data": _extract_node_data(current_node or "llm_planner", messages, entities, cache),
                        })
                        # 检查点：保存节点完成后的状态快照
                        try:
                            save_checkpoint(log_user, log_window, current_node or "llm_planner",
                                          _serializable_state(full_state))
                        except Exception:
                            pass

                # ===== 第 6 步：Agent 执行完毕后的收尾工作 =====

                # 6a. 将本轮病例存入向量索引（用于后续检索相似病例）
                # 存入的内容：用户问题 + 最终回答 + NER 实体
                try:
                    final_answer = ""
                    # 从消息历史中找最后一条不含 tool_calls 的 AIMessage = 最终回答
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

                # 6b. 流水日志：记录最终回答并保存
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
                # 标记对话正常完成，清除检查点
                try:
                    mark_completed(log_user, log_window)
                except Exception:
                    pass
                # 推送 done 事件：告诉前端"Agent 执行完毕"
                q.put({"type": "done", "final_state": _serializable_state(full_state)})
            except Exception as e:
                # 推送 error 事件：告诉前端"出错了"
                q.put({"type": "error", "message": str(e)})
            # None 作为哨兵信号，告诉主线程"没有更多事件了"
            q.put(None)

        # asyncio.run() 创建新的事件循环，运行 _stream() 直到完成
        asyncio.run(_stream())

    # ===== 启动后台线程 =====
    # daemon=True: 主线程退出时后台线程自动结束（不阻塞程序退出）
    thread = Thread(target=_run_async, daemon=True)
    thread.start()

    # ===== 主线程循环：从 Queue 取事件并 yield =====
    # 这是一个同步生成器，Streamlit 的 for event in stream_agent() 会逐次消费
    while True:
        try:
            # timeout=0.1: 每 100ms 检查一次队列
            # 既能及时响应事件，又不至于空转消耗 CPU
            item = q.get(timeout=0.1)
            if item is None:
                break  # 哨兵信号：后台线程已经推送完所有事件
            yield item  # 产出事件给 Streamlit/调用方
            if item.get("type") in ("done", "error"):
                break  # done/error 之后不再有有效事件
        except Empty:
            continue  # 队列暂时为空，继续等待


# ==========================================
# 节点推断函数
# ==========================================
# LangGraph 的 astream(stream_mode="values") 只返回累积状态，不返回"哪个节点刚完成"。
# _infer_node 通过分析状态变化反推当前完成的节点。
# 推断规则基于一个关键事实：
#   - node_preprocess 完成后：raw_entities 被填充，但还没有 AIMessage
#   - node_llm_planner 完成后：messages 末尾多了一条 AIMessage
#   - node_tool_executor 完成后：messages 末尾多了一条 ToolMessage
# ==========================================

def _infer_node(messages: list, entities: dict, cache: dict, next_action: str, prev: set) -> str:
    """根据状态特征推断最近完成的节点。

    推断优先级（从高到低）：
    1. preprocess  —— entities 非空 且 还没有任何 AIMessage（模型还没推理过）
    2. tool_executor —— 最后一条消息是 ToolMessage（工具刚返回结果）
    3. llm_planner  —— 最后一条消息是 AIMessage（LLM 刚输出）

    Args:
        messages: 累积消息列表
        entities: raw_entities
        cache: knowledge_cache
        next_action: 路由状态
        prev: 已经报告过的节点集合（用于区分首次和循环经过）

    Returns:
        节点名字符串，或空字符串（无法推断）
    """
    if not messages:
        return ""

    last_msg = messages[-1] if messages else None

    # 规则 1：preprocess 完成后 raw_entities 被填充，且还没有 AIMessage
    # "preprocess" not in prev 确保只报告一次（首次经过）
    if entities and "preprocess" not in prev:
        # 进一步确认：没有任何 AIMessage 带 tool_calls（说明 LLM 还没推理过）
        has_ai = any(
            (hasattr(m, "tool_calls") and m.tool_calls) or
            (isinstance(m, dict) and m.get("tool_calls"))
            for m in messages
        )
        if not has_ai:
            return "preprocess"

    # 规则 2：tool_executor 完成后会追加 ToolMessage
    # ToolMessage 是工具执行结果的载体——只要最后一条是 ToolMessage，就是 tool_executor 刚完成
    if last_msg and isinstance(last_msg, ToolMessage):
        return "tool_executor"

    # 规则 3：llm_planner 完成后会追加 AIMessage
    # 无论是带 tool_calls 的 AIMessage 还是最终回答的 AIMessage，都是 llm_planner 的产物
    if last_msg and isinstance(last_msg, AIMessage):
        return "llm_planner"

    return ""


def _extract_node_data(node_name: str, messages: list, entities: dict, cache: dict) -> dict:
    """从累积状态中提取节点相关的展示数据。

    不同节点提取不同的数据：
      - preprocess:    实体列表（用于前端显示 "实体识别: 疾病=头痛"）
      - llm_planner:   tool_calls 列表（用于前端显示 "正在检索知识图谱"）
                       或 content（用于前端显示最终回答）
      - tool_executor: 工具返回结果列表（用于前端显示 "图谱查询完成"）

    Args:
        node_name: 节点名
        messages: 累积消息列表
        entities: raw_entities
        cache: knowledge_cache

    Returns:
        {"entities": {...}} 或 {"tool_calls": [...]} 或 {"content": "..."}
        或 {"tool_results": [...]}
    """
    data = {}

    if node_name == "preprocess":
        # preprocess 的数据就是 NER 抽取的实体
        data["entities"] = dict(entities)

    elif node_name == "llm_planner":
        # 找最新的 AIMessage（可能包含 tool_calls 或最终文本）
        # 从后往前找——最新的 AIMessage 才是当前节点产出的
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                # 如果有 tool_calls：提取调用列表（含中英文名）
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    data["tool_calls"] = [
                        {
                            "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                            # name_cn: 中文名，前端展示用；回退到英文名
                            "name_cn": _TOOL_CN_NAMES.get(
                                tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                                tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                            ),
                            "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                        }
                        for tc in msg.tool_calls
                    ]
                # 如果有 content（纯文本回答）：提取内容
                if hasattr(msg, "content") and msg.content:
                    data["content"] = msg.content
                break

    elif node_name == "tool_executor":
        # 收集所有 ToolMessage 的内容
        # ToolMessage.content 是 JSON 字符串，解析后便于前端展示
        results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    results.append(json.loads(msg.content))
                except (json.JSONDecodeError, TypeError):
                    # JSON 解析失败时保留原始字符串
                    results.append({"raw": msg.content})
        data["tool_results"] = results

    return data


def _serializable_state(state: dict) -> dict:
    """将 AgentState 转为可 JSON 序列化的精简版。

    为什么需要这个函数？
      - AgentState 中包含 LangChain Message 对象，这些对象不能直接 JSON 序列化
      - 前端只需要关键摘要信息（消息类型、内容摘要、实体、缓存键），不需要完整对象
      - 减少网络传输量（完整 state 可能很大）

    Args:
        state: 完整的 AgentState 字典

    Returns:
        精简的、可 JSON 序列化的 dict
    """
    messages = state.get("messages", [])
    serialized_msgs = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            serialized_msgs.append({
                "type": "AIMessage",
                "content": getattr(msg, "content", ""),
                "has_tool_calls": bool(getattr(msg, "tool_calls", [])),  # 只记录有无 tool_calls，不传完整内容
            })
        elif isinstance(msg, ToolMessage):
            serialized_msgs.append({
                "type": "ToolMessage",
                "content": getattr(msg, "content", "")[:200],  # 截断前 200 字符，减少传输量
            })
        elif hasattr(msg, "content"):
            serialized_msgs.append({
                "type": type(msg).__name__,  # 如 "HumanMessage"
                "content": msg.content[:200],
            })

    return {
        "messages": serialized_msgs,
        "raw_entities": state.get("raw_entities", {}),
        "knowledge_cache_keys": list(state.get("knowledge_cache", {}).keys()),  # 只传键名
        "next_action": state.get("next_action", ""),
    }
