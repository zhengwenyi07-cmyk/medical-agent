"""医疗诊断 Agent 前端 — Streamlit 多窗口问诊界面 (LangGraph ReAct)

本文件是 Agent 系统的前端入口，负责：
  (A) 用户登录/注册/多窗口管理
  (B) Agent 推理步骤的实时可视化展示
  (C) 报告卡片：NER 实体 + 工具调用链路 + 知识图谱查询结果
  (D) 全链路流水日志查看

数据流：用户输入 → stream_agent(query) → 事件生成器 → 动态 UI 渲染
"""

import os
import json
import streamlit as st
import torch  # 仅用于检查 CUDA 可用性

from agent_stream import stream_agent
from conversation_storage import save_conversation, load_all_windows, save_agent_memory, load_agent_memory


# ==========================================
# 全局配置
# ==========================================

def _get_secret(section, key, fallback=None):
    """安全读取 st.secrets，无配置时返回 fallback"""
    try:
        return st.secrets[section][key]
    except Exception:
        return fallback


# ==========================================
# UI/UX 美化 — 临床医疗系统风格
# ==========================================

def load_css():
    """注入全局 CSS 样式，定义医疗系统的视觉体系。

    视觉方向：Clean Clinical（干净、可信赖的现代医疗风格）
    色彩体系：主色 #0D7377（深医青）、背景渐冷灰白、表面纯白
    """
    st.markdown("""
        <style>
        /* ===== 全局基础 ===== */
        .stApp {
            background: linear-gradient(180deg, #F5F9FC 0%, #EEF4F8 100%);
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ===== 主标题 ===== */
        .main-header {
            color: #0D7377;
            font-weight: 700;
            font-size: 1.4em;
            letter-spacing: 0.02em;
            border-bottom: 3px solid #0D7377;
            padding-bottom: 12px;
            margin-bottom: 24px;
            text-align: center;
        }

        /* ===== 侧边栏 ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFC 100%);
            box-shadow: 2px 0 12px rgba(0,0,0,0.04);
            border-right: 1px solid #E3ECF2;
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #0D7377;
        }

        /* ===== 用户卡片 ===== */
        .user-card {
            background: linear-gradient(135deg, #E0F7F6 0%, #D4F0EF 100%);
            padding: 16px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #0D7377;
            color: #1A4A4D;
        }

        /* ===== 聊天消息气泡 ===== */
        .stChatMessage {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            margin-bottom: 12px;
            border: 1px solid #E8EEF2;
            transition: box-shadow 0.2s;
        }
        .stChatMessage:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        /* ===== 知识卡 ===== */
        .medical-card {
            background: #F8FAFB;
            border-left: 4px solid #0D7377;
            padding: 14px 16px;
            border-radius: 0 6px 6px 0;
            margin: 12px 0;
            font-size: 0.9em;
            color: #3A4F5C;
            line-height: 1.6;
        }

        /* ===== 免责声明 ===== */
        .disclaimer {
            font-size: 0.78em;
            color: #8899A6;
            text-align: center;
            margin-top: 18px;
            padding-top: 14px;
            border-top: 1px solid #E8EEF2;
            letter-spacing: 0.02em;
        }

        /* ===== 按钮 ===== */
        .stButton button {
            background-color: #0D7377;
            color: white;
            border-radius: 8px;
            border: none;
            transition: all 0.2s;
            font-weight: 500;
            letter-spacing: 0.01em;
        }
        .stButton button:hover {
            background-color: #095C60;
            color: white;
            border: none;
            box-shadow: 0 2px 6px rgba(13,115,119,0.25);
        }

        /* ===== 意图标签 ===== */
        .intent-badge {
            background-color: #2C4251;
            color: white;
            padding: 3px 10px;
            border-radius: 4px;
            margin: 2px 4px 2px 0;
            font-size: 0.8em;
            display: inline-block;
            letter-spacing: 0.02em;
        }

        /* ===== 全链路日志区域 ===== */
        .pipeline-log-section {
            background: #FAFBFC;
            border: 1px solid #E3ECF2;
            border-radius: 8px;
            padding: 0;
            margin: 24px 0 12px 0;
            max-height: 600px;
            overflow-y: auto;
        }
        .pipeline-log-section .stMarkdown {
            padding: 12px 20px;
        }
        .pipeline-log-section h2 { color: #0D7377; font-size: 1.1em; border-bottom: 1px solid #E3ECF2; padding-bottom: 8px; }
        .pipeline-log-section h3 { color: #2C4251; font-size: 0.95em; }
        .pipeline-log-section h4 { color: #4A6A7D; font-size: 0.88em; }
        .pipeline-log-section code { font-size: 0.82em; background: #F0F4F7; padding: 1px 4px; border-radius: 3px; }
        .pipeline-log-section pre { background: #F5F7F9; border: 1px solid #E3ECF2; border-radius: 6px; }

        /* ===== 输入框 ===== */
        .stChatInput textarea {
            border-radius: 10px;
            border: 1.5px solid #DDE5EC;
        }
        .stChatInput textarea:focus {
            border-color: #0D7377;
            box-shadow: 0 0 0 2px rgba(13,115,119,0.12);
        }
        </style>
    """, unsafe_allow_html=True)


# ==========================================
# 实体渲染
# ==========================================

def render_entities_pretty(entities):
    """将 NER 实体字典渲染为彩色标签 HTML。

    每种实体类型有独立的颜色：
      疾病=红, 疾病症状=橙, 药品=蓝, 检查项目=紫, 科目=青, 食物=绿, 药品商=深灰, 治疗方法=黄

    兼容 str 和 list 两种值格式。

    Args:
        entities: NER 实体 dict 或 str

    Returns:
        HTML 字符串
    """
    if not entities or entities == "{}":
        return "<span style='color:#999; font-size:0.8em;'>未检测到关键医疗实体</span>"
    if isinstance(entities, str):
        try:
            import ast
            entities = ast.literal_eval(entities)
        except Exception:
            return entities
    if not isinstance(entities, dict):
        return str(entities)

    html = ""
    color_map = {
        "疾病": "#E74C3C", "疾病症状": "#E67E22", "药品": "#3498DB",
        "检查项目": "#9B59B6", "科目": "#1ABC9C", "食物": "#2ECC71",
        "药品商": "#34495E", "治疗方法": "#F1C40F"
    }
    for key, value in entities.items():
        color = color_map.get(key, "#95A5A6")
        display_value = ", ".join(value) if isinstance(value, list) else value
        html += (
            f"<span style='background-color: {color}; color: white; padding: 4px 10px; "
            f"border-radius: 12px; margin-right: 5px; font-size: 0.85em; "
            f"display: inline-block; margin-bottom: 5px; "
            f"box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>"
            f"<b>{key}</b>: {display_value}</span>"
        )
    return html


# ==========================================
# 组件预热
# ==========================================

def warmup_agent():
    """预热 Agent 的全部组件，避免首次请求等待过久。

    预热内容（总计约 18s，在页面加载时完成）：
      1. ONNX NER（RoBERTa 分词器 + ONNX Session + AC 自动机 + TF-IDF）~14s
      2. BGE 嵌入模型（SentenceTransformer）~3s
      3. FAISS 向量索引（从磁盘加载或创建）~0.5s
      4. LLM 降级链（仅构建实例，不调 API）<1s

    预热完成后，首次问诊只需等 LLM API 返回（~5s）。
    """
    import time

    # 1. NER 组件
    try:
        t0 = time.time()
        from agent_graph import _load_ner_components
        _load_ner_components()
        print(f"[Warmup] NER 组件就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"NER 预热失败: {e}")

    # 2. BGE 嵌入模型
    try:
        t0 = time.time()
        from vector_memory import _get_model
        _get_model()
        print(f"[Warmup] BGE 嵌入模型就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"BGE 预热失败: {e}")

    # 3. FAISS 索引
    try:
        t0 = time.time()
        from vector_memory import _get_index
        _get_index()
        print(f"[Warmup] FAISS 索引就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"FAISS 预热失败: {e}")

    # 4. LLM 降级链（不调 API，只构建实例列表）
    try:
        t0 = time.time()
        from agent_graph import _get_llm
        _get_llm(with_tools=True)
        print(f"[Warmup] LLM 降级链就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"LLM 预热失败: {e}")


# ==========================================
# 主界面
# ==========================================

def main(is_admin, usname):
    """Streamlit 主界面入口函数。

    由 login3.py 在用户登录成功后调用。

    界面布局：
      侧边栏: 用户信息 + 会话管理 + 系统设置 + 退出
      主区域: 标题 + 聊天历史 + 流式 AI 回答 + 分析报告 + 流水日志

    Args:
        is_admin: 是否为管理员
        usname: 登录用户名
    """
    load_css()

    # ===== 主标题 =====
    st.markdown("<h1 class='main-header'>智能医疗问答与诊疗辅助系统</h1>", unsafe_allow_html=True)

    # ===== 侧边栏 =====
    with st.sidebar:
        # 用户信息卡片
        st.markdown(f"""
        <div class="user-card">
            <h4>欢迎</h4>
            <p><b>用户:</b> {usname if usname else '访客'}</p>
            <p><b>角色:</b> {'管理员' if is_admin else '普通用户'}</p>
        </div>
        """, unsafe_allow_html=True)

        # 会话管理：多窗口支持
        if 'chat_windows' not in st.session_state:
            # 首次加载：从磁盘恢复对话历史
            saved_windows = load_all_windows(usname)
            st.session_state.chat_windows = [list(range(len(w))) for w in saved_windows]  # 占位元数据
            st.session_state.messages = [w for w in saved_windows]
            if not st.session_state.messages:
                st.session_state.messages = [[]]
                st.session_state.chat_windows = [[]]

        st.caption("会话管理")
        col_add, col_del = st.columns([3, 1])
        with col_add:
            if st.button('+ 新建窗口', use_container_width=True):
                st.session_state.chat_windows.append([])
                st.session_state.messages.append([])
                st.rerun()
        with col_del:
            # 至少保留 1 个窗口，防止全部删除
            if st.button('删除', use_container_width=True, disabled=len(st.session_state.chat_windows) <= 1):
                selected = st.session_state.get('_active_window', 0)
                from conversation_storage import delete_conversation
                delete_conversation(usname, selected)
                if len(st.session_state.chat_windows) > 1:
                    st.session_state.chat_windows.pop(selected)
                    st.session_state.messages.pop(selected)
                st.rerun()

        # 窗口切换
        window_options = [f"病例窗口 {i+1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox(
            '切换当前会话:', window_options, label_visibility="collapsed", key='_active_window_select'
        )
        active_window_index = int(selected_window.split()[-1]) - 1
        st.session_state['_active_window'] = active_window_index

        st.divider()

        # 系统设置
        with st.expander("系统设置 & 调试", expanded=True):
            st.caption("结果可视化选项")
            show_ent = st.checkbox("显示实体识别 (NER)", value=True)
            show_int = st.checkbox("显示意图分析", value=True)
            show_prompt = st.checkbox("显示图谱知识", value=False)
            show_pipeline = st.checkbox("查看全链路流水日志", value=False)
            gpu_status = "CUDA GPU 就绪" if torch.cuda.is_available() else "仅 CPU 运行"
            st.caption(f"当前 PyTorch 硬件: {gpu_status}")
            if is_admin:
                st.markdown('[管理知识图谱 (Neo4j)](http://127.0.0.1:7474/)', unsafe_allow_html=True)

        # 退出登录
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()

    # ===== 主内容区域 =====

    # 预热全部组件（首次约 18s，后续毫秒级）
    warmup_agent()

    # 检查是否有未完成的对话（上次崩溃/中断）
    from checkpoint import has_any_unfinished, get_unfinished_checkpoint
    if has_any_unfinished(usname):
        checkpoint = get_unfinished_checkpoint(usname, active_window_index)
        if checkpoint:
            last_node = checkpoint.get("node", "未知")
            st.warning(f"检测到上次对话在「{last_node}」节点中断。如需继续，请重新发送您的问题，Agent 将从检查点恢复。")

    current_messages = st.session_state.messages[active_window_index]

    # ===== 渲染历史消息 =====
    for message in current_messages:
        role = message["role"]
        avatar = "🩺" if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if role == "assistant":
                ent_data = message.get("ent", "")
                yitu_data = message.get("yitu", "")
                prompt_data = message.get("prompt", "")
                # 根据开关显示/隐藏分析报告
                if (show_ent and ent_data) or (show_int and yitu_data) or (show_prompt and prompt_data):
                    st.markdown("---")
                    st.caption("AI 诊疗分析报告")
                    c1, c2 = st.columns(2)
                    if show_ent and ent_data:
                        with c1:
                            st.markdown("**关键医学实体:**")
                            st.markdown(render_entities_pretty(ent_data), unsafe_allow_html=True)
                    if show_int and yitu_data:
                        with c2:
                            st.markdown("**工具调用链路:**")
                            tools_list = yitu_data.split(", ") if isinstance(yitu_data, str) else []
                            intent_html = "".join([f"<span class='intent-badge'>{t}</span>" for t in tools_list])
                            st.markdown(intent_html, unsafe_allow_html=True)
                    if show_prompt and prompt_data:
                        with st.expander("知识图谱查询结果", expanded=True):
                            st.markdown(f'<div class="medical-card">{prompt_data}</div>', unsafe_allow_html=True)
                st.markdown('<div class="disclaimer">AI 生成内容仅供参考，不可替代专业医生诊断。</div>', unsafe_allow_html=True)

    # ===== 处理新输入 —— LangGraph ReAct Agent 流式执行 =====

    # 初始化跨轮记忆（整个会话共享）
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = load_agent_memory(usname)

    if query := st.chat_input("请描述您的症状或问题...", key=f"chat_input_{active_window_index}"):
        # 用户消息
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Agent 回答区域
        with st.chat_message("assistant", avatar="🩺"):
            response_placeholder = st.empty()  # 最终回答占位
            status_placeholder = st.empty()    # 实时状态提示占位

            memory_hint = ""
            if st.session_state.agent_memory:
                memory_hint = f"（已加载 {len(st.session_state.agent_memory)} 条历史知识）"
            status_placeholder.info(f"自主推理引擎启动{memory_hint}，正在分析症状...")

            full_response = ""
            entities = {}
            tools_used = []
            tool_results_summary = []
            error_occurred = False

            # 传递完整历史消息，让 Agent 看到之前的对话上下文
            history = [{"role": m["role"], "content": m["content"]} for m in current_messages]

            # 流式执行 Agent
            for event in stream_agent(
                query,
                memory=st.session_state.agent_memory,
                history_messages=history,
                log_user=usname,
                log_window=active_window_index,
            ):
                t = event.get("type")

                if t == "node_completed":
                    node = event.get("node", "")
                    data = event.get("data", {})

                    if node == "preprocess":
                        # NER 实体抽取完成
                        entities = data.get("entities", {})
                        if entities:
                            ent_desc = "、".join(
                                f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                                for k, v in entities.items()
                            )
                            status_placeholder.info(f"实体识别: {ent_desc}")

                    elif node == "llm_planner":
                        # LLM 推理完成 — 可能出 tool_calls 或最终回答
                        tool_calls = data.get("tool_calls", [])
                        if tool_calls:
                            tc_names = [tc.get("name_cn", tc["name"]) for tc in tool_calls]
                            tools_used.extend(tc_names)
                            status_placeholder.info(f"检索知识图谱: {', '.join(tc_names)}...")
                        elif data.get("content"):
                            status_placeholder.info("综合推理中，正在生成诊断建议...")

                    elif node == "tool_executor":
                        # 工具执行完成 — 提取关键结果用于报告
                        results = data.get("tool_results", [])
                        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
                        status_placeholder.info(f"图谱查询完成 ({success_count}/{len(results)} 成功)，继续推理...")
                        for r in results:
                            if isinstance(r, dict) and r.get("success"):
                                rd = r.get("data", {})
                                if "items" in rd:
                                    items = rd["items"][:5]
                                    tool_results_summary.append(
                                        f"{rd.get('disease', '')} {rd.get('relation_type', '')}: {', '.join(items)}"
                                    )
                                elif "diseases" in rd:
                                    tool_results_summary.append(
                                        f"{rd.get('symptom', '')} -> {', '.join(rd['diseases'][:5])}"
                                    )
                                elif "value" in rd:
                                    tool_results_summary.append(
                                        f"{rd.get('disease', '')} {rd.get('attr_type', '')}: {str(rd['value'])[:80]}"
                                    )

                elif t == "done":
                    # Agent 执行完毕 — 提取最终回答
                    final_state = event.get("final_state", {})
                    serialized_msgs = final_state.get("messages", [])
                    for msg in reversed(serialized_msgs):
                        if msg.get("type") == "AIMessage" and msg.get("content") and not msg.get("has_tool_calls"):
                            full_response = msg["content"]
                            break
                    # 合并知识缓存到跨轮记忆
                    cache_keys = final_state.get("knowledge_cache_keys", [])
                    if cache_keys:
                        st.session_state.agent_memory.update({k: True for k in cache_keys})

                elif t == "error":
                    error_occurred = True
                    full_response = f"抱歉，推理引擎遇到错误：{event.get('message', '')}"
                    status_placeholder.error(full_response)

            # 兜底：如果 LLM 全挂了，给一个友好的提示
            if not full_response and not error_occurred:
                full_response = "抱歉，根据已知信息无法回答该问题，建议咨询专业医生。"

            # 展示最终回答
            response_placeholder.markdown(full_response)
            status_placeholder.empty()

            # ===== 分析报告卡片 =====
            if (show_ent and entities) or (show_int and tools_used) or (show_prompt and tool_results_summary):
                st.markdown("---")
                st.caption("AI 诊疗分析报告")
                c1, c2 = st.columns(2)
                if show_ent and entities:
                    with c1:
                        st.markdown("**关键医学实体 (NER):**")
                        st.markdown(render_entities_pretty(entities), unsafe_allow_html=True)
                if show_int and tools_used:
                    with c2:
                        st.markdown("**Agent 工具调用链路:**")
                        unique_tools = list(dict.fromkeys(tools_used))  # 去重保持顺序
                        intent_html = "".join([f"<span class='intent-badge'>{t}</span>" for t in unique_tools])
                        st.markdown(intent_html, unsafe_allow_html=True)
                if show_prompt and tool_results_summary:
                    with st.expander("知识图谱查询结果", expanded=True):
                        summary_text = "\n\n".join(f"- {s}" for s in tool_results_summary)
                        st.markdown(f'<div class="medical-card">{summary_text}</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="disclaimer">AI 生成内容仅供参考，不可替代专业医生诊断。如遇紧急情况请及时就医。</div>',
                unsafe_allow_html=True,
            )

            # 保存本轮回答到对话历史
            current_messages.append({
                "role": "assistant",
                "content": full_response,
                "yitu": ", ".join(dict.fromkeys(tools_used)),
                "prompt": "\n".join(f"- {s}" for s in tool_results_summary),
                "ent": str(entities),
            })

    # 持久化：保存对话历史到磁盘 + 跨轮记忆
    st.session_state.messages[active_window_index] = current_messages
    save_conversation(usname, active_window_index, current_messages)

    # ===== 全链路流水日志（主内容区全宽展示，不在侧边栏） =====
    if show_pipeline:
        safe_name = usname.replace("/", "_").replace("\\", "_")
        log_path = os.path.join("tmp_data", "pipeline_logs", f"{safe_name}_window{active_window_index}.md")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as _f:
                log_content = _f.read()
            st.divider()
            with st.expander("全链路流水日志（当前窗口）", expanded=False):
                st.markdown(log_content)
        else:
            st.caption("暂无流水日志（请先发送一条消息）")

    # 持久化跨轮记忆
    save_agent_memory(usname, st.session_state.agent_memory)


# ==========================================
# 启动入口
# ==========================================

if __name__ == "__main__":
    # login3.py 在用户登录后设置这些 session_state 并调用 main()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'admin' not in st.session_state:
        st.session_state.admin = False
    if 'usname' not in st.session_state:
        st.session_state.usname = ""

    if not st.session_state.logged_in:
        st.error("请先登录系统")
    else:
        main(st.session_state.admin, st.session_state.usname)
