"""FastAPI 生产级服务端 — 替代 Streamlit 用于并发部署

架构：
  Client (HTTP/WebSocket) → FastAPI → AsyncAgentRunner → LangGraph Agent
                                          │
                                    asyncio 协程并发
                                    单进程支持数百并发

启动方式：
  uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
  (4 个 worker 进程 × N 个协程 = 高并发吞吐)

WebSocket 端点：ws://localhost:8000/ws/chat
  发送: {"query": "头痛吃什么药", "history": [...], "memory": {}}
  接收: {"type": "node_completed", ...} / {"type": "done", ...}

HTTP 端点：
  POST /chat         — 同步请求（等待完整回答）
  POST /chat/stream  — SSE 流式请求
  GET  /health       — 健康检查
"""

# ========== 核心依赖 ==========
import json
import asyncio  # 异步 I/O 框架，FastAPI 的异步端点依赖它
from contextlib import asynccontextmanager  # 用于定义 FastAPI 生命周期上下文管理器

# FastAPI 核心：WebSocket 支持双向流、StreamingResponse 支持 SSE
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel  # 请求体模型校验

# agent_async 中的 AsyncAgentRunner 是对 LangGraph agent 的纯异步封装
# 与 agent_stream.py 不同——agent_stream 用 Queue+Thread 桥接，这里直接用原生 asyncio
from agent_async import AsyncAgentRunner


# ========== 全局 Runner 实例 ==========
# _runner 在 FastAPI 应用生命周期内复用——服务启动时创建，关闭时销毁
# 类型标注为 AsyncAgentRunner | None，初始为 None 避免模块加载时报错
_runner: AsyncAgentRunner = None


# ========== 应用生命周期管理 ==========
@asynccontextmanager  # 将 async generator 转为 FastAPI 可识别的上下文管理器
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期：启动时初始化 Runner，关闭时清理。

    使用方法：
      app = FastAPI(lifespan=lifespan)
      FastAPI 在收到第一个请求前执行 yield 之前的代码（启动逻辑），
      在关闭时执行 yield 之后的代码（清理逻辑）。
    """
    global _runner
    # === 启动阶段 ===
    # 创建 AsyncAgentRunner 实例（内部会初始化 LangGraph 图）
    _runner = AsyncAgentRunner()
    print("[Server] AsyncAgentRunner 已就绪")

    # === 运行阶段 ===
    yield  # 控制权交给 FastAPI，应用正常运行

    # === 关闭阶段 ===
    _runner = None  # 释放引用，让 GC 回收


# ========== FastAPI 应用实例 ==========
# lifespan 参数将上面的生命周期管理器注入应用
app = FastAPI(title="医疗诊断 Agent API", version="2.0", lifespan=lifespan)


# ==========================================
# 请求模型 — 定义接口的输入格式
# ==========================================
# Pydantic BaseModel 自动完成 JSON → Python 对象的转换和参数校验
class ChatRequest(BaseModel):
    """聊天请求体。

    Fields:
        query: 用户输入的自然语言问题（必填）
        history: 之前的对话历史，每项 {"role": "user/assistant", "content": "..."}
        memory: 跨轮知识缓存，来自前端 st.session_state.agent_memory
    """
    query: str                    # 必填字段（无默认值）
    history: list[dict] = []       # 可选，默认为空列表
    memory: dict = {}              # 可选，默认为空字典


# ==========================================
# WebSocket — 双向流式通信（推荐用于前端）
# ==========================================
# WebSocket 与普通 HTTP 的区别：
#   HTTP: 请求-响应，一问一答，连接即关闭
#   WebSocket: 建立后保持长连接，双方都可以随时发送消息，适合实时推送
#
# 本端点用于前端实时展示 Agent 的每一步推理过程。
# 客户端发一条 query，服务端逐步推送每个节点的执行结果。
# ==========================================

@app.websocket("/ws/chat")  # 将函数注册为 /ws/chat 路径的 WebSocket 处理器
async def websocket_chat(ws: WebSocket):
    """WebSocket 端点：接收用户消息，实时推送 Agent 执行事件。

    通信协议（JSON 格式）：

    客户端 → 服务端:
      {"query": "头痛吃什么药", "history": [...], "memory": {...}}
      query: 用户问题
      history: 可选，对话历史列表
      memory: 可选，跨轮知识缓存

    服务端 → 客户端:
      {"type": "node_completed", "node": "preprocess", "data": {"entities": {...}}}
      {"type": "node_completed", "node": "llm_planner", "data": {"tool_calls": [...]}}
      {"type": "done", "final_state": {...}}
      {"type": "error", "message": "错误信息"}

    特点：
      - 一条客户端消息，多条服务端事件推送
      - 事件类型与 agent_stream.py 的 stream_agent 输出格式一致
    """
    # 1. 接受 WebSocket 连接（三次握手完成后执行）
    await ws.accept()
    print("[WS] 客户端已连接")

    try:
        # 2. 循环接收客户端消息（一条连接可以发送多次 query）
        while True:
            raw = await ws.receive_text()  # 等待客户端发送文本消息

            # 3. 解析 JSON 请求体
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                # JSON 格式错误 → 告知客户端但不关闭连接
                await ws.send_json({"type": "error", "message": "无效 JSON"})
                continue  # 继续等待下一条消息

            # 4. 校验必填字段
            query = req.get("query", "")
            if not query:
                await ws.send_json({"type": "error", "message": "query 不能为空"})
                continue

            # 5. 执行 Agent 并逐个推送事件
            try:
                # _runner.run() 返回异步生成器，每完成一个节点就 yield 一个事件
                async for event in _runner.run(
                    query,
                    memory=req.get("memory"),         # 跨轮知识缓存
                    history_messages=req.get("history"), # 对话历史
                ):
                    # 实时推送事件给客户端（前端可据此更新 UI 状态）
                    await ws.send_json(event)

            except Exception as e:
                # Agent 执行异常 → 推送错误事件
                await ws.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        # 客户端主动断开连接（关闭浏览器、刷新页面等）
        # 这是正常的退出路径，不需要额外处理
        print("[WS] 客户端断开")


# ==========================================
# HTTP POST — 同步请求
# ==========================================
# 适用于不需要实时推送的场景，如批量测试、API 调用
# ==========================================

@app.post("/chat")  # 将函数注册为 POST /chat 路径的处理器
async def chat(req: ChatRequest) -> dict:
    """同步端点：等待 Agent 完整执行完毕，返回最终结果。

    与 WebSocket 的区别：
      - 不会逐步推送中间事件
      - 只返回最终的 done 事件（包含完整回答）
      - 适合批量处理或 API 集成场景

    Args:
        req: ChatRequest 对象（FastAPI 自动从 JSON body 反序列化）

    Returns:
        done 事件的 dict，或错误事件的 dict
    """
    final_event = None
    # 遍历所有事件，只保留最后一条 done 事件
    async for event in _runner.run(req.query, req.memory, req.history):
        if event["type"] == "done":
            final_event = event
    # 如果没有 done（异常情况），返回错误
    return final_event or {"type": "error", "message": "无结果"}


# ==========================================
# HTTP SSE — 流式请求（Server-Sent Events）
# ==========================================
# SSE 是一种基于 HTTP 的单向流式传输协议。
# 与 WebSocket 的区别：
#   - WebSocket: 全双工（双向），需要协议升级
#   - SSE: 半双工（服务器→客户端），纯 HTTP，更容易通过防火墙和代理
# ==========================================

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式端点：逐步推送 Agent 执行事件。

    返回格式：text/event-stream，每行以 "data: " 开头。
    客户端（如浏览器的 EventSource）会自动解析并逐个事件回调。

    Args:
        req: ChatRequest 对象

    Returns:
        StreamingResponse，MIME 类型为 text/event-stream
    """
    async def generate():
        """异步生成器：将 Agent 事件转为 SSE 格式的字符串流。

        SSE 格式要求：
          data: <JSON字符串>\n\n
          每个事件以 data: 开头，以两个换行符结尾。
        """
        # _runner.run() 返回异步生成器，逐个产出事件
        async for event in _runner.run(req.query, req.memory, req.history):
            # json.dumps 序列化 + ensure_ascii=False 保留中文
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # StreamingResponse 包装异步生成器，让 FastAPI 以流式方式返回响应
    return StreamingResponse(generate(), media_type="text/event-stream")


# ==========================================
# 健康检查 — 用于负载均衡和服务监控
# ==========================================
# Kubernetes / Docker Compose / Nginx 等编排工具会定期访问此端点
# 来判断服务是否存活。如果返回非 200，编排工具会重启或摘除该实例。
# ==========================================

@app.get("/health")  # GET /health
async def health():
    """健康检查端点。返回服务状态和核心配置信息。"""
    return {"status": "ok", "model": "qwen-max", "framework": "LangGraph + FastAPI"}


# ==========================================
# 启动入口 — 直接运行 python server.py 时使用
# ==========================================
# 注意：生产环境建议用 uvicorn 命令行启动（支持多 worker）
#   uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

if __name__ == "__main__":
    import uvicorn
    # reload=False: 生产环境不启用热重载
    # host="0.0.0.0": 监听所有网络接口（非仅本地）
    # port=8000: 监听端口
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
