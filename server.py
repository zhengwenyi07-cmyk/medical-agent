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

import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_async import AsyncAgentRunner


# 全局 Runner 实例（应用生命周期内复用）
_runner: AsyncAgentRunner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runner
    _runner = AsyncAgentRunner()
    print("[Server] AsyncAgentRunner 已就绪")
    yield
    _runner = None


app = FastAPI(title="医疗诊断 Agent API", version="2.0", lifespan=lifespan)


# ==========================================
# 请求模型
# ==========================================

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    memory: dict = {}


# ==========================================
# WebSocket — 双向流式通信（推荐用于前端）
# ==========================================

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """WebSocket 端点：接收用户消息，实时推送 Agent 执行事件。

    客户端发送 JSON:
      {"query": "头痛吃什么药", "history": [...], "memory": {...}}

    服务端推送 JSON 事件:
      {"type": "node_completed", "node": "preprocess", "data": {...}}
      {"type": "done", "final_state": {...}}
    """
    await ws.accept()
    print("[WS] 客户端已连接")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "无效 JSON"})
                continue

            query = req.get("query", "")
            if not query:
                await ws.send_json({"type": "error", "message": "query 不能为空"})
                continue

            try:
                async for event in _runner.run(
                    query,
                    memory=req.get("memory"),
                    history_messages=req.get("history"),
                ):
                    await ws.send_json(event)
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        print("[WS] 客户端断开")


# ==========================================
# HTTP POST — 同步请求
# ==========================================

@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """同步端点：等待 Agent 完整执行完毕，返回最终结果。"""
    final_event = None
    async for event in _runner.run(req.query, req.memory, req.history):
        if event["type"] == "done":
            final_event = event
    return final_event or {"type": "error", "message": "无结果"}


# ==========================================
# HTTP SSE — 流式请求（Server-Sent Events）
# ==========================================

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式端点：逐步推送 Agent 执行事件。"""
    async def generate():
        async for event in _runner.run(req.query, req.memory, req.history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


# ==========================================
# 健康检查
# ==========================================

@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen-max", "framework": "LangGraph + FastAPI"}


# ==========================================
# 启动入口
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
