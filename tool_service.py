"""工具层微服务 — 将 Neo4j 查询独立为 HTTP 服务

架构收益：
  1. 独立扩缩容：工具服务可以单独加实例，不依赖 Agent 推理资源
  2. 技术栈解耦：工具服务可以用 Go/Rust 重写，不绑定 Python
  3. 故障隔离：Neo4j 挂了只影响工具服务，Agent 推理不受影响（有降级策略）
  4. 多 Agent 复用：多个 Agent 实例共享同一个工具服务，减少数据库连接数

启动方式：
  uvicorn tool_service:app --host 0.0.0.0 --port 8001

Agent 侧调用方式：
  # 替代原始 from tools import get_disease_relations
  from tool_service import ToolClient
  client = ToolClient("http://localhost:8001")
  result = client.call("get_disease_relations", {"disease_name": "感冒", "relation_type": "药品"})
"""

import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ==========================================
# 工具注册表（直接从 tools.py 导入）
# ==========================================

def _load_tools():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from tools import (
        check_entity_in_kg,
        search_symptom_to_disease,
        get_disease_attr,
        get_disease_relations,
    )
    return {
        "check_entity_in_kg": check_entity_in_kg,
        "search_symptom_to_disease": search_symptom_to_disease,
        "get_disease_attr": get_disease_attr,
        "get_disease_relations": get_disease_relations,
    }


TOOLS = _load_tools()
print(f"[ToolService] 已注册 {len(TOOLS)} 个工具: {list(TOOLS.keys())}")


# ==========================================
# FastAPI 应用
# ==========================================

app = FastAPI(title="知识图谱工具微服务", version="1.0")


class ToolCallRequest(BaseModel):
    tool_name: str
    args: dict


class ToolCallResponse(BaseModel):
    success: bool
    data: dict = None
    error: str = None


@app.post("/call")
async def call_tool(req: ToolCallRequest) -> dict:
    """执行单个工具调用。

    POST /call
    Body: {"tool_name": "get_disease_relations", "args": {"disease_name": "感冒", "relation_type": "药品"}}
    Response: {"success": true, "data": {"disease": "感冒", "items": [...]}}
    """
    tool = TOOLS.get(req.tool_name)
    if tool is None:
        raise HTTPException(404, f"未知工具: {req.tool_name}。可用: {list(TOOLS.keys())}")

    try:
        if hasattr(tool, "invoke"):
            result_json = tool.invoke(req.args)
        else:
            result_json = tool(**req.args)

        result = json.loads(result_json)
        return result

    except Exception as e:
        raise HTTPException(500, f"工具执行失败: {str(e)}")


@app.post("/batch")
async def batch_call(reqs: list[ToolCallRequest]) -> list[dict]:
    """批量执行工具调用（减少网络往返）。

    POST /batch
    Body: [
      {"tool_name": "get_disease_relations", "args": {...}},
      {"tool_name": "check_entity_in_kg", "args": {...}}
    ]
    """
    results = []
    for req in reqs:
        try:
            tool = TOOLS.get(req.tool_name)
            if tool is None:
                results.append({"success": False, "error": f"未知工具: {req.tool_name}"})
                continue
            raw = tool.invoke(req.args) if hasattr(tool, "invoke") else tool(**req.args)
            results.append(json.loads(raw))
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    return results


@app.get("/tools")
async def list_tools():
    """返回可用工具列表及其参数 Schema"""
    tool_info = {}
    for name, tool in TOOLS.items():
        info = {"name": name}
        if hasattr(tool, "description"):
            info["description"] = tool.description
        if hasattr(tool, "args_schema"):
            info["args_schema"] = str(tool.args_schema)
        tool_info[name] = info
    return {"tools": list(TOOLS.keys()), "details": tool_info}


@app.get("/health")
async def health():
    return {"status": "ok", "tools_loaded": len(TOOLS)}


# ==========================================
# Agent 侧 HTTP 客户端
# ==========================================


class ToolClient:
    """工具微服务的 HTTP 客户端。

    用法：
      client = ToolClient("http://localhost:8001")
      result = client.call("get_disease_relations", {"disease_name": "感冒", "relation_type": "药品"})
      # result ≡ tools.get_disease_relations("感冒", "药品")  的函数调用结果

    或批量调用：
      results = client.batch([
          ("get_disease_relations", {"disease_name": "感冒", "relation_type": "药品"}),
          ("check_entity_in_kg", {"entity_name": "感冒", "entity_type": "疾病"}),
      ])
    """

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call(self, tool_name: str, args: dict) -> dict:
        """调用单个工具"""
        import httpx
        try:
            resp = httpx.post(
                f"{self.base_url}/call",
                json={"tool_name": tool_name, "args": args},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error": f"工具服务调用失败: {str(e)}"}

    def batch(self, calls: list[tuple[str, dict]]) -> list[dict]:
        """批量调用工具"""
        import httpx
        try:
            reqs = [{"tool_name": name, "args": args} for name, args in calls]
            resp = httpx.post(
                f"{self.base_url}/batch",
                json=reqs,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return [{"success": False, "error": f"批量调用失败: {str(e)}"} for _ in calls]

    def health_check(self) -> bool:
        """检查工具服务是否可用"""
        import httpx
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ==========================================
# 启动入口
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tool_service:app", host="0.0.0.0", port=8001, reload=False)
