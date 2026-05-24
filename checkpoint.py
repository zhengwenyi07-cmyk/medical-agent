"""Agent 检查点机制 — 节点级状态快照与断点续跑

每个节点执行完成后自动保存检查点到磁盘。
进程崩溃后，下次请求时检测未完成的检查点并从中断处恢复。

存储位置: tmp_data/checkpoints/{username}_window{N}.json
"""

import json
import os

CHECKPOINT_DIR = os.path.join("tmp_data", "checkpoints")


def _ensure_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _path(username: str, window_index: int) -> str:
    safe = username.replace("/", "_").replace("\\", "_")
    return os.path.join(CHECKPOINT_DIR, f"{safe}_window{window_index}.json")


def save_checkpoint(username: str, window_index: int, node_name: str, state_snapshot: dict):
    """在节点完成后保存检查点。

    Args:
        username: 用户名
        window_index: 窗口索引
        node_name: 刚完成的节点名（preprocess/llm_planner/tool_executor/reflection）
        state_snapshot: 可序列化的状态快照
    """
    _ensure_dir()
    data = {
        "node": node_name,
        "state": state_snapshot,
        "completed": False,  # Agent 完全执行完毕后设为 True
    }
    with open(_path(username, window_index), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mark_completed(username: str, window_index: int):
    """标记对话已完成（正常结束），清除检查点。"""
    path = _path(username, window_index)
    if os.path.exists(path):
        os.remove(path)


def get_unfinished_checkpoint(username: str, window_index: int) -> dict | None:
    """检查是否有未完成的检查点。

    Returns:
        {"node": str, "state": dict} 或 None（无未完成的检查点）
    """
    path = _path(username, window_index)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("completed") is False:
            return data
    except (json.JSONDecodeError, IOError):
        pass
    return None


def has_any_unfinished(username: str) -> bool:
    """检查用户是否有任何窗口的未完成对话。"""
    _ensure_dir()
    safe = username.replace("/", "_").replace("\\", "_")
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith(safe) and fname.endswith(".json"):
            path = os.path.join(CHECKPOINT_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if not json.load(f).get("completed", True):
                        return True
            except Exception:
                pass
    return False
