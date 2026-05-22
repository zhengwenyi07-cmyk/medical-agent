"""对话历史持久化存储 — 按用户+窗口保存到磁盘，支持跨会话恢复"""
import json
import os

STORAGE_DIR = os.path.join("tmp_data", "conversations")


def _ensure_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def _file_path(username: str, window_index: int) -> str:
    safe_name = username.replace("/", "_").replace("\\", "_")
    return os.path.join(STORAGE_DIR, f"{safe_name}_window{window_index}.json")


def save_conversation(username: str, window_index: int, messages: list):
    """保存某个用户某个窗口的完整对话历史到磁盘。"""
    _ensure_dir()
    serializable = []
    for msg in messages:
        serializable.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
            "yitu": msg.get("yitu", ""),
            "prompt": msg.get("prompt", ""),
            "ent": msg.get("ent", ""),
        })
    with open(_file_path(username, window_index), "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_conversation(username: str, window_index: int) -> list:
    """从磁盘加载对话历史，文件不存在则返回空列表。"""
    path = _file_path(username, window_index)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def load_all_windows(username: str) -> list:
    """加载用户的所有窗口对话历史。返回列表的列表。"""
    windows = []
    i = 0
    while True:
        msgs = load_conversation(username, i)
        if not msgs and i > 0:
            break
        windows.append(msgs)
        i += 1
    return windows if windows else [[]]


def delete_conversation(username: str, window_index: int):
    """删除指定窗口的对话历史文件。"""
    path = _file_path(username, window_index)
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def save_agent_memory(username: str, memory: dict):
    """保存跨轮知识缓存。"""
    _ensure_dir()
    safe_name = username.replace("/", "_").replace("\\", "_")
    path = os.path.join(STORAGE_DIR, f"{safe_name}_memory.json")
    # 只存 keys（布尔标记），实际数据在每轮对话中
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(memory.keys()) if memory else [], f, ensure_ascii=False)


def load_agent_memory(username: str) -> dict:
    """加载跨轮知识缓存。返回 {key: True} 格式。"""
    safe_name = username.replace("/", "_").replace("\\", "_")
    path = os.path.join(STORAGE_DIR, f"{safe_name}_memory.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            keys = json.load(f)
            return {k: True for k in keys}
    except (json.JSONDecodeError, IOError):
        return {}
