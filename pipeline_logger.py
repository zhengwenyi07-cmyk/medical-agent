"""Agent 全链路流水日志记录器 — 可追溯每一步的完整输入输出

记录内容：
  1. 用户原始 Query
  2. NER 实体抽取结果（raw_entities 完整输出）
  3. 注入实体后的完整 System Prompt
  4. 传给 LLM 的完整 messages 列表
  5. LLM 返回的原始 AIMessage（含 tool_calls 或 content）
  6. 每个工具调用的参数 + 返回结果
  7. Reflection 护栏检查结果 + 校验 Agent 输出
  8. 最终回答内容

输出格式：Markdown 文件，每轮对话追加，人类可读。
"""

import json
import os
from datetime import datetime

LOG_DIR = os.path.join("tmp_data", "pipeline_logs")


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


class PipelineLogger:
    """单次请求的流水日志记录器。

    用法:
        logger = PipelineLogger(username="zhengwenyi07", window_index=0)
        logger.log_query("头痛吃什么药")
        logger.log_entities({"疾病症状": ["头痛"]})
        logger.log_messages(messages)
        logger.log_llm_output(aimessage)
        logger.log_tool_call("get_disease_relations", {"disease_name":"感冒","relation_type":"药品"}, result_json)
        logger.log_reflection(guardrail_issues, verification_result)
        logger.save()
    """

    def __init__(self, username: str, window_index: int):
        _ensure_dir()
        self.username = username
        self.window_index = window_index
        self.log_path = os.path.join(LOG_DIR, f"{username}_window{window_index}.md")
        self.buffer = []
        self._round = 0

    def _append(self, text: str):
        self.buffer.append(text)

    def new_round(self, query: str):
        """开始新一轮对话"""
        self._round += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append(f"\n---\n\n## 第 {self._round} 轮对话 | {now}\n")
        self._append(f"### 👤 用户输入\n\n```\n{query}\n```\n")

    def log_entities(self, entities: dict):
        """记录 NER 实体抽取结果"""
        self._append(f"### 🔬 NER 实体抽取\n\n")
        if entities:
            for k, v in entities.items():
                vals = ", ".join(v) if isinstance(v, list) else str(v)
                self._append(f"- **{k}**: {vals}\n")
        else:
            self._append("*(未抽到实体)*\n")
        self._append(f"\n```json\n{json.dumps(entities, ensure_ascii=False, indent=2)}\n```\n")

    def log_system_prompt(self, system_content: str):
        """记录完整的 System Prompt（含实体注入）"""
        self._append(f"### 📝 完整 System Prompt\n\n")
        self._append(f"```\n{system_content}\n```\n")

    def log_messages(self, messages: list):
        """记录传给 LLM 的完整消息列表"""
        self._append(f"### 📨 传给 LLM 的完整消息链（共 {len(messages)} 条）\n\n")
        for i, msg in enumerate(messages):
            role = getattr(msg, "type", type(msg).__name__)
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)

            self._append(f"#### [{i}] {role}\n")
            if content:
                # 截断过长内容
                display = content if len(str(content)) <= 2000 else str(content)[:2000] + "\n...(截断)"
                self._append(f"```\n{display}\n```\n")
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    self._append(f"- 🔧 tool_call: `{name}`\n")
                    self._append(f"  ```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n  ```\n")
            self._append("\n")

    def log_llm_output(self, response, iteration: int = 1):
        """记录 LLM 的原始输出"""
        content = getattr(response, "content", "")
        tool_calls = getattr(response, "tool_calls", None)

        self._append(f"### 🤖 LLM 输出（第 {iteration} 次推理）\n\n")

        if content:
            self._append(f"**content:**\n```\n{content}\n```\n")
        if tool_calls:
            self._append(f"**tool_calls ({len(tool_calls)} 个):**\n\n")
            for tc in tool_calls:
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                self._append(f"#### `{name}`\n")
                self._append(f"```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```\n")
        if not content and not tool_calls:
            self._append("*(空输出)*\n")
        self._append("\n")

    def log_tool_call(self, name: str, args: dict, result_json: str, elapsed: float = None):
        """记录单个工具调用的完整输入输出"""
        self._append(f"#### 🔧 工具调用: `{name}`\n\n")
        self._append(f"**参数:**\n```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```\n")
        if elapsed:
            self._append(f"**耗时:** {elapsed:.3f}s\n\n")
        try:
            result_obj = json.loads(result_json)
            formatted = json.dumps(result_obj, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            formatted = str(result_json)
        self._append(f"**返回结果:**\n```json\n{formatted}\n```\n")

    def log_reflection(self, guardrail_issues: list, verification_result: str, tool_facts: dict = None):
        """记录 Reflection 校验结果"""
        self._append(f"### 🛡️ Reflection 安全校验\n\n")

        self._append(f"**规则护栏:**\n")
        if guardrail_issues:
            for issue in guardrail_issues:
                self._append(f"- {issue}\n")
        else:
            self._append(f"- ✅ 全部通过\n")
        self._append("\n")

        if tool_facts:
            self._append(f"**KG 已知事实（校验依据）:**\n```json\n{json.dumps({k: list(v)[:10] for k, v in tool_facts.items()}, ensure_ascii=False, indent=2)}\n```\n")

        if verification_result:
            self._append(f"**校验 Agent 审核结果:**\n```\n{verification_result[:2000]}\n```\n")

        self._append("\n")

    def log_final_answer(self, answer: str):
        """记录最终回答"""
        self._append(f"### ✅ 最终回答\n\n")
        self._append(f"```\n{answer[:3000]}\n```\n")
        if len(answer) > 3000:
            self._append(f"\n*(共 {len(answer)} 字符，已截断)*\n")

    def log_raw_state_snapshot(self, state: dict, node_name: str):
        """记录节点完成后的原始状态快照"""
        self._append(f"<details><summary>📊 节点 [{node_name}] 状态快照</summary>\n\n")
        # 精简输出
        snap = {
            "raw_entities": state.get("raw_entities", {}),
            "next_action": state.get("next_action", ""),
            "knowledge_cache_keys": list(state.get("knowledge_cache", {}).keys())[:10],
            "messages_count": len(state.get("messages", [])),
        }
        self._append(f"```json\n{json.dumps(snap, ensure_ascii=False, indent=2)}\n```\n")
        self._append("</details>\n\n")

    def save(self):
        """追加写入磁盘文件"""
        _ensure_dir()
        mode = "a" if os.path.exists(self.log_path) else "w"
        with open(self.log_path, mode, encoding="utf-8") as f:
            f.write("".join(self.buffer))

    def clear(self):
        """清空缓冲区（用于新请求）"""
        self.buffer = []
