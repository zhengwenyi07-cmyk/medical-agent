"""
知识图谱交互工具集 —— 为 LangChain Agent 提供标准化的 Neo4j 查询能力
==================================================================
四个工具通过 @tool 装饰器暴露给 LLM, LLM 通过 Function Calling 调用它们:
  1. check_entity_in_kg       — "感冒"在知识图谱里吗? → {exists: true}
  2. search_symptom_to_disease — "头痛"是什么病?     → {diseases: ["感冒","偏头痛",...]}
  3. get_disease_attr         — "感冒"的病因是什么?  → {value: "病毒感染..."}
  4. get_disease_relations    — "感冒"吃什么药?      → {items: ["阿莫西林","布洛芬",...]}

安全设计: 白名单校验 + 参数化 Cypher($占位符), 杜绝注入攻击
返回格式: 统一 JSON {"success": bool, "data"/"error": str}
"""

import json

from neo4j_client import Neo4jClient  # Neo4j 连接池单例(全局一个驱动实例)
from config import get_neo4j_config   # 从 .streamlit/secrets.toml 读取数据库连接信息

# 初始化全局 Neo4j 客户端 —— 整个进程生命周期内复用
_cfg = get_neo4j_config()
client = Neo4jClient(_cfg["uri"], _cfg["user"], _cfg["password"])

# ==========================================
# 辅助函数: 统一 JSON 返回格式
# ==========================================
# 所有工具统一返回 {"success": bool, "data": {...}} 或 {"success": false, "error": "..."}
# 统一格式的好处: LLM 和校验层可以一致地解析所有工具的返回值

def _ok(data: dict) -> str:
    """成功响应"""
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)

def _err(msg: str) -> str:
    """失败响应"""
    return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


# ==========================================
# 白名单定义 —— 安全的基石
# ==========================================
# 所有用户可控的参数(entity_type, relation_type, attr_type)必须先过白名单
# 不在白名单中的值直接拒绝, 不会进入 Cypher 拼接阶段

_VALID_ENTITY_TYPES = {
    "疾病", "药品", "食物", "疾病症状", "检查项目", "科目", "治疗方法", "药品商"
}

_VALID_ATTR_TYPES = {
    "疾病简介", "疾病病因", "疾病预防措施", "疾病治疗周期", "治愈概率", "疾病易感人群"
}

# 关系映射表: LLM 传入的中文参数 → Neo4j 中的关系名 + 目标节点标签
# 例: LLM 说查"药品" → Cypher 中用关系"疾病使用药品" + 目标标签"药品"
_RELATION_MAP = {
    "药品":      {"rel": "疾病使用药品",   "target": "药品"},
    "检查":      {"rel": "疾病所需检查",   "target": "检查项目"},
    "宜吃食物":  {"rel": "疾病宜吃食物",   "target": "食物"},
    "忌吃食物":  {"rel": "疾病忌吃食物",   "target": "食物"},
    "科目":      {"rel": "疾病所属科目",   "target": "科目"},
    "症状":      {"rel": "疾病的症状",     "target": "疾病症状"},
    "治疗方法":  {"rel": "疾病的治疗方法", "target": "治疗方法"},
    "并发疾病":  {"rel": "疾病并发疾病",   "target": "疾病"},
}


# ==========================================
# 工具 1: 实体存在性验证
# ==========================================

def check_entity_in_kg(entity_name: str, entity_type: str) -> str:
    """验证医疗实体是否存在于知识图谱中。

    Args:
        entity_name: 实体名称, 如 "感冒"、"阿莫西林"、"头痛"。
        entity_type: 实体类别, 必须是以下之一: 疾病, 药品, 食物, 疾病症状, 检查项目, 科目, 治疗方法, 药品商

    Returns:
        JSON 字符串。
        例: {"success": true, "data": {"exists": true, "name": "感冒", "type": "疾病"}}
    """
    if not entity_name or not entity_name.strip():
        return _err("entity_name 不能为空")
    if entity_type not in _VALID_ENTITY_TYPES:  # ← 白名单校验
        return _err(f"entity_type 非法: '{entity_type}'。允许值: {_VALID_ENTITY_TYPES}")

    try:
        # $name 参数化查询: Neo4j 自动转义, 杜绝 Cypher 注入
        cypher = f"MATCH (n:`{entity_type}` {{名称: $name}}) RETURN n LIMIT 1"
        rows = client.run_query(cypher, name=entity_name)
        return _ok({"exists": len(rows) > 0, "name": entity_name, "type": entity_type})
    except Exception as e:
        return _err(f"数据库查询失败：{str(e)}")


# ==========================================
# 工具 2: 症状反推疾病(反向关系遍历)
# ==========================================

def search_symptom_to_disease(symptom_name: str) -> str:
    """根据症状名称反推可能的疾病列表。

    Cypher 逻辑: 从症状节点反向遍历 "疾病的症状" 关系, 找到所有关联的疾病节点。

    Args:
        symptom_name: 症状名称, 如 "头痛"、"发烧"、"咳嗽"。

    Returns:
        {"success": true, "data": {"symptom": "头痛", "diseases": ["感冒","偏头痛","脑炎",...]}}
    """
    if not symptom_name or not symptom_name.strip():
        return _err("symptom_name 不能为空")

    try:
        # 反向关系遍历: 疾病节点 -[:疾病的症状]-> 症状节点
        # 已知症状名, 查哪些疾病有这个症状
        cypher = """
            MATCH (d:疾病)-[:`疾病的症状`]->(s:疾病症状 {名称: $name})
            RETURN d.名称 AS disease
        """
        rows = client.run_query(cypher, name=symptom_name)
        return _ok({"symptom": symptom_name, "diseases": [r["disease"] for r in rows]})
    except Exception as e:
        return _err(f"数据库查询失败：{str(e)}")


# ==========================================
# 工具 3: 获取疾病属性(动态属性名查询)
# ==========================================

def get_disease_attr(disease_name: str, attr_type: str) -> str:
    """获取疾病的基础属性信息。

    使用动态属性查询: Cypher 中用反引号包裹中文属性名。

    Args:
        disease_name: 疾病标准中文名, 如 "感冒"、"高血压"。
        attr_type: 属性类别, 必须是: 疾病简介, 疾病病因, 疾病预防措施, 疾病治疗周期, 治愈概率, 疾病易感人群

    Returns:
        {"success": true, "data": {"disease": "感冒", "attr_type": "疾病病因", "value": "70%-80%由病毒引起..."}}
    """
    if not disease_name or not disease_name.strip():
        return _err("disease_name 不能为空")
    if attr_type not in _VALID_ATTR_TYPES:  # ← 白名单校验
        return _err(f"attr_type 非法: '{attr_type}'。允许值: {_VALID_ATTR_TYPES}")

    try:
        # 动态属性名: Neo4j 中 d.\`属性名\` 语法支持中文属性
        cypher = f"MATCH (d:疾病 {{名称: $name}}) RETURN d.`{attr_type}` AS value"
        rows = client.run_query(cypher, name=disease_name)
        value = rows[0]["value"] if rows else None
        return _ok({"disease": disease_name, "attr_type": attr_type, "value": value})
    except Exception as e:
        return _err(f"数据库查询失败：{str(e)}")


# ==========================================
# 工具 4: 获取疾病关联(最常用工具)
# ==========================================

def get_disease_relations(disease_name: str, relation_type: str) -> str:
    """获取疾病与其他实体之间的关联数据。

    支持的查询: 药品(吃什么药)/检查(做什么检查)/宜吃食物/忌吃食物/科目/症状/治疗方法/并发疾病

    Args:
        disease_name: 疾病标准中文名。
        relation_type: 关系类别(药品/检查/宜吃食物/忌吃食物/科目/症状/治疗方法/并发疾病)。

    Returns:
        {"success": true, "data": {"disease": "感冒", "relation_type": "药品",
                                   "items": ["阿莫西林","布洛芬","感冒清热颗粒",...]}}
    """
    if not disease_name or not disease_name.strip():
        return _err("disease_name 不能为空")
    if relation_type not in _RELATION_MAP:  # ← 白名单校验
        return _err(f"relation_type 非法: '{relation_type}'。允许值: {list(_RELATION_MAP.keys())}")

    try:
        mapping = _RELATION_MAP[relation_type]
        # 安全拼接: relation_type 已经过白名单, 它只可能是 8 个枚举值之一
        cypher = f"""
            MATCH (d:疾病 {{名称: $name}})-[:`{mapping['rel']}`]->(t:`{mapping['target']}`)
            RETURN t.名称 AS item
        """
        rows = client.run_query(cypher, name=disease_name)
        return _ok({"disease": disease_name, "relation_type": relation_type, "items": [r["item"] for r in rows]})
    except Exception as e:
        return _err(f"数据库查询失败：{str(e)}")
