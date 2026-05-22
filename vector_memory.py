"""FAISS 向量记忆模块 — 语义检索相似历史病例

使用轻量级 BGE 中文嵌入模型 + FAISS IndexFlatIP（内积索引），
将每次问诊记录编码为向量存入索引，新问题到来时检索最相似的历史病例作为参考。

设计理念：
- 轻量级：BGE-small 模型仅 ~100MB，CPU 推理即可
- 本地化：FAISS 纯 CPU 运行，无需外部向量数据库服务
- 语义检索：基于句子嵌入的余弦相似度（内积在归一化后等价于余弦相似度）
"""

import os
import json
import numpy as np
from threading import Lock

# 全局锁，保证多线程安全
_lock = Lock()

# 懒加载的模型和索引
_embedding_model = None
_faiss_index = None
_case_records = []  # [(question, answer, entities_dict, embedding_norm), ...]

STORAGE_DIR = os.path.join("tmp_data", "vector_memory")
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
RECORDS_PATH = os.path.join(STORAGE_DIR, "records.json")


def _get_model():
    """懒加载 BGE 中文嵌入模型。

    BGE (BAAI General Embedding) 是智源研究院推出的中文嵌入模型，
    bge-small-zh-v1.5 为轻量版，仅 ~100MB，512 维输出，
    MTEB 中文榜单排名前列，专为检索任务优化。
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    _embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print("[Vector] BGE 嵌入模型加载完成（512维）")
    return _embedding_model


def _ensure_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def _get_index():
    """加载或创建 FAISS 索引"""
    global _faiss_index, _case_records

    if _faiss_index is not None:
        return _faiss_index

    _ensure_dir()

    if os.path.exists(INDEX_PATH) and os.path.exists(RECORDS_PATH):
        try:
            import faiss
            _faiss_index = faiss.read_index(INDEX_PATH)
            with open(RECORDS_PATH, "r", encoding="utf-8") as f:
                _case_records = json.load(f)
            print(f"[Vector] 从磁盘加载索引: {_faiss_index.ntotal} 条病例")
            return _faiss_index
        except Exception as e:
            print(f"[Vector] 加载索引失败: {e}，将创建新索引")

    # 创建新索引
    import faiss
    dim = 512  # BGE-small-zh-v1.5 输出维度
    _faiss_index = faiss.IndexFlatIP(dim)  # 内积索引（等价于余弦相似度）
    _case_records = []
    print(f"[Vector] 创建新 FAISS 索引，维度: {dim}")
    return _faiss_index


def _embed(text: str) -> np.ndarray:
    """将文本编码为归一化向量（BGE 内置 normalize_embeddings）"""
    model = _get_model()
    # BGE 建议对查询加 instruction 前缀提升检索质量
    emb = model.encode(text, normalize_embeddings=True)
    return emb.astype(np.float32)


def _format_case_text(question: str, answer: str, entities: dict) -> str:
    """将病例格式化为检索用的文本（嵌入时用此文本）"""
    parts = [f"问题: {question}"]
    if entities:
        entity_parts = []
        for k, v in entities.items():
            entity_parts.append(f"{k}:{v}")
        parts.append("实体: " + ", ".join(entity_parts))
    if answer:
        # 截取答案前 200 字作为摘要
        summary = answer[:200] + "..." if len(answer) > 200 else answer
        parts.append(f"回答摘要: {summary}")
    return "\n".join(parts)


def add_case(question: str, answer: str, entities: dict) -> int:
    """新增一条病例到向量索引。

    Args:
        question: 用户问题。
        answer: Agent 的最终回答。
        entities: NER 抽取的实体字典。

    Returns:
        索引中当前的总病例数。
    """
    with _lock:
        index = _get_index()
        case_text = _format_case_text(question, answer, entities)
        vec = _embed(case_text).reshape(1, -1)
        index.add(vec)

        # 保存原始记录用于展示
        _case_records.append({
            "question": question,
            "answer": answer[:500],  # 只存前 500 字
            "entities": entities,
        })

        print(f"[Vector] 新增病例，当前总数: {index.ntotal}")
        return index.ntotal


def search_similar(query: str, top_k: int = 3) -> list:
    """检索与当前问题最相似的历史病例。

    Args:
        query: 当前用户问题。
        top_k: 返回最相似的病例数量。

    Returns:
        [{"question": str, "answer": str, "entities": dict, "score": float}, ...]
        按相似度降序排列。索引为空时返回空列表。
    """
    with _lock:
        index = _get_index()
        if index.ntotal == 0:
            return []

        vec = _embed(query).reshape(1, -1)
        scores, indices = index.search(vec, min(top_k, index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(_case_records):
                rec = _case_records[idx]
                results.append({
                    "question": rec["question"],
                    "answer": rec["answer"],
                    "entities": rec.get("entities", {}),
                    "score": float(score),  # 内积值（归一化后 = 余弦相似度，范围[-1,1]）
                })

        return results


def save_index():
    """将当前索引和病例记录持久化到磁盘。"""
    with _lock:
        index = _get_index()
        if index is None or index.ntotal == 0:
            return
        _ensure_dir()
        import faiss
        faiss.write_index(index, INDEX_PATH)
        with open(RECORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(_case_records, f, ensure_ascii=False, indent=2)
        print(f"[Vector] 索引已保存: {index.ntotal} 条病例")


def format_similar_cases(results: list, min_score: float = 0.3, current_entities: dict = None) -> str:
    """将相似病例列表格式化为可注入 LLM 的上下文字符串。

    Args:
        results: search_similar 返回的结果列表。
        min_score: 最低相似度阈值，低于此值的结果不展示。
        current_entities: 当前 NER 提取的实体 dict，用于冲突检测。

    Returns:
        格式化的上下文字符串，无有效结果时返回空字符串。
    """
    filtered = [r for r in results if r["score"] >= min_score]
    if not filtered:
        return ""

    # 实体冲突检测：标注与当前 NER 结果不一致的病例
    filtered_with_conflict = []
    for r in filtered:
        case_entities = r.get("entities", {})
        conflict_types = _detect_entity_conflicts(case_entities, current_entities or {})
        r["_conflict"] = conflict_types
        r["_conflict_penalty"] = 0.5 if conflict_types else 0.0  # 冲突降权
        filtered_with_conflict.append(r)

    # 按调整后的分数降序排列（有冲突的病例排在后面）
    filtered_with_conflict.sort(key=lambda r: r["score"] - r.get("_conflict_penalty", 0), reverse=True)

    lines = ["[相似历史病例参考] 以下为历史病例，仅供参考——当前诊断请以本次知识图谱查询结果为准："]
    for i, r in enumerate(filtered_with_conflict, 1):
        score = r["score"]
        penalized = score - r.get("_conflict_penalty", 0)

        lines.append(f"\n病例 {i}（相似度: {score:.0%}）")
        lines.append(f"  问题: {r['question']}")
        if r.get("entities"):
            ent_str = ", ".join(f"{k}={v}" for k, v in r["entities"].items())
            lines.append(f"  关键实体: {ent_str}")

        # 冲突标注
        if r["_conflict"]:
            conflict_desc = "、".join(r["_conflict"])
            lines.append(f"  ⚠️ 实体冲突: 该病例中的 [{conflict_desc}] 与当前用户陈述不一致，仅供参考")

        if r.get("answer"):
            lines.append(f"  诊断结论: {r['answer'][:300]}")

    lines.append("\n请注意：以上历史病例的知识可能已过时或与当前用户情况不同。所有医学结论必须以本次知识图谱查询结果为准。如果历史病例与知识图谱查询结果不一致，请以知识图谱为准。")
    return "\n".join(lines)


def _detect_entity_conflicts(case_entities: dict, current_entities: dict) -> list:
    """检测历史病例实体与当前 NER 实体之间的类型冲突。

    规则：同一实体类型下，如果历史病例的值与当前 NER 的值不同，则判定为冲突。

    Args:
        case_entities: 历史病例的实体 dict，如 {"疾病": "感冒", "药品": "布洛芬"}
        current_entities: 当前 NER 抽取的实体 dict，如 {"疾病": "偏头痛"}

    Returns:
        冲突的实体类型列表，如 ["疾病"]。
    """
    conflicts = []
    for etype, current_vals in current_entities.items():
        if etype not in case_entities:
            continue
        # 兼容 str 和 list 两种格式
        case_vals = case_entities[etype]
        if isinstance(case_vals, str):
            case_vals = [case_vals]
        if isinstance(current_vals, str):
            current_vals = [current_vals]

        # 检查是否有交集——如果没有交集，就是冲突
        case_set = set(case_vals)
        current_set = set(current_vals)
        if not case_set.intersection(current_set):
            conflicts.append(etype)
    return conflicts


def get_case_count() -> int:
    """获取当前索引中的病例总数。"""
    index = _get_index()
    return index.ntotal if index else 0
