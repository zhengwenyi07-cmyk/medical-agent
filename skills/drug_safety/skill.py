"""Drug Safety Skill — 查询 OpenFDA 药品安全信息"""
import json
import os
import requests

# 从环境变量或配置读取 API Key（优先环境变量，回退到硬编码）
_OPENFDA_API_KEY = os.environ.get(
    "OPENFDA_API_KEY",
    "wR8ApSiP4U1uH9hldsWs0ZPpptj7kgZBiSDHoy79"
)

# 常见药的静态降级数据（API 不可用时兜底）
_STATIC_FALLBACK = {
    "ibuprofen": {
        "contraindications": "对布洛芬过敏者禁用；活动性消化性溃疡患者禁用；严重心力衰竭患者禁用；妊娠晚期禁用。",
        "adverse_reactions": "常见：胃肠道不适（恶心、腹痛、腹泻）、头晕、头痛。少见：皮疹、肾功能损害、消化道出血。",
        "drug_interactions": "与抗凝血药（如华法林）合用增加出血风险；与ACE抑制剂合用降低降压效果；与甲氨蝶呤合用增加毒性。",
        "warnings": "心血管风险：长期大剂量使用增加心梗和卒中风险。胃肠道风险：可引起消化道溃疡、出血和穿孔。",
    },
    "aspirin": {
        "contraindications": "对阿司匹林过敏者禁用；活动性消化道溃疡禁用；血友病禁用；妊娠晚期禁用。",
        "adverse_reactions": "常见：胃肠道不适、出血倾向增加。少见：哮喘发作、耳鸣、肝功能异常。",
        "drug_interactions": "与抗凝血药合用显著增加出血风险；与布洛芬合用降低阿司匹林心脏保护作用；与甲氨蝶呤合用增加毒性。",
        "warnings": "儿童和青少年病毒感染期间使用可能引起Reye综合征。手术前应停用。",
    },
    "acetaminophen": {
        "contraindications": "对乙酰氨基酚过敏者禁用；严重肝功能损害者禁用。",
        "adverse_reactions": "常规剂量下安全性较好。过量可导致严重肝损伤（超过4g/日）。",
        "drug_interactions": "与酒精合用增加肝毒性风险；与华法林合用长期可能增加出血风险。",
        "warnings": "每日最大剂量不超过4g。含多种感冒药中可能含对乙酰氨基酚成分，需注意避免重复用药。",
    },
}


def _safe_extract(field_data) -> str:
    """安全提取 FDA 返回字段（可能是字符串或列表），合并为可读文本"""
    if not field_data:
        return "暂无数据"
    if isinstance(field_data, str):
        return field_data
    if isinstance(field_data, list):
        return "\n".join(str(item) for item in field_data[:10])
    return str(field_data)


def run(drug_name: str) -> str:
    """查询 OpenFDA 药品安全信息。

    Args:
        drug_name: 药品英文通用名，如 ibuprofen、aspirin

    Returns:
        JSON 字符串，格式 {"success": bool, "drug": str, "contraindications": str, ...}
    """
    # 输入校验
    if not drug_name or not isinstance(drug_name, str):
        return json.dumps({"success": False, "error": "药物名称不能为空"}, ensure_ascii=False)
    drug_name = drug_name.strip().lower()
    if len(drug_name) > 200:
        return json.dumps({"success": False, "error": "药物名称过长"}, ensure_ascii=False)

    # 尝试 API 查询
    api_key = _OPENFDA_API_KEY
    url = "https://api.fda.gov/drug/label.json"
    params = {
        "api_key": api_key,
        "search": f"openfda.brand_name.exact:{drug_name}",
        "limit": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("results", [{}])[0]
            openfda = result.get("openfda", {})

            return json.dumps({
                "success": True,
                "drug": drug_name,
                "brand_name": ", ".join(openfda.get("brand_name", [drug_name])),
                "generic_name": ", ".join(openfda.get("generic_name", ["未知"])),
                "contraindications": _safe_extract(result.get("contraindications")),
                "adverse_reactions": _safe_extract(result.get("adverse_reactions")),
                "drug_interactions": _safe_extract(result.get("drug_interactions")),
                "warnings": _safe_extract(result.get("warnings")),
            }, ensure_ascii=False)

        elif resp.status_code == 404:
            # FDA 未收录 → 尝试静态字典降级
            if drug_name in _STATIC_FALLBACK:
                fb = _STATIC_FALLBACK[drug_name]
                return json.dumps({
                    "success": True,
                    "drug": drug_name,
                    "brand_name": drug_name,
                    "generic_name": drug_name,
                    "contraindications": fb.get("contraindications", "暂无"),
                    "adverse_reactions": fb.get("adverse_reactions", "暂无"),
                    "drug_interactions": fb.get("drug_interactions", "暂无"),
                    "warnings": fb.get("warnings", "暂无"),
                    "_source": "static_fallback",
                }, ensure_ascii=False)
            return json.dumps({
                "success": False,
                "error": f"FDA 数据库中未找到药品 '{drug_name}'。请检查药名拼写，或尝试使用英文通用名。",
            }, ensure_ascii=False)

        else:
            return json.dumps({
                "success": False,
                "error": f"FDA API 返回错误 (HTTP {resp.status_code})，请稍后重试。",
            }, ensure_ascii=False)

    except requests.Timeout:
        # 超时 → 静态字典降级
        if drug_name in _STATIC_FALLBACK:
            fb = _STATIC_FALLBACK[drug_name]
            return json.dumps({
                "success": True, "drug": drug_name, "_source": "static_fallback",
                "contraindications": fb.get("contraindications", "暂无"),
                "adverse_reactions": fb.get("adverse_reactions", "暂无"),
                "drug_interactions": fb.get("drug_interactions", "暂无"),
                "warnings": fb.get("warnings", "暂无"),
            }, ensure_ascii=False)
        return json.dumps({"success": False, "error": "FDA API 请求超时，请稍后重试。"}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": f"查询异常: {str(e)}"}, ensure_ascii=False)
