"""Drug Safety Skill — 查询 OpenFDA 药品安全信息"""
import json
import os
import requests

# 从环境变量或配置读取 API Key（优先环境变量，回退到硬编码）
_OPENFDA_API_KEY = os.environ.get(
    "OPENFDA_API_KEY",
    "wR8ApSiP4U1uH9hldsWs0ZPpptj7kgZBiSDHoy79"
)

# 中英文药名映射表（中文→英文通用名）
# 覆盖中国常见药品，解决中文用户输入无法直接查询 OpenFDA 的问题
_CN_TO_EN_DRUG = {
    # 解热镇痛药
    "布洛芬": "ibuprofen", "芬必得": "ibuprofen", "美林": "ibuprofen",
    "阿司匹林": "aspirin", "拜阿司匹林": "aspirin", "巴米尔": "aspirin",
    "对乙酰氨基酚": "acetaminophen", "扑热息痛": "acetaminophen",
    "泰诺": "acetaminophen", "必理通": "acetaminophen",
    "萘普生": "naproxen", "双氯芬酸": "diclofenac",
    "塞来昔布": "celecoxib", "西乐葆": "celecoxib",
    "吲哚美辛": "indomethacin", "消炎痛": "indomethacin",
    # 抗生素
    "阿莫西林": "amoxicillin", "头孢克洛": "cefaclor",
    "头孢拉定": "cefradine", "头孢氨苄": "cephalexin",
    "阿奇霉素": "azithromycin", "红霉素": "erythromycin",
    "左氧氟沙星": "levofloxacin", "环丙沙星": "ciprofloxacin",
    "甲硝唑": "metronidazole", "替硝唑": "tinidazole",
    "青霉素": "penicillin", "四环素": "tetracycline",
    "多西环素": "doxycycline", "克林霉素": "clindamycin",
    # 心血管
    "阿托伐他汀": "atorvastatin", "立普妥": "atorvastatin",
    "瑞舒伐他汀": "rosuvastatin", "辛伐他汀": "simvastatin",
    "美托洛尔": "metoprolol", "比索洛尔": "bisoprolol",
    "氨氯地平": "amlodipine", "硝苯地平": "nifedipine",
    "氯沙坦": "losartan", "缬沙坦": "valsartan",
    "厄贝沙坦": "irbesartan", "替米沙坦": "telmisartan",
    "卡托普利": "captopril", "依那普利": "enalapril",
    "华法林": "warfarin", "氯吡格雷": "clopidogrel",
    # 糖尿病
    "二甲双胍": "metformin", "格列美脲": "glimepiride",
    "胰岛素": "insulin", "阿卡波糖": "acarbose",
    # 消化系统
    "奥美拉唑": "omeprazole", "兰索拉唑": "lansoprazole",
    "雷贝拉唑": "rabeprazole", "泮托拉唑": "pantoprazole",
    "铝碳酸镁": "hydrotalcite", "多潘立酮": "domperidone",
    "莫沙必利": "mosapride", "蒙脱石": "montmorillonite",
    # 抗过敏
    "氯雷他定": "loratadine", "西替利嗪": "cetirizine",
    "扑尔敏": "chlorpheniramine", "非索非那定": "fexofenadine",
    # 呼吸系统
    "氨溴索": "ambroxol", "右美沙芬": "dextromethorphan",
    "沙丁胺醇": "albuterol", "布地奈德": "budesonide",
    # 其他常见药
    "甲氨蝶呤": "methotrexate", "泼尼松": "prednisone",
    "地塞米松": "dexamethasone", "异烟肼": "isoniazid",
    "利福平": "rifampin", "氟康唑": "fluconazole",
    "阿昔洛韦": "acyclovir", "奥司他韦": "oseltamivir",
    "达菲": "oseltamivir",
}
# 反向映射：英文名 → 中文名（用于结果展示）
_EN_TO_CN_DRUG = {v: k for k, v in _CN_TO_EN_DRUG.items() if not any(
    c in k for c in ['芬必得', '美林', '拜阿司匹林', '巴米尔', '泰诺', '必理通', '西乐葆', '消炎痛', '立普妥', '扑尔敏', '扑热息痛', '达菲']
)}


def _to_english_drug_name(name: str) -> str:
    """将中文药名转换为英文通用名。

    如果输入已经是英文，直接返回小写形式。
    如果是已知中文药名，返回对应的英文通用名。
    如果无法转换，返回原始输入（让 API 尝试匹配）。

    Args:
        name: 用户输入的药名（中文或英文）

    Returns:
        英文通用名（小写）
    """
    name = name.strip()
    # 如果已经是英文（纯ASCII），直接返回小写
    if all(ord(c) < 128 for c in name):
        return name.lower()
    # 查中→英映射表
    return _CN_TO_EN_DRUG.get(name, name.lower())


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
    drug_name = drug_name.strip()
    if len(drug_name) > 200:
        return json.dumps({"success": False, "error": "药物名称过长"}, ensure_ascii=False)

    # 中→英转换（中文用户输入 → 英文通用名）
    original_name = drug_name
    drug_name_en = _to_english_drug_name(drug_name)
    if drug_name_en != original_name.lower():
        print(f"[DrugSafety] 中文药名转换: {original_name} → {drug_name_en}")

    drug_name = drug_name_en.lower()

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
                "drug_cn": original_name if original_name != drug_name else _EN_TO_CN_DRUG.get(drug_name, ""),
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

    except (requests.Timeout, requests.ConnectionError, requests.exceptions.SSLError,
            requests.exceptions.ProxyError, OSError) as e:
        # 网络问题（超时/SSL/代理/连接拒绝）→ 静态字典降级
        error_type = type(e).__name__
        if drug_name in _STATIC_FALLBACK:
            fb = _STATIC_FALLBACK[drug_name]
            return json.dumps({
                "success": True,
                "drug": drug_name,
                "drug_cn": original_name if original_name != drug_name else _EN_TO_CN_DRUG.get(drug_name, ""),
                "_source": "static_fallback",
                "_network_error": error_type,
                "contraindications": fb.get("contraindications", "暂无"),
                "adverse_reactions": fb.get("adverse_reactions", "暂无"),
                "drug_interactions": fb.get("drug_interactions", "暂无"),
                "warnings": fb.get("warnings", "暂无"),
            }, ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": f"FDA API 不可用（{error_type}），且该药品无离线缓存。请稍后重试或尝试查询其他常见药品。",
        }, ensure_ascii=False)

    except Exception as e:
        # 其他异常也尝试降级
        error_type = type(e).__name__
        if drug_name in _STATIC_FALLBACK:
            fb = _STATIC_FALLBACK[drug_name]
            return json.dumps({
                "success": True, "drug": drug_name, "_source": "static_fallback",
                "_network_error": error_type,
                "contraindications": fb.get("contraindications", "暂无"),
                "adverse_reactions": fb.get("adverse_reactions", "暂无"),
                "drug_interactions": fb.get("drug_interactions", "暂无"),
                "warnings": fb.get("warnings", "暂无"),
            }, ensure_ascii=False)
        return json.dumps({"success": False, "error": f"查询异常: {str(e)}"}, ensure_ascii=False)
