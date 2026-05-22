"""工具链沙盒测试 — 不加载任何 LLM，直接验证 4 个工具函数的输入输出"""
import json
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(__file__))

from tools import (
    check_entity_in_kg,
    search_symptom_to_disease,
    get_disease_attr,
    get_disease_relations,
)


def assert_valid_json(response: str, label: str):
    """断言返回值为合法 JSON 字符串"""
    assert isinstance(response, str), f"[{label}] 返回值不是字符串，类型为 {type(response)}"
    try:
        obj = json.loads(response)
    except json.JSONDecodeError as e:
        raise AssertionError(f"[{label}] 返回值不是合法 JSON：{e}")
    return obj


def assert_success(obj: dict, label: str):
    """断言 success 为 True"""
    assert obj.get("success") is True, f"[{label}] success 不为 True：{obj}"


def assert_failure(obj: dict, label: str):
    """断言 success 为 False"""
    assert obj.get("success") is False, f"[{label}] 应返回失败但 success 为 True"


def test_01_check_entity_exists():
    """check_entity_in_kg — 已知存在的实体应返回 exists=True"""
    print("[TEST 01] check_entity_in_kg — 已知实体存在性验证")
    resp = check_entity_in_kg(entity_name="感冒", entity_type="疾病")
    obj = assert_valid_json(resp, "TEST 01")
    assert_success(obj, "TEST 01")
    assert obj["data"]["exists"] is True, f"感冒应存在于图谱中：{obj}"
    print(f"  [PASS] 感冒存在性验证通过：{obj}")


def test_02_check_entity_not_exists():
    """check_entity_in_kg — 不存在的实体应返回 exists=False"""
    print("[TEST 02] check_entity_in_kg — 未知实体不存在验证")
    resp = check_entity_in_kg(entity_name="火星综合症", entity_type="疾病")
    obj = assert_valid_json(resp, "TEST 02")
    assert_success(obj, "TEST 02")
    assert obj["data"]["exists"] is False, f"火星综合症不应存在于图谱中：{obj}"
    print(f"  [PASS]未知实体正确返回不存在：{obj}")


def test_03_check_invalid_entity_type():
    """check_entity_in_kg — 非法 entity_type 应返回失败"""
    print("[TEST 03] check_entity_in_kg — 非法 entity_type 拒绝验证")
    resp = check_entity_in_kg(entity_name="感冒", entity_type="非法类别")
    obj = assert_valid_json(resp, "TEST 03")
    assert_failure(obj, "TEST 03")
    print(f"  [PASS]非法 entity_type 被正确拒绝：{obj}")


def test_04_check_empty_name():
    """check_entity_in_kg — 空名称应返回失败"""
    print("[TEST 04] check_entity_in_kg — 空名称拒绝验证")
    resp = check_entity_in_kg(entity_name="", entity_type="疾病")
    obj = assert_valid_json(resp, "TEST 04")
    assert_failure(obj, "TEST 04")
    print(f"  [PASS]空名称被正确拒绝：{obj}")


def test_05_search_symptom():
    """search_symptom_to_disease — 有效症状应返回疾病列表"""
    print("[TEST 05] search_symptom_to_disease — 症状反推疾病")
    resp = search_symptom_to_disease(symptom_name="头痛")
    obj = assert_valid_json(resp, "TEST 05")
    assert_success(obj, "TEST 05")
    diseases = obj["data"]["diseases"]
    assert isinstance(diseases, list), f"diseases 应为列表：{obj}"
    assert len(diseases) > 0, f"头痛应至少关联一种疾病：{obj}"
    print(f"  [PASS]头痛反推疾病通过（{len(diseases)} 个）：{diseases[:5]}...")


def test_06_search_symptom_no_match():
    """search_symptom_to_disease — 不存在症状应返回空列表"""
    print("[TEST 06] search_symptom_to_disease — 不存在症状返回空列表")
    resp = search_symptom_to_disease(symptom_name="不明所以的症状XYZ")
    obj = assert_valid_json(resp, "TEST 06")
    assert_success(obj, "TEST 06")
    assert obj["data"]["diseases"] == [], f"不存在的症状应返回空列表：{obj}"
    print(f"  [PASS]不存在症状正确返回空列表")


def test_07_get_disease_attr():
    """get_disease_attr — 已知疾病+属性应返回值"""
    print("[TEST 07] get_disease_attr — 获取疾病属性")
    resp = get_disease_attr(disease_name="感冒", attr_type="疾病病因")
    obj = assert_valid_json(resp, "TEST 07")
    assert_success(obj, "TEST 07")
    value = obj["data"]["value"]
    assert value is not None, f"感冒的病因不应为 None：{obj}"
    assert len(value) > 0, f"感冒的病因不应为空字符串：{obj}"
    print(f"  [PASS]疾病属性获取通过：{obj['data']['disease']}.{obj['data']['attr_type']} = {value[:50]}...")


def test_08_get_disease_attr_not_found():
    """get_disease_attr — 不存在的疾病应返回 value=None"""
    print("[TEST 08] get_disease_attr — 不存在疾病返回 None")
    resp = get_disease_attr(disease_name="不存在的疾病XYZ", attr_type="疾病简介")
    obj = assert_valid_json(resp, "TEST 08")
    assert_success(obj, "TEST 08")
    assert obj["data"]["value"] is None, f"不存在的疾病应返回 None：{obj}"
    print(f"  [PASS]不存在疾病正确返回 None")


def test_09_get_disease_attr_invalid_type():
    """get_disease_attr — 非法 attr_type 应返回失败"""
    print("[TEST 09] get_disease_attr — 非法 attr_type 拒绝")
    resp = get_disease_attr(disease_name="感冒", attr_type="非法属性")
    obj = assert_valid_json(resp, "TEST 09")
    assert_failure(obj, "TEST 09")
    print(f"  [PASS]非法 attr_type 被正确拒绝")


def test_10_get_disease_relations():
    """get_disease_relations — 有效疾病+关系应返回物品列表"""
    print("[TEST 10] get_disease_relations — 获取疾病关联（药品）")
    resp = get_disease_relations(disease_name="感冒", relation_type="药品")
    obj = assert_valid_json(resp, "TEST 10")
    assert_success(obj, "TEST 10")
    items = obj["data"]["items"]
    assert isinstance(items, list), f"items 应为列表：{obj}"
    assert len(items) > 0, f"感冒的药品列表不应为空：{obj}"
    print(f"  [PASS]药品关联获取通过（{len(items)} 个）：{items[:5]}...")


def test_11_get_disease_relations_empty():
    """get_disease_relations — 所有 8 种关系类型都应合法且不抛异常"""
    print("[TEST 11] get_disease_relations — 全部 8 种关系类型遍历")
    relation_types = ["药品", "检查", "宜吃食物", "忌吃食物", "科目", "症状", "治疗方法", "并发疾病"]
    for rt in relation_types:
        resp = get_disease_relations(disease_name="感冒", relation_type=rt)
        obj = assert_valid_json(resp, f"TEST 11 [{rt}]")
        assert_success(obj, f"TEST 11 [{rt}]")
        items = obj["data"]["items"]
        assert isinstance(items, list), f"TEST 11 [{rt}] items 应为列表：{obj}"
        print(f"  [PASS]{rt}: {len(items)} 条结果")


def test_12_get_disease_relations_invalid_type():
    """get_disease_relations — 非法 relation_type 应返回失败"""
    print("[TEST 12] get_disease_relations — 非法 relation_type 拒绝")
    resp = get_disease_relations(disease_name="感冒", relation_type="非法关系")
    obj = assert_valid_json(resp, "TEST 12")
    assert_failure(obj, "TEST 12")
    print(f"  [PASS]非法 relation_type 被正确拒绝")


if __name__ == "__main__":
    print("=" * 60)
    print(">>> 医疗知识图谱工具链 -- 沙盒单元测试（无 LLM）")
    print("=" * 60)

    tests = [
        test_01_check_entity_exists,
        test_02_check_entity_not_exists,
        test_03_check_invalid_entity_type,
        test_04_check_empty_name,
        test_05_search_symptom,
        test_06_search_symptom_no_match,
        test_07_get_disease_attr,
        test_08_get_disease_attr_not_found,
        test_09_get_disease_attr_invalid_type,
        test_10_get_disease_relations,
        test_11_get_disease_relations_empty,
        test_12_get_disease_relations_invalid_type,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {e}")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"测试结果：{passed} 通过 / {passed + failed} 总计")
    if failed > 0:
        print(f"[WARN] {failed} 项测试失败，请检查 Neo4j 是否正常运行")
        sys.exit(1)
    else:
        print("[OK] 全部测试通过！工具链可在 Agent 中安全调用。")
