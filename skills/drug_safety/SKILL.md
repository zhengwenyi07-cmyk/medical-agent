# Drug Safety Skill — 药品安全信息查询

## 技能目标
查询美国 FDA 公开数据库（OpenFDA）中药品的禁忌症、副作用、药物相互作用和警告信息。

## 适用场景
- 用户询问某种药是否安全、有什么副作用
- 用户询问某种药的禁忌症（什么人不能吃）
- 用户询问两种药能不能一起吃（药物相互作用）
- 用户询问药品的 FDA 警告信息

## 调用方式
1. 从用户输入中提取药品名称，**优先使用英文通用名**（brand name 或 generic name），如 ibuprofen、aspirin、acetaminophen
2. 调用 `drug_safety` 工具，传入 `drug_name` 参数
3. 根据返回结果向用户说明药品的安全性信息

## 返回格式说明
- `drug`: 查询的药品名
- `contraindications`: 禁忌症（哪些人/情况下不能使用）
- `adverse_reactions`: 不良反应/副作用
- `drug_interactions`: 药物相互作用（与哪些药不能同服）
- `warnings`: FDA 警告信息

## 免责声明
以上信息来自美国 FDA 公开数据库（OpenFDA），仅供参考，不构成医疗建议。请咨询医生或药师。

## 失败处理
- 如果 API 返回未找到：可能药名拼写错误，或该药品未收录在 FDA 数据库中。请尝试使用英文通用名重新查询。
- 如果网络超时或 API 不可用：请稍后重试。
