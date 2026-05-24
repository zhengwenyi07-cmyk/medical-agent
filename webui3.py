"""增强版医疗问答系统 - 自主推理诊断 Agent v4 (LangGraph ReAct)"""
import os
import re
import ast
import json
import streamlit as st
import torch
import py2neo

from agent_stream import stream_agent
from conversation_storage import save_conversation, load_all_windows, save_agent_memory, load_agent_memory

# ==========================================
# 全局配置（实体过滤关闭，避免调用外部 API）
# ==========================================
ENABLE_ENTITY_FILTER = False   # 关闭千问实体过滤，避免不稳定和费用

# ---------- 从 Streamlit Secrets 读取敏感配置 ----------
def _get_secret(section, key, fallback=None):
    """安全读取 st.secrets，无配置时返回 fallback"""
    try:
        return st.secrets[section][key]
    except Exception:
        return fallback

OLLAMA_BASE_URL = _get_secret("ollama", "base_url", "http://localhost:11434")
NEO4J_URI = _get_secret("neo4j", "uri", "bolt://localhost:7687")
NEO4J_USER = _get_secret("neo4j", "user", "neo4j")
NEO4J_PASSWORD = _get_secret("neo4j", "password", "")

# ==========================================
# UI/UX 美化 — 临床医疗系统风格
# ==========================================
def load_css():
    st.markdown("""
        <style>
        /* ===== 全局基础 ===== */
        .stApp {
            background: linear-gradient(180deg, #F5F9FC 0%, #EEF4F8 100%);
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ===== 主标题 ===== */
        .main-header {
            color: #0D7377;
            font-weight: 700;
            font-size: 1.4em;
            letter-spacing: 0.02em;
            border-bottom: 3px solid #0D7377;
            padding-bottom: 12px;
            margin-bottom: 24px;
            text-align: center;
        }

        /* ===== 侧边栏 ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFC 100%);
            box-shadow: 2px 0 12px rgba(0,0,0,0.04);
            border-right: 1px solid #E3ECF2;
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #0D7377;
        }

        /* ===== 用户卡片 ===== */
        .user-card {
            background: linear-gradient(135deg, #E0F7F6 0%, #D4F0EF 100%);
            padding: 16px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #0D7377;
            color: #1A4A4D;
        }

        /* ===== 聊天消息气泡 ===== */
        .stChatMessage {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            margin-bottom: 12px;
            border: 1px solid #E8EEF2;
            transition: box-shadow 0.2s;
        }
        .stChatMessage:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        /* ===== 知识卡 ===== */
        .medical-card {
            background: #F8FAFB;
            border-left: 4px solid #0D7377;
            padding: 14px 16px;
            border-radius: 0 6px 6px 0;
            margin: 12px 0;
            font-size: 0.9em;
            color: #3A4F5C;
            line-height: 1.6;
        }

        /* ===== 免责声明 ===== */
        .disclaimer {
            font-size: 0.78em;
            color: #8899A6;
            text-align: center;
            margin-top: 18px;
            padding-top: 14px;
            border-top: 1px solid #E8EEF2;
            letter-spacing: 0.02em;
        }

        /* ===== 按钮 ===== */
        .stButton button {
            background-color: #0D7377;
            color: white;
            border-radius: 8px;
            border: none;
            transition: all 0.2s;
            font-weight: 500;
            letter-spacing: 0.01em;
        }
        .stButton button:hover {
            background-color: #095C60;
            color: white;
            border: none;
            box-shadow: 0 2px 6px rgba(13,115,119,0.25);
        }

        /* ===== 意图标签 ===== */
        .intent-badge {
            background-color: #2C4251;
            color: white;
            padding: 3px 10px;
            border-radius: 4px;
            margin: 2px 4px 2px 0;
            font-size: 0.8em;
            display: inline-block;
            letter-spacing: 0.02em;
        }

        /* ===== 全链路日志区域 — 全宽显示 ===== */
        .pipeline-log-section {
            background: #FAFBFC;
            border: 1px solid #E3ECF2;
            border-radius: 8px;
            padding: 0;
            margin: 24px 0 12px 0;
            max-height: 600px;
            overflow-y: auto;
        }
        .pipeline-log-section .stMarkdown {
            padding: 12px 20px;
        }
        .pipeline-log-section h2 { color: #0D7377; font-size: 1.1em; border-bottom: 1px solid #E3ECF2; padding-bottom: 8px; }
        .pipeline-log-section h3 { color: #2C4251; font-size: 0.95em; }
        .pipeline-log-section h4 { color: #4A6A7D; font-size: 0.88em; }
        .pipeline-log-section code { font-size: 0.82em; background: #F0F4F7; padding: 1px 4px; border-radius: 3px; }
        .pipeline-log-section pre { background: #F5F7F9; border: 1px solid #E3ECF2; border-radius: 6px; }

        /* ===== 输入框 ===== */
        .stChatInput textarea {
            border-radius: 10px;
            border: 1.5px solid #DDE5EC;
        }
        .stChatInput textarea:focus {
            border-color: #0D7377;
            box-shadow: 0 0 0 2px rgba(13,115,119,0.12);
        }

        /* ===== 健康检查提示 ===== */
        .stAlert {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_entities_pretty(entities):
    if not entities or entities == "{}":
        return "<span style='color:#999; font-size:0.8em;'>未检测到关键医疗实体</span>"
    if isinstance(entities, str):
        try: entities = ast.literal_eval(entities)
        except: return entities
    if not isinstance(entities, dict): return str(entities)
    html = ""
    color_map = {
        "疾病": "#E74C3C", "疾病症状": "#E67E22", "药品": "#3498DB",
        "检查项目": "#9B59B6", "科目": "#1ABC9C", "食物": "#2ECC71",
        "药品商": "#34495E", "治疗方法": "#F1C40F"
    }
    for key, value in entities.items():
        color = color_map.get(key, "#95A5A6")
        # 兼容 list 和 str 两种格式
        display_value = ", ".join(value) if isinstance(value, list) else value
        html += f"""<span style='background-color: {color}; color: white; padding: 4px 10px; border-radius: 12px; margin-right: 5px; font-size: 0.85em; display: inline-block; margin-bottom: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'><b>{key}</b>: {display_value}</span>"""
    return html

# ==========================================
# 后端逻辑
# ==========================================
# @st.cache_resource(show_spinner=False)
# def load_model_and_components(cache_model='best_roberta_rnn_model_ent_aug'):
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     print(f"模型加载到设备: {device}")
#     try:
#         with open('tmp_data/tag2idx.npy', 'rb') as f:
#             tag2idx = pickle.load(f)
#         idx2tag = list(tag2idx)
#         rule = zwk.rule_find()
#         tfidf_r = zwk.tfidf_alignment()
#         model_name = 'model/chinese-roberta-wwm-ext'
#         bert_tokenizer = BertTokenizer.from_pretrained(model_name)
#         bert_model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
#         bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt', map_location=device))
#         bert_model = bert_model.to(device)
#         bert_model.eval()
#         return bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device
#     except Exception as e:
#         st.error(f"模型加载失败: {str(e)}")
#         return None, None, None, None, None, device


def warmup_agent():
    """预热 Agent 的全部组件，避免首次请求等待过久。

    预热内容：
    1. ONNX NER（RoBERTa+AC自动机+TF-IDF）~14s
    2. BGE 嵌入模型 ~3s
    3. FAISS 向量索引 ~0.5s
    4. LLM 降级链（仅构建实例，不调API）
    总计约 18s，在页面加载时完成，首次问诊只需等 LLM 返回。
    """
    import time

    # 1. NER 组件
    try:
        t0 = time.time()
        from agent_graph import _load_ner_components
        _load_ner_components()
        print(f"[Warmup] NER 组件就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"NER 预热失败: {e}")

    # 2. BGE 嵌入模型
    try:
        t0 = time.time()
        from vector_memory import _get_model
        _get_model()
        print(f"[Warmup] BGE 嵌入模型就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"BGE 预热失败: {e}")

    # 3. FAISS 索引
    try:
        t0 = time.time()
        from vector_memory import _get_index
        _get_index()
        print(f"[Warmup] FAISS 索引就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"FAISS 预热失败: {e}")

    # 4. LLM 降级链（不调 API，只构建实例）
    try:
        t0 = time.time()
        from agent_graph import _get_llm
        _get_llm(with_tools=True)
        print(f"[Warmup] LLM 降级链就绪 ({time.time()-t0:.1f}s)")
    except Exception as e:
        st.warning(f"LLM 预热失败: {e}")


def check_ollama_connection():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except:
        return False

def local_intent_recognition(query):
    intent_keywords = {
        "查询疾病简介": ["是什么", "简介", "介绍", "什么叫", "什么是"],
        "查询疾病病因": ["原因", "病因", "为什么", "怎么引起", "怎么回事"],
        "查询疾病预防措施": ["预防", "怎么预防", "如何预防", "避免", "防止"],
        "查询疾病治疗周期": ["多久", "多长时间", "治疗周期", "恢复期"],
        "查询治愈概率": ["治愈率", "治愈概率", "能不能治好", "能治好吗"],
        "查询疾病易感人群": ["易感人群", "容易得", "适合人群", "高发人群"],
        "查询疾病所需药品": ["吃什么药", "药品", "药物", "药方", "吃什么"],
        "查询疾病宜吃食物": ["宜吃", "推荐吃", "可以吃", "吃什么好"],
        "查询疾病忌吃食物": ["忌吃", "不能吃", "不要吃", "禁忌", "不宜"],
        "查询疾病所需检查项目": ["检查", "做什么检查", "需要检查", "检测"],
        "查询疾病所属科目": ["挂什么科", "科室", "看什么科", "就诊科室"],
        "查询疾病的症状": ["症状", "表现", "有什么症状", "有哪些表现"],
        "查询疾病的治疗方法": ["怎么治", "治疗方法", "治疗方案", "疗法", "如何治疗"],
        "查询疾病的并发疾病": ["并发症", "并发", "会引发", "伴随"],
        "查询药品的生产商": ["生产商", "生产厂家", "哪个厂", "谁生产"]
    }
    detected = []
    for intent, kws in intent_keywords.items():
        for kw in kws:
            if kw in query:
                detected.append(intent); break
    if not detected: detected = ["查询疾病简介"]
    elif len(detected) > 5: detected = detected[:5]
    if any("疾病" in i for i in detected) and "查询疾病简介" not in detected:
        detected.insert(0, "查询疾病简介")
    return str(detected)

def Intent_Recognition(query, choice, progress_queue=None):
    if not check_ollama_connection():
        if progress_queue: progress_queue.put("Ollama 未连接，使用本地规则识别...")
        return local_intent_recognition(query)
    # ========== 补全的意图识别 Prompt ==========
    prompt = f"""
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询药品的生产商"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询疾病简介","查询疾病病因"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询疾病简介","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，检查是根本性的！其次查看病因。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 检查是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病简介","查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法，然后推荐药、检查、食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病简介","查询疾病的并发疾病"]  # 用户问的是疾病并发疾病，随后可以科普一下癌症简介。
**示例10：**
输入："葡萄糖浆的生产者是谁？葡萄糖浆是谁生产的？"
输出：["查询药品的生产商"]  # 显然，用户想要问药品的生产商
通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{query}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！
"""
    try:
        if progress_queue: progress_queue.put("正在调用 AI 推断意图...")
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {"model": choice, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 200}}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()['response']
        if progress_queue: progress_queue.put("意图识别完成")
        return result
    except:
        if progress_queue: progress_queue.put("API出错，切换本地模式")
        return local_intent_recognition(query)

def add_shuxing_prompt(entity, shuxing, client):
    try:
        sql = "MATCH (a:疾病{{名称:'{}'}}) RETURN a.{}".format(entity, shuxing)
        res = client.run(sql).data()[0].values()
        prompt = f"<提示>用户对{entity}可能有查询{shuxing}需求，知识库内容如下："
        if len(res) > 0: prompt += "".join(res)
        else: prompt += "图谱中无信息。"
        prompt += "</提示>"
        return prompt
    except: return ""

def add_lianxi_prompt(entity, lianxi, target, client):
    try:
        sql = "MATCH (a:疾病{{名称:'{}'}})-[r:{}]->(b:{}) RETURN b.名称".format(entity, lianxi, target)
        res = client.run(sql).data()
        res = [list(d.values())[0] for d in res]
        prompt = f"<提示>用户对{entity}可能有查询{lianxi}需求，知识库内容如下："
        if len(res) > 0: prompt += "、".join(res)
        else: prompt += "图谱中无信息。"
        prompt += "</提示>"
        return prompt
    except: return ""

def generate_prompt(response, query, client, entities):
    """
    重构后的 prompt 组装函数：纯粹的图谱查询工具，不再进行 NER 推理
    """
    yitu = []
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。如果提示中没有相关信息，请明确告知。</指令>"

    # 症状推断：取第一个疾病（不再随机）
    if '疾病症状' in entities and '疾病' not in entities:
        try:
            sql = "MATCH (a:疾病)-[r:疾病的症状]->(b:疾病症状 {{名称:'{}'}}) RETURN a.名称".format(entities['疾病症状'])
            res = list(client.run(sql).data())
            if len(res) > 0:
                disease_names = [list(d.values())[0] for d in res]
                # 取第一个作为最相关（数据库默认顺序）
                entities['疾病'] = disease_names[0]
                prompt += "<提示>用户有{}的情况，知识库推测其可能是得了{}。</提示>".format(entities['疾病症状'], "、".join(disease_names))
        except: pass

    pre_len = len(prompt)
    intent_map = {
        "简介": ('疾病简介', '查询疾病简介', add_shuxing_prompt),
        "病因": ('疾病病因', '查询疾病病因', add_shuxing_prompt),
        "预防": ('预防措施', '查询疾病预防措施', add_shuxing_prompt),
        "治疗周期": ('治疗周期', '查询疾病治疗周期', add_shuxing_prompt),
        "治愈概率": ('治愈概率', '查询治愈概率', add_shuxing_prompt),
        "易感人群": ('疾病易感人群', '查询疾病易感人群', add_shuxing_prompt),
        "药品": ('疾病使用药品', '查询疾病使用药品', lambda e,l,c: add_lianxi_prompt(e,l,'药品',c)),
        "宜吃食物": ('疾病宜吃食物', '查询疾病宜吃食物', lambda e,l,c: add_lianxi_prompt(e,l,'食物',c)),
        "忌吃食物": ('疾病忌吃食物', '查询疾病忌吃食物', lambda e,l,c: add_lianxi_prompt(e,l,'食物',c)),
        "检查项目": ('疾病所需检查', '查询疾病所需检查', lambda e,l,c: add_lianxi_prompt(e,l,'检查项目',c)),
        "查询疾病所属科目": ('疾病所属科目', '查询疾病所属科目', lambda e,l,c: add_lianxi_prompt(e,l,'科目',c)),
        "症状": ('疾病的症状', '查询疾病的症状', lambda e,l,c: add_lianxi_prompt(e,l,'疾病症状',c)),
        "治疗": ('治疗的方法', '查询治疗的方法', lambda e,l,c: add_lianxi_prompt(e,l,'治疗方法',c)),
        "并发": ('疾病并发疾病', '查询疾病并发疾病', lambda e,l,c: add_lianxi_prompt(e,l,'疾病',c)),
    }
    if '疾病' in entities:
        for key, val in intent_map.items():
            if key in response:
                if len(val) == 3:
                    prompt += val[2](entities['疾病'], val[0], client)
                yitu.append(val[1])

    if "生产商" in response and '药品' in entities:
        try:
            sql = "MATCH (a:药品商)-[r:生产]->(b:药品{{名称:'{}'}}) RETURN a.名称".format(entities['药品'])
            res = client.run(sql).data()[0].values()
            prompt += f"<提示>用户对{entities['药品']}可能有查询药品生产商的需求，知识图谱内容如下："
            prompt += "".join(res) if len(res)>0 else "图谱中无信息"
            prompt += "</提示>"
        except: pass
        yitu.append('查询药物生产商')

    if pre_len == len(prompt):
        prompt += "<提示>知识库中暂无相关详细信息。</提示>"
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += "<注意>请基于提示内容回答，如果无法回答，请明确告知。</注意>"
    
    return prompt, "、".join(yitu)

def run_intent_recognition_thread(query, choice, result_queue, progress_queue):
    try:
        result = Intent_Recognition(query, choice, progress_queue)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(f"错误: {str(e)}")


def fetch_intent_and_entities(query, choice, client, ort_session, bert_tokenizer, rule, tfidf_r, idx2tag):
    """
    精简后的实体与意图识别函数，专为 ONNX 优化
    """
    # 1. 本地 NER 识别 (使用 ONNX)
    if ort_session:
        raw_entities = zwk.get_ner_result_onnx(ort_session, bert_tokenizer, query, rule, tfidf_r, idx2tag)
    else:
        raw_entities = {}
        
    # 2. 结构化记忆预热
    agent_memory = [{"type": k, "name": v} for k, v in raw_entities.items()]

    # 3. 意图识别
    intent_response = Intent_Recognition(query, choice, progress_queue=None)
    
    # 4. 图谱查询组装
    if client:
        # 彻底解耦：直接传入已经识别好的 raw_entities，不再传一堆模型参数
        prompt, yitu = generate_prompt(intent_response, query, client, raw_entities)
    else:
        prompt, yitu = f"无法连接数据库。用户问题:{query}", "无"
        
    return prompt, yitu, raw_entities, agent_memory

# ==========================================
# 流式回答生成器（修复Ollama体验问题）
# ==========================================
def stream_ollama_chat(model, messages, options):
    """使用 requests 流式读取 Ollama /api/chat，返回生成器逐段输出 token 文本"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {"model": model, "messages": messages, "stream": True, "options": options}
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        content = data.get('message', {}).get('content', '')
                        if content:
                            yield content
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"\n[生成错误: {str(e)}]"

# ==========================================
# 主程序
# ==========================================
def main(is_admin, usname):
    load_css()

    onnx_path = 'model/ner_model.onnx'
    cache_model = 'best_roberta_rnn_model_ent_aug'

    st.markdown("<h1 class='main-header'>🏥 智能医疗问答与诊疗辅助系统</h1>", unsafe_allow_html=True)

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.markdown(f"""
        <div class="user-card">
            <h4>👋 欢迎回来</h4>
            <p><b>用户:</b> {usname if usname else '访客'}</p>
            <p><b>角色:</b> {'管理员' if is_admin else '普通用户'}</p>
        </div>
        """, unsafe_allow_html=True)

        # 会话管理 — 从磁盘加载历史
        if 'chat_windows' not in st.session_state:
            saved_windows = load_all_windows(usname)
            st.session_state.chat_windows = [list(range(len(w))) for w in saved_windows]  # dummy metadata
            st.session_state.messages = [w for w in saved_windows]
            if not st.session_state.messages:
                st.session_state.messages = [[]]
                st.session_state.chat_windows = [[]]
        st.caption("💬 会话管理")
        col_add, col_del = st.columns([3, 1])
        with col_add:
            if st.button('+ 新建窗口', use_container_width=True):
                st.session_state.chat_windows.append([])
                st.session_state.messages.append([])
                st.rerun()
        with col_del:
            if st.button('🗑 删除', use_container_width=True, disabled=len(st.session_state.chat_windows) <= 1):
                # 先确定要删哪个（当前选中的）
                window_options = [f"📋 病例窗口 {i+1}" for i in range(len(st.session_state.chat_windows))]
                selected = st.session_state.get('_active_window', 0)
                from conversation_storage import delete_conversation
                delete_conversation(st.session_state.usname, selected)
                # 从 session 中移除
                if len(st.session_state.chat_windows) > 1:
                    st.session_state.chat_windows.pop(selected)
                    st.session_state.messages.pop(selected)
                st.rerun()

        window_options = [f"📋 病例窗口 {i+1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox(
            '切换当前会话:', window_options, label_visibility="collapsed",
            key='_active_window_select'
        )
        active_window_index = int(selected_window.split()[-1]) - 1
        st.session_state['_active_window'] = active_window_index

        st.divider()
        with st.expander("⚙️ 系统设置 & 专家调试", expanded=True):
            selected_option = st.selectbox(
                '🧠 选择 AI 模型引擎:',
                ['qwen:1.8b', 'qwen:4b', 'qwen3:4b', 'llama3:8b'],
                index=0
            )
            choice = selected_option
            st.divider()
            st.caption("👁️ 结果可视化选项")
            show_ent = st.checkbox("显示实体识别 (NER)", value=True)
            show_int = st.checkbox("显示意图分析", value=True)
            show_prompt = st.checkbox("显示图谱知识", value=False)
            show_pipeline = st.checkbox("🔍 查看全链路流水日志", value=False)
            gpu_status = "✅ CUDA GPU 就绪" if torch.cuda.is_available() else "⚠️ 仅 CPU 运行"
            st.caption(f"💻 当前 PyTorch 硬件: {gpu_status}")
            if is_admin:
                st.markdown('[🔗 管理知识图谱 (Neo4j)](http://127.0.0.1:7474/)', unsafe_allow_html=True)

        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()

    # 预热 Agent 的全部组件
    warmup_agent()

    # 检查是否有未完成的对话（上次崩溃/中断）
    from checkpoint import has_any_unfinished, get_unfinished_checkpoint
    if has_any_unfinished(usname):
        checkpoint = get_unfinished_checkpoint(usname, active_window_index)
        if checkpoint:
            last_node = checkpoint.get("node", "未知")
            st.warning(f"检测到上次对话在「{last_node}」节点中断。如需继续，请重新发送您的问题，Agent 将从检查点恢复。")

    current_messages = st.session_state.messages[active_window_index]

    # 渲染历史消息
    for message in current_messages:
        role = message["role"]
        avatar = "🩺" if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if role == "assistant":
                ent_data = message.get("ent", "")
                yitu_data = message.get("yitu", "")
                prompt_data = message.get("prompt", "")
                if (show_ent and ent_data) or (show_int and yitu_data) or (show_prompt and prompt_data):
                    st.markdown("---")
                    st.caption("🔍 **AI 诊疗分析报告**")
                    c1, c2 = st.columns(2)
                    if show_ent and ent_data:
                        with c1:
                            st.markdown("**关键医学实体:**")
                            st.markdown(render_entities_pretty(ent_data), unsafe_allow_html=True)
                    if show_int and yitu_data:
                        with c2:
                            st.markdown("**工具调用链路:**")
                            tools_list = yitu_data.split(", ") if isinstance(yitu_data, str) else []
                            intent_html = "".join([f"<span class='intent-badge'>{t}</span>" for t in tools_list])
                            st.markdown(intent_html, unsafe_allow_html=True)
                    if show_prompt and prompt_data:
                        with st.expander("📚 知识图谱查询结果", expanded=True):
                            st.markdown(f'<div class="medical-card">{prompt_data}</div>', unsafe_allow_html=True)
                st.markdown('<div class="disclaimer">⚠️ AI 生成内容仅供参考，不可替代专业医生诊断。</div>', unsafe_allow_html=True)

    # ==========================================
    # 处理新输入 -- LangGraph ReAct Agent 流式执行
    # ==========================================
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = load_agent_memory(usname)

    if query := st.chat_input("请描述您的症状或问题...", key=f"chat_input_{active_window_index}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🩺"):
            response_placeholder = st.empty()
            status_placeholder = st.empty()

            memory_hint = ""
            if st.session_state.agent_memory:
                memory_hint = f"（已加载 {len(st.session_state.agent_memory)} 条历史知识）"
            status_placeholder.info(f"🧠 自主推理引擎启动{memory_hint}，正在分析症状...")

            full_response = ""
            entities = {}
            tools_used = []
            tool_results_summary = []
            error_occurred = False

            # 传递完整历史消息，让 Agent 看到之前的对话上下文
            history = []
            for m in current_messages:
                history.append({"role": m["role"], "content": m["content"]})

            for event in stream_agent(query, memory=st.session_state.agent_memory, history_messages=history,
                                     log_user=st.session_state.usname, log_window=active_window_index):
                t = event.get("type")
                if t == "node_completed":
                    node = event.get("node", "")
                    data = event.get("data", {})

                    if node == "preprocess":
                        entities = data.get("entities", {})
                        if entities:
                            ent_desc = "、".join(f"{k}:{v}" for k, v in entities.items())
                            status_placeholder.info(f"🔬 实体识别: {ent_desc}")

                    elif node == "llm_planner":
                        tool_calls = data.get("tool_calls", [])
                        if tool_calls:
                            tc_names = [tc.get("name_cn", tc["name"]) for tc in tool_calls]
                            tools_used.extend(tc_names)
                            status_placeholder.info(f"🔧 检索知识图谱: {', '.join(tc_names)}...")
                        elif data.get("content"):
                            status_placeholder.info("📝 综合推理中，正在生成诊断建议...")

                    elif node == "tool_executor":
                        results = data.get("tool_results", [])
                        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
                        status_placeholder.info(f"📚 图谱查询完成 ({success_count}/{len(results)} 成功)，继续推理...")
                        for r in results:
                            if isinstance(r, dict) and r.get("success"):
                                rd = r.get("data", {})
                                if "items" in rd:
                                    items = rd["items"][:5]
                                    tool_results_summary.append(
                                        f"{rd.get('disease', '')} {rd.get('relation_type', '')}: {', '.join(items)}"
                                    )
                                elif "diseases" in rd:
                                    tool_results_summary.append(
                                        f"{rd.get('symptom', '')} -> {', '.join(rd['diseases'][:5])}"
                                    )
                                elif "value" in rd:
                                    tool_results_summary.append(
                                        f"{rd.get('disease', '')} {rd.get('attr_type', '')}: {str(rd['value'])[:80]}"
                                    )

                elif t == "done":
                    final_state = event.get("final_state", {})
                    serialized_msgs = final_state.get("messages", [])
                    for msg in reversed(serialized_msgs):
                        if msg.get("type") == "AIMessage" and msg.get("content") and not msg.get("has_tool_calls"):
                            full_response = msg["content"]
                            break
                    # 将本轮知识缓存合并到跨轮记忆
                    cache_keys = final_state.get("knowledge_cache_keys", [])
                    if cache_keys:
                        st.session_state.agent_memory.update({k: True for k in cache_keys})

                elif t == "error":
                    error_occurred = True
                    full_response = f"抱歉，推理引擎遇到错误：{event.get('message', '')}"
                    status_placeholder.error(full_response)

            if not full_response and not error_occurred:
                full_response = "抱歉，根据已知信息无法回答该问题，建议咨询专业医生。"

            clean_response = re.sub(
                r"<think[^>]*>.*?</think>", "", full_response, flags=re.DOTALL
            ).strip()
            if not clean_response:
                clean_response = full_response
            full_response = clean_response

            response_placeholder.markdown(full_response)
            status_placeholder.empty()

            if (show_ent and entities) or (show_int and tools_used) or (show_prompt and tool_results_summary):
                st.markdown("---")
                st.caption("🔍 **AI 诊疗分析报告**")
                c1, c2 = st.columns(2)
                if show_ent and entities:
                    with c1:
                        st.markdown("**关键医学实体 (NER):**")
                        st.markdown(render_entities_pretty(entities), unsafe_allow_html=True)
                if show_int and tools_used:
                    with c2:
                        st.markdown("**Agent 工具调用链路:**")
                        unique_tools = list(dict.fromkeys(tools_used))
                        intent_html = "".join([f"<span class='intent-badge'>{t}</span>" for t in unique_tools])
                        st.markdown(intent_html, unsafe_allow_html=True)
                if show_prompt and tool_results_summary:
                    with st.expander("📚 知识图谱查询结果", expanded=True):
                        summary_text = "\n\n".join(f"- {s}" for s in tool_results_summary)
                        st.markdown(f'<div class="medical-card">{summary_text}</div>', unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">⚠️ AI 生成内容仅供参考，不可替代专业医生诊断。如遇紧急情况请及时就医。</div>', unsafe_allow_html=True)

            current_messages.append({
                "role": "assistant",
                "content": full_response,
                "yitu": ", ".join(dict.fromkeys(tools_used)),
                "prompt": "\n".join(f"- {s}" for s in tool_results_summary),
                "ent": str(entities),
            })

    st.session_state.messages[active_window_index] = current_messages

    # 自动持久化到磁盘
    save_conversation(usname, active_window_index, current_messages)

    # ===== 全链路流水日志 — 主内容区全宽展示（不在侧边栏） =====
    if show_pipeline:
        import os as _os
        safe_name = usname.replace("/", "_").replace("\\", "_")
        log_path = _os.path.join("tmp_data", "pipeline_logs", f"{safe_name}_window{active_window_index}.md")
        if _os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as _f:
                log_content = _f.read()
            st.divider()
            with st.expander("📊 全链路流水日志（当前窗口）", expanded=False):
                st.markdown(log_content)
        else:
            st.caption("📊 暂无流水日志（请先发送一条消息）")
    save_agent_memory(usname, st.session_state.agent_memory)
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'admin' not in st.session_state:
        st.session_state.admin = False
    if 'usname' not in st.session_state:
        st.session_state.usname = ""

    if not st.session_state.logged_in:
        st.error("请先登录系统")
    else:
        # 使用登录时保存的管理员状态，不再强制覆盖
        main(st.session_state.admin, st.session_state.usname)