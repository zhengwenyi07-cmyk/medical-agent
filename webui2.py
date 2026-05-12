"""
增强版医疗问答系统 - UI/UX 美化版 (Clinical Clean Style) v2
修复了窗口切换的 Bug，保留了模型选择和调试信息显示功能
"""

import os
import streamlit as st
import ner_model as zwk
import pickle
import ollama
from transformers import BertTokenizer
import torch
import py2neo
import random
import re
from threading import Thread
from queue import Queue
import time
import requests
import json
import ast  # 用于安全地解析保存的字典字符串

# ==========================================
# UI/UX 美化配置区域
# ==========================================

def load_css():
    """注入自定义 CSS 样式"""
    st.markdown("""
        <style>
        /* 全局字体与背景 */
        .stApp {
            background-color: #F4F8FB;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* 顶部标题栏 */
        .main-header {
            color: #008B8B;
            font-weight: 700;
            border-bottom: 2px solid #008B8B;
            padding-bottom: 15px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        /* 侧边栏美化 */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
            border-right: 1px solid #E0E0E0;
        }
        
        /* 侧边栏用户卡片 */
        .user-card {
            background-color: #E0F2F1;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #008B8B;
        }
        
        /* 聊天气泡优化 */
        .stChatMessage {
            background-color: #FFFFFF;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 10px;
            border: 1px solid #EFEFEF;
        }
        
        /* 知识库/调试信息卡片 */
        .medical-card {
            background-color: #FAFAFA;
            border-left: 4px solid #008B8B;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 0.9em;
            color: #444;
        }
        
        /* 免责声明文字 */
        .disclaimer {
            font-size: 0.75em;
            color: #95A5A6;
            text-align: center;
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px dashed #E0E0E0;
        }
        
        /* 按钮样式美化 */
        .stButton button {
            background-color: #008B8B;
            color: white;
            border-radius: 8px;
            border: none;
            transition: 0.2s;
            font-weight: 500;
        }
        .stButton button:hover {
            background-color: #006666;
            color: white;
            border: none;
        }
        
        /* 意图标签样式 */
        .intent-badge {
            background-color: #34495E;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            margin: 2px;
            font-size: 0.8em;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)

def render_entities_pretty(entities):
    """
    将实体字典渲染为漂亮的 HTML 标签
    """
    if not entities or entities == "{}":
        return "<span style='color:#999; font-size:0.8em;'>未检测到关键医疗实体</span>"
    
    # 如果是字符串，尝试解析回字典
    if isinstance(entities, str):
        try:
            entities = ast.literal_eval(entities)
        except:
            return entities 
            
    if not isinstance(entities, dict):
         return str(entities)

    html = ""
    # 颜色映射
    color_map = {
        "疾病": "#E74C3C",    # 红
        "疾病症状": "#E67E22", # 橙
        "药品": "#3498DB",    # 蓝
        "检查项目": "#9B59B6", # 紫
        "科目": "#1ABC9C",    # 青
        "食物": "#2ECC71",    # 绿
        "药品商": "#34495E",   # 灰蓝
        "治疗方法": "#F1C40F"  # 黄
    }
    
    for key, value in entities.items():
        color = color_map.get(key, "#95A5A6") 
        html += f"""
        <span style='background-color: {color}; color: white; 
            padding: 4px 10px; border-radius: 12px; margin-right: 5px; 
            font-size: 0.85em; display: inline-block; margin-bottom: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
            <b>{key}</b>: {value}
        </span>
        """
    return html

# ==========================================
# 后端逻辑区域 (保持不变)
# ==========================================

@st.cache_resource
def load_model(cache_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"模型加载到设备: {device}")
    
    try:
        with open('tmp_data/tag2idx.npy', 'rb') as f:
            tag2idx = pickle.load(f)
        idx2tag = list(tag2idx)
        rule = zwk.rule_find()
        tfidf_r = zwk.tfidf_alignment()
        model_name = 'model/chinese-roberta-wwm-ext'
        bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
        
        bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt', map_location=device))
        bert_model = bert_model.to(device)
        bert_model.eval()
        return bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}。请确保模型文件存在。")
        return None, None, None, None, None, device

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
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
    
    detected_intents = []
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in query:
                detected_intents.append(intent)
                break
    
    if len(detected_intents) == 0:
        detected_intents = ["查询疾病简介"]
    elif len(detected_intents) > 5:
        detected_intents = detected_intents[:5]
    else:
        has_disease_intent = any("疾病" in intent for intent in detected_intents)
        if has_disease_intent and "查询疾病简介" not in detected_intents:
            detected_intents.insert(0, "查询疾病简介")
    return str(detected_intents)

def Intent_Recognition(query, choice, progress_queue=None):
    if not check_ollama_connection():
        if progress_queue: progress_queue.put("Ollama未连接，使用本地规则识别...")
        return local_intent_recognition(query)
    
    prompt = f"""
    阅读下列提示，回答问题（问题在输入的最后）:
    当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断...
    问题输入："{query}"
    输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！
    """
    try:
        if progress_queue: progress_queue.put("正在调用 AI 进行意图推断...")
        response = ollama.generate(model=choice, prompt=prompt, options={"temperature": 0.3, "num_predict": 200})
        rec_result = response['response']
        if progress_queue: progress_queue.put("意图识别完成")
        return rec_result
    except Exception as e:
        if progress_queue: progress_queue.put(f"API出错，切换本地模式")
        return local_intent_recognition(query)

def add_shuxing_prompt(entity, shuxing, client):
    add_prompt = ""
    try:
        sql_q = "MATCH (a:疾病{{名称:'{}'}}) RETURN a.{}".format(entity, shuxing)
        res = client.run(sql_q).data()[0].values()
        add_prompt += "<提示>用户对{}可能有查询{}需求，知识库内容如下：".format(entity, shuxing)
        if len(res) > 0:
            join_res = "".join(res)
            add_prompt += join_res
        else:
            add_prompt += "图谱中无信息。"
        add_prompt += "</提示>"
    except: pass
    return add_prompt

def add_lianxi_prompt(entity, lianxi, target, client):
    add_prompt = ""
    try:
        sql_q = "MATCH (a:疾病{{名称:'{}'}})-[r:{}]->(b:{}) RETURN b.名称".format(entity, lianxi, target)
        res = client.run(sql_q).data()
        res = [list(data.values())[0] for data in res]
        add_prompt += "<提示>用户对{}可能有查询{}需求，知识库内容如下：".format(entity, lianxi)
        if len(res) > 0:
            join_res = "、".join(res)
            add_prompt += join_res
        else:
            add_prompt += "图谱中无信息。"
        add_prompt += "</提示>"
    except: pass
    return add_prompt

def generate_prompt(response, query, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag):
    if bert_model is None:
        return "<提示>模型未加载，无法识别实体</提示>", "", {}

    entities = zwk.get_ner_result(bert_model, bert_tokenizer, query, rule, tfidf_r, device, idx2tag)
    yitu = []
    
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。如果提示中没有相关信息，请明确告知。</指令>"
    
    if '疾病症状' in entities and '疾病' not in entities:
        try:
            sql_q = "MATCH (a:疾病)-[r:疾病的症状]->(b:疾病症状 {{名称:'{}'}}) RETURN a.名称".format(entities['疾病症状'])
            res = list(client.run(sql_q).data()[0].values())
            if len(res) > 0:
                entities['疾病'] = random.choice(res)
                all_en = "、".join(res)
                prompt += "<提示>用户有{}的情况，知识库推测其可能是得了{}。</提示>".format(entities['疾病症状'], all_en)
        except: pass

    pre_len = len(prompt)
    intent_map = {
        "简介": ('疾病简介', '查询疾病简介', add_shuxing_prompt),
        "病因": ('疾病病因', '查询疾病病因', add_shuxing_prompt),
        "预防": ('预防措施', '查询疾病预防措施', add_shuxing_prompt),
        "治疗周期": ('治疗周期', '查询疾病治疗周期', add_shuxing_prompt),
        "治愈概率": ('治愈概率', '查询治愈概率', add_shuxing_prompt),
        "易感人群": ('疾病易感人群', '查询疾病易感人群', add_shuxing_prompt),
        "药品": ('疾病使用药品', '查询疾病使用药品', lambda e, l, c: add_lianxi_prompt(e, l, '药品', c)),
        "宜吃食物": ('疾病宜吃食物', '查询疾病宜吃食物', lambda e, l, c: add_lianxi_prompt(e, l, '食物', c)),
        "忌吃食物": ('疾病忌吃食物', '查询疾病忌吃食物', lambda e, l, c: add_lianxi_prompt(e, l, '食物', c)),
        "检查项目": ('疾病所需检查', '查询疾病所需检查', lambda e, l, c: add_lianxi_prompt(e, l, '检查项目', c)),
        "查询疾病所属科目": ('疾病所属科目', '查询疾病所属科目', lambda e, l, c: add_lianxi_prompt(e, l, '科目', c)),
        "症状": ('疾病的症状', '查询疾病的症状', lambda e, l, c: add_lianxi_prompt(e, l, '疾病症状', c)),
        "治疗": ('治疗的方法', '查询治疗的方法', lambda e, l, c: add_lianxi_prompt(e, l, '治疗方法', c)),
        "并发": ('疾病并发疾病', '查询疾病并发疾病', lambda e, l, c: add_lianxi_prompt(e, l, '疾病', c)),
    }

    if '疾病' in entities:
        for key, val in intent_map.items():
            if key in response:
                if len(val) == 3: 
                    prompt += val[2](entities['疾病'], val[0], client)
                yitu.append(val[1])

    if "生产商" in response and '药品' in entities:
        try:
            sql_q = "MATCH (a:药品商)-[r:生产]->(b:药品{{名称:'{}'}}) RETURN a.名称".format(entities['药品'])
            res = client.run(sql_q).data()[0].values()
            prompt += "<提示>用户对{}可能有查询药品生产商的需求，知识图谱内容如下：".format(entities['药品'])
            if len(res) > 0: prompt += "".join(res)
            else: prompt += "图谱中无信息"
            prompt += "</提示>"
        except: pass
        yitu.append('查询药物生产商')

    if pre_len == len(prompt):
        prompt += "<提示>知识库中暂无相关详细信息。</提示>"
    
    prompt += "<用户问题>{}</用户问题>".format(query)
    prompt += "<注意>请基于提示内容回答，如果无法回答，请明确告知。</注意>"
    
    return prompt, "、".join(yitu), entities

def run_intent_recognition_thread(query, choice, result_queue, progress_queue):
    try:
        result = Intent_Recognition(query, choice, progress_queue)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(f"错误: {str(e)}")

# ==========================================
# 主程序
# ==========================================

def main(is_admin, usname):
    load_css()
    
    # 强制 Admin 权限以便演示功能 (实际可改回参数)
    is_admin = True 
    cache_model = 'best_roberta_rnn_model_ent_aug'
    
    st.markdown("<h1 class='main-header'>🏥 智能医疗问答与诊疗辅助系统</h1>", unsafe_allow_html=True)

    # ---------------------------
    # 侧边栏重构 (包含你需要的所有功能)
    # ---------------------------
    with st.sidebar:
        # 1. 用户信息
        st.markdown(f"""
        <div class="user-card">
            <h4>👋 欢迎回来</h4>
            <p><b>用户:</b> {usname if usname else '访客'}</p>
            <p><b>角色:</b> {'管理员' if is_admin else '普通用户'}</p>
        </div>
        """, unsafe_allow_html=True)

        # 2. 会话管理
        if 'chat_windows' not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]

        st.caption("💬 会话管理")
        if st.button('➕ 新建问诊窗口', use_container_width=True):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [f"📋 病例窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox('切换当前会话:', window_options, label_visibility="collapsed")
        
        # --- [修复 Bug 的关键行] ---
        # 字符串是 "📋 病例窗口 1"，split后是 ['📋', '病例窗口', '1']
        # 我们取最后一个元素 [-1] 也就是 '1'
        active_window_index = int(selected_window.split()[-1]) - 1

        st.divider()

        # 3. 设置与调试 (这里包含了你要的模型选择和开关)
        with st.expander("⚙️ 系统设置 & 专家调试", expanded=True):
            # [功能 1] 模型选择
            selected_option = st.selectbox(
                label='🧠 选择 AI 模型引擎:',
                options=['qwen:4b', 'qwen3:4b', 'llama3:8b']
            )
            choice = selected_option
            
            st.divider()
            st.caption("👁️ 结果可视化选项")
            
            # [功能 2] 结果显示开关
            show_ent = st.checkbox("显示实体识别 (NER)", value=True, help="显示识别出的疾病、药品等关键词")
            show_int = st.checkbox("显示意图分析", value=True, help="显示用户查询的目的，如查病因、查药")
            show_prompt = st.checkbox("显示图谱知识", value=False, help="显示从知识图谱检索到的原始数据")
            
            if is_admin:
                 st.markdown('[🔗 管理知识图谱 (Neo4j)](http://127.0.0.1:7474/)', unsafe_allow_html=True)

        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()

    # 加载模型与数据库连接
    bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model(cache_model)
    try:
        client = py2neo.Graph('http://localhost:7474', user='neo4j', password='12345678', name='neo4j')
    except:
        st.sidebar.error("❌ 无法连接到 Neo4j 数据库")
        client = None

    current_messages = st.session_state.messages[active_window_index]

    # ---------------------------
    # 历史消息渲染
    # ---------------------------
    for message in current_messages:
        role = message["role"]
        avatar = "🩺" if role == "assistant" else None
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            
            if role == "assistant":
                ent_data = message.get("ent", "")
                yitu_data = message.get("yitu", "")
                prompt_data = message.get("prompt", "")
                
                # 根据侧边栏的开关决定是否显示详细信息
                if (show_ent and ent_data) or (show_int and yitu_data) or (show_prompt and prompt_data):
                    with st.container():
                        st.markdown("---")
                        st.caption("🔍 **AI 诊疗分析报告**")
                        c1, c2 = st.columns(2)
                        
                        if show_ent and ent_data:
                            with c1:
                                st.markdown("**关键医学实体:**")
                                st.markdown(render_entities_pretty(ent_data), unsafe_allow_html=True)
                        
                        if show_int and yitu_data:
                            with c2:
                                st.markdown("**识别意图:**")
                                intents = yitu_data.split('、') if isinstance(yitu_data, str) else []
                                intent_html = "".join([f"<span class='intent-badge'>{i}</span>" for i in intents])
                                st.markdown(intent_html, unsafe_allow_html=True)
                        
                        if show_prompt and prompt_data:
                            with st.expander("📚 参考医学文献/图谱数据", expanded=True):
                                clean_knowledge = prompt_data.replace("<提示>", "").replace("</提示>", "\n")
                                st.markdown(f'<div class="medical-card">{clean_knowledge}</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="disclaimer">⚠️ AI 生成内容仅供参考，不可替代专业医生诊断。</div>', unsafe_allow_html=True)

    # ---------------------------
    # 新消息处理
    # ---------------------------
    if query := st.chat_input("请描述您的症状或问题...", key=f"chat_input_{active_window_index}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🩺"):
            response_placeholder = st.empty()
            status_placeholder = st.empty()
            
            if not check_ollama_connection():
                status_placeholder.warning("⚠️ AI 引擎响应缓慢，切换至基础模式...")
            else:
                status_placeholder.info("✅ 正在连接医疗 AI 引擎...")
            
            # Step 1: 意图
            status_placeholder.markdown("**Step 1:** 正在分析症状描述与查询意图...")
            result_queue = Queue()
            progress_queue = Queue()
            intent_thread = Thread(target=run_intent_recognition_thread, 
                                args=(query, choice, result_queue, progress_queue))
            intent_thread.start()
            
            start_time = time.time()
            while intent_thread.is_alive():
                if not progress_queue.empty():
                    status_placeholder.caption(f"🔄 {progress_queue.get_nowait()}")
                if time.time() - start_time > 30:
                    break
                time.sleep(0.1)
            
            intent_response = result_queue.get() if not result_queue.empty() else "查询疾病简介"
            
            # Step 2: 图谱检索
            status_placeholder.markdown("**Step 2:** 正在检索医学知识图谱...")
            if client:
                prompt, yitu, entities = generate_prompt(intent_response, query, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag)
            else:
                prompt, yitu, entities = f"无法连接数据库。用户问题:{query}", "无", {}
            
            # Step 3: 回答
            status_placeholder.markdown("**Step 3:** 正在生成诊疗建议...")
            last = ""
            try:
                if check_ollama_connection():
                    for chunk in ollama.chat(model=choice, messages=[{'role': 'user', 'content': prompt}], stream=True, options={"temperature": 0.3}):
                        last += chunk['message']['content']
                        response_placeholder.markdown(last + "▌")
                    response_placeholder.markdown(last)
                else:
                    knowledge = re.findall(r'<提示>(.*?)</提示>', prompt)
                    if knowledge:
                        last = "根据知识库为您找到以下信息：\n\n" + "\n".join([f"- {k}" for k in knowledge])
                    else:
                        last = "抱歉，根据已知信息无法回答该问题，建议咨询专业医生。"
                    response_placeholder.markdown(last)
            except Exception as e:
                last = f"生成回答时发生错误: {str(e)}"
                response_placeholder.error(last)

            status_placeholder.empty()

            # 显示结果卡片 (根据开关)
            zhishiku_content = "\n".join(re.findall(r'<提示>(.*?)</提示>', prompt))
            if (show_ent and entities) or (show_int and yitu) or (show_prompt and zhishiku_content):
                st.markdown("---")
                st.caption("🔍 **AI 诊疗分析报告**")
                c1, c2 = st.columns(2)
                
                if show_ent and entities:
                    with c1:
                        st.markdown("**关键医学实体:**")
                        st.markdown(render_entities_pretty(entities), unsafe_allow_html=True)
                
                if show_int and yitu:
                    with c2:
                        st.markdown("**识别意图:**")
                        intents = yitu.split('、') if isinstance(yitu, str) else []
                        intent_html = "".join([f"<span class='intent-badge'>{i}</span>" for i in intents])
                        st.markdown(intent_html, unsafe_allow_html=True)

                if show_prompt and zhishiku_content:
                    with st.expander("📚 参考医学文献/图谱数据", expanded=True):
                        st.markdown(f'<div class="medical-card">{zhishiku_content}</div>', unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">⚠️ AI 生成内容仅供参考，不可替代专业医生诊断。如遇紧急情况请及时就医。</div>', unsafe_allow_html=True)
            
            current_messages.append({
                "role": "assistant", 
                "content": last, 
                "yitu": yitu, 
                "prompt": zhishiku_content, 
                "ent": str(entities)
            })

    st.session_state.messages[active_window_index] = current_messages

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
        main(st.session_state.admin, st.session_state.usname)