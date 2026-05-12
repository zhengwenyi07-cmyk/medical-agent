import streamlit as st
from user_data_storage import credentials, write_credentials, storage_file, Credentials
from webui2 import main as app_main
import time

# ==========================================
# 页面样式配置
# ==========================================
def load_login_css():
    st.markdown("""
        <style>
        /* 全局背景色 - 与主系统一致的淡灰蓝 */
        .stApp {
            background-color: #F4F8FB;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /* 隐藏登录页面的侧边栏汉堡菜单，保持界面纯净 */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* 标题样式 */
        .login-header {
            text-align: center;
            color: #008B8B;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .login-subheader {
            text-align: center;
            color: #7F8C8D;
            font-size: 0.9em;
            margin-bottom: 30px;
        }

        /* 输入框样式优化 */
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            padding: 10px;
        }
        .stTextInput input:focus {
            border-color: #008B8B;
            box-shadow: 0 0 5px rgba(0, 139, 139, 0.2);
        }

        /* 按钮样式 - 宽度拉满，医疗青色 */
        .stButton button {
            width: 100%;
            background-color: #008B8B;
            color: white;
            border-radius: 8px;
            padding: 10px 0;
            font-weight: 600;
            border: none;
            margin-top: 10px;
        }
        .stButton button:hover {
            background-color: #006666;
            color: white;
            border: none;
        }

        /* 标签页样式 */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            border-bottom: 1px solid #E0E0E0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            color: #95A5A6;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            color: #008B8B;
            border-bottom: 2px solid #008B8B;
        }
        
        /* 底部版权信息 */
        .footer {
            text-align: center;
            color: #BDC3C7;
            font-size: 0.8em;
            margin-top: 50px;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 状态初始化
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'admin' not in st.session_state:
    st.session_state.admin = False
if 'usname' not in st.session_state:
    st.session_state.usname = ""

# ==========================================
# 登录/注册 逻辑
# ==========================================

def login_form():
    with st.form("login_form"):
        st.caption("请输入您的账号信息以访问医疗系统")
        username = st.text_input("用户名", placeholder="请输入用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        
        submit = st.form_submit_button("登 录")
        
        if submit:
            if not username or not password:
                st.warning("⚠️ 请输入用户名和密码")
                return

            user_cred = credentials.get(username)
            if user_cred and user_cred.password == password:
                st.success("✅ 登录成功！正在跳转...")
                st.session_state.logged_in = True
                st.session_state.admin = user_cred.is_admin
                st.session_state.usname = username
                time.sleep(0.5) # 给一点时间显示成功动画
                st.rerun()
            else:
                st.error("❌ 用户名或密码错误")

def register_form():
    with st.form("register_form"):
        st.caption("创建新账户以使用问答服务")
        new_username = st.text_input("设置用户名", placeholder="建议使用字母或数字")
        new_password = st.text_input("设置密码", type="password", placeholder="请设置安全密码")
        confirm_password = st.text_input("确认密码", type="password", placeholder="请再次输入密码")
        
        register_submit = st.form_submit_button("注 册")
        
        if register_submit:
            if not new_username or not new_password:
                st.warning("⚠️ 请填写完整信息")
                return
            
            if new_password != confirm_password:
                st.error("❌ 两次输入的密码不一致")
                return

            if new_username in credentials:
                st.error("❌ 用户名已存在，请更换")
            else:
                new_user = Credentials(new_username, new_password, is_admin=False)
                credentials[new_username] = new_user
                write_credentials(storage_file, credentials)
                st.success(f"🎉 用户 {new_username} 注册成功！请切换至登录页。")

# ==========================================
# 主入口
# ==========================================

if __name__ == "__main__":
    # 如果已登录，直接进入主程序
    if st.session_state.logged_in:
        app_main(st.session_state.admin, st.session_state.usname)
    else:
        # 加载登录页样式
        load_login_css()
        
        # 使用列布局将内容居中 (左空-中实-右空)
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col2:
            # 顶部 Logo 区域
            st.markdown("<h1 class='login-header'>🏥 智能医疗问答系统</h1>", unsafe_allow_html=True)
            st.markdown("<p class='login-subheader'>AI Medical Assistant & Knowledge Graph</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 使用 Tabs 替代原来的 Sidebar 切换，体验更像原生 App
            tab_login, tab_register = st.tabs(["🔐 登 录", "📝 注 册"])
            
            with tab_login:
                login_form()
            
            with tab_register:
                register_form()
            
            # 底部版权
            st.markdown("<div class='footer'>© 2024 AI Medical System. All Rights Reserved.</div>", unsafe_allow_html=True)