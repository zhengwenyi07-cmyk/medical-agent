import json
import os
import hashlib

def hash_password(password: str) -> str:
    """使用 SHA-256 对密码进行哈希加密"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

class Credentials:
    def __init__(self, username, password, is_admin=False, is_hashed=False):
        self.username = username
        # 如果从文件读取的已经是哈希值，则直接赋值；若是新密码则加密
        self.password = password if is_hashed else hash_password(password)
        self.is_admin = is_admin
        self.is_hashed = True

    def verify_password(self, password: str) -> bool:
        """验证密码是否正确"""
        return self.password == hash_password(password)

    def to_dict(self):
        return {
            'username': self.username,
            'password': self.password,
            'is_admin': self.is_admin,
            'is_hashed': self.is_hashed
        }

def create_folder_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def read_credentials(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            # 向下兼容：如果旧数据没有 is_hashed 字段，默认为 False（即明文），加载时会自动触发加密
            return {k: Credentials(
                username=v.get('username'),
                password=v.get('password'),
                is_admin=v.get('is_admin', False),
                is_hashed=v.get('is_hashed', False)
            ) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def write_credentials(file_path, credentials_dict):
    data = {k: v.to_dict() for k, v in credentials_dict.items()}
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

# 文件存储位置
storage_folder = "tmp_data"
storage_file = os.path.join(storage_folder, "user_credentials.json")

# 确保文件夹存在
create_folder_if_not_exist(storage_folder)

# 读取现有的用户数据
credentials = read_credentials(storage_file)

# 如果初始文件为空，则初始化管理员账户
if not credentials:
    admin = Credentials("admin", "admin123", True)
    credentials['admin'] = admin
    write_credentials(storage_file, credentials)