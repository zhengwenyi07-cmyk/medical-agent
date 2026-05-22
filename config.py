"""统一配置读取模块 — 供非 Streamlit 脚本读取 .streamlit/secrets.toml"""
import os

_SECRETS_PATH = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")


def _parse_toml(path):
    """轻量 TOML 解析器，解析 [section] 和 key = value"""
    config = {}
    current_section = config
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1]
                current_section = config.setdefault(section_name, {})
            elif "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                current_section[key] = value
    return config


def get_config():
    """读取 secrets.toml 配置，优先 tomllib/tomli，回退到内置解析器"""
    if os.path.exists(_SECRETS_PATH):
        try:
            import tomllib
            with open(_SECRETS_PATH, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import tomli
                with open(_SECRETS_PATH, "rb") as f:
                    return tomli.load(f)
            except ImportError:
                return _parse_toml(_SECRETS_PATH)
    return {}


def get_neo4j_config():
    """获取 Neo4j 连接配置"""
    config = get_config()
    neo4j = config.get("neo4j", {})
    return {
        "uri": neo4j.get("uri", "bolt://localhost:7687"),
        "user": neo4j.get("user", "neo4j"),
        "password": neo4j.get("password", ""),
    }


def get_ollama_config():
    """获取 Ollama 连接配置"""
    config = get_config()
    ollama = config.get("ollama", {})
    return {
        "base_url": ollama.get("base_url", "http://localhost:11434"),
    }
