"""
应用配置模块
使用 Pydantic Settings 管理所有环境变量和应用配置
"""
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# 关键：把 .env 加载进 os.environ。
# pydantic-settings 只把值读进 Settings 对象，不写 os.environ；
# 但 langfuse / openai 等库读的是 os.environ，所以这里显式 load_dotenv。
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Settings(BaseSettings):
    """应用全局配置，自动从 .env 文件和环境变量加载"""
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- 应用 ---
    APP_NAME: str = "RAG API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "RAG项目 — 文档上传与管理API（Week5: PostgreSQL + Redis）"
    DEBUG: bool = True

    # --- 文件上传 ---
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set[str] = {
        ".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".xml"
    }

    # --- 分页 ---
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    # --- 认证（Week5 临时方案：正式鉴权前的默认用户）---
    # 只存用户名不存 UUID：UUID 由 seed.sql 随机生成，写死会在换库/重建库后失效
    DEFAULT_USERNAME: str = "demo_user"

    # --- PostgreSQL ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str = "ragpass"
    POSTGRES_DB: str = "ragdb"

    @property
    def DATABASE_URL(self) -> str:
        """SQLAlchemy 兼容的 DSN 格式"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_DSN(self) -> dict:
        """asyncpg create_pool 使用的 dict 参数（更安全，避免密码特殊字符问题）"""
        return {
            "host": self.POSTGRES_HOST,
            "port": self.POSTGRES_PORT,
            "user": self.POSTGRES_USER,
            "password": self.POSTGRES_PASSWORD,
            "database": self.POSTGRES_DB,
        }

    # --- Redis ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # --- LLM 提供者（OpenAI 兼容协议，换提供者只改配置不改代码） ---
    LLM_PROVIDER: str = "deepseek"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-pro"
    LLM_MAX_TOKENS: int = 16000          # V4 Pro 支持 384K 输出；思考模式消耗大，默认 16000
    LLM_REASONING_EFFORT: str = "high"   # low / high / max
    LLM_THINKING_ENABLED: bool = True    # 控制 extra_body 里 thinking.enabled

    # --- 可观测性（LangFuse，提供者无关） ---
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # --- Embedding（Week9，阿里云百炼，OpenAI 兼容） ---
    DASHSCOPE_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "qwen3.7-text-embedding-flash"
    EMBEDDING_DIM: int = 1024
    # 单次 API 调用的最大文本条数。百炼的批量上限会**动态变化**（实测同一模型
    # 曾返回 25 条、后又返回 20 条的超限报错，疑似按负载/文本长度动态收紧）。
    # 超限直接 400 InvalidParameter（不是限流，重试无用），因此取保守值 16。
    EMBEDDING_BATCH_SIZE: int = 16

    # --- 向量索引（Week10）---
    # 每种 chunk 策略一个独立索引目录：data/index/{strategy}/
    INDEX_DIR: Path = Path("data/index")
    CHUNK_STRATEGY: str = "splitter"     # 线上生效的策略名
    # 检索返回条数 → 阈值过滤 → 最终喂给 LLM 的条数
    RAG_RETRIEVE_K: int = 20             # FAISS 单次取回条数（多个配置复用同一次检索）
    RAG_FINAL_K: int = 5                 # 最终上下文条数
    RAG_MIN_SCORE: float = 0.0           # 相似度阈值；0.0 = 不过滤（保证 demo 可跑通）

    # --- 评估（Week10，LLM-as-Judge）---
    EVAL_DIR: Path = Path("data/eval")
    CORPUS_DIR: Path = Path("data/corpus")
    # 判官调用参数：关思考 + 温度 0 —— 对推理模型而言关思考比降温度对稳定性的贡献大得多，
    # 且让调用快 5-10 倍。judge 用低 effort 即可。
    JUDGE_THINKING_ENABLED: bool = False
    JUDGE_TEMPERATURE: float = 0.0
    JUDGE_REASONING_EFFORT: str = "low"
    JUDGE_MODEL: str = ""                # 空 = 复用 DEEPSEEK_MODEL
    JUDGE_PROMPT_VERSION: str = "judge-v1"   # 换 prompt 必须改版本号（旧缓存与分数作废）
    # 评测时的生成参数：同样关思考，否则 50 题 × N 配置会跑到天亮
    EVAL_GEN_THINKING_ENABLED: bool = False


settings = Settings()
