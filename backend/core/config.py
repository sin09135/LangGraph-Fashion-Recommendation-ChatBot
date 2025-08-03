import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class Config:
    """애플리케이션 설정"""
    
    # OpenAI 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # 데이터베이스 설정
    DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://username:password@localhost:5432/fashion_recommendation")
    
    # ChromaDB 설정
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # CLIP 모델 설정
    CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    
    # 서버 설정
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # CORS 설정
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    
    # 추천 설정
    MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "10"))
    DEFAULT_PRICE_RANGES = {
        "low": 40000,
        "medium": 100000,
        "high": float('inf')
    }

# 전역 설정 인스턴스
config = Config() 