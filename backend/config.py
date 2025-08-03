import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# OpenAI 설정
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# 데이터베이스 설정
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://username:password@localhost:5432/fashion_recommendation")

# ChromaDB 설정
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# 이미지 설정
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "/Users/kimsinwoo/Desktop/LLM/data")
MAX_IMAGE_RESULTS = int(os.getenv("MAX_IMAGE_RESULTS", "10"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "10"))

# 서버 설정
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# CORS 설정
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# CLIP 모델 설정
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# 추천 설정
DEFAULT_PRICE_RANGES = {
    "low": int(os.getenv("PRICE_LOW", "50000")),
    "medium": int(os.getenv("PRICE_MEDIUM", "150000")),
    "high": float(os.getenv("PRICE_HIGH", "inf"))
}

# 표시 설정
DEFAULT_DISPLAY_COUNT = int(os.getenv("DEFAULT_DISPLAY_COUNT", "3"))

# 이미지 유사도 설정
DEFAULT_SIMILARITY_SCORE = float(os.getenv("DEFAULT_SIMILARITY_SCORE", "0.5"))

# 스타일 옵션
AVAILABLE_STYLES = os.getenv("AVAILABLE_STYLES", "캐주얼,스포티,오피스,데이트,스트릿").split(",")

# 피드백 키워드
FEEDBACK_KEYWORDS = {
    "positive": os.getenv("POSITIVE_FEEDBACK_KEYWORDS", "좋아,괜찮아,마음에 들어,좋아요").split(","),
    "negative": os.getenv("NEGATIVE_FEEDBACK_KEYWORDS", "비싸요,싸요,별로야,마음에 안들어").split(","),
    "style_change": os.getenv("STYLE_CHANGE_KEYWORDS", "다른 스타일,다른 색,다른 브랜드").split(",")
}

# 유사 상품 키워드
SIMILAR_PRODUCT_KEYWORDS = os.getenv("SIMILAR_PRODUCT_KEYWORDS", "유사한,비슷한,같은 스타일,같은 느낌,이런 스타일,이런 느낌").split(",") 