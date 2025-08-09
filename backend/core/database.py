from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import chromadb
from chromadb.config import Settings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import DB_URL, CHROMA_DB_PATH

# PostgreSQL 엔진 생성
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ChromaDB 클라이언트 생성
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_db():
    """데이터베이스 세션 생성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_chroma_collection(collection_name: str):
    """ChromaDB 컬렉션 가져오기"""
    return chroma_client.get_or_create_collection(collection_name)

# 기본 컬렉션들
product_collection = get_chroma_collection("products")
review_collection = get_chroma_collection("reviews")
image_collection = get_chroma_collection("product_images")

def get_database_session():
    """데이터베이스 세션 반환"""
    return SessionLocal() 