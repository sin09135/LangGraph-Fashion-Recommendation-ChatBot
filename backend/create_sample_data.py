import json
import os
from datetime import datetime
from sqlalchemy import create_engine, text

# SQLite 데이터베이스 직접 생성
DB_URL = "sqlite:///./fashion_recommendation.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

def create_products_table():
    """상품 테이블 생성"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    brand TEXT,
                    category TEXT,
                    price INTEGER,
                    description TEXT,
                    image_url TEXT,
                    style_tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("✅ 상품 테이블 생성 완료")
    except Exception as e:
        print(f"❌ 상품 테이블 생성 오류: {e}")

def create_reviews_table():
    """리뷰 테이블 생성"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id INTEGER,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    review_text TEXT,
                    reviewer_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            """))
            conn.commit()
            print("✅ 리뷰 테이블 생성 완료")
    except Exception as e:
        print(f"❌ 리뷰 테이블 생성 오류: {e}")

def insert_sample_products():
    """샘플 상품 데이터 삽입"""
    sample_products = [
        {
            "name": "스트라이프 티셔츠",
            "brand": "ZARA",
            "category": "상의",
            "price": 35000,
            "description": "클래식한 스트라이프 패턴의 편안한 티셔츠",
            "image_url": "https://via.placeholder.com/300x400/6366f1/ffffff?text=스트라이프+티셔츠",
            "style_tags": "캐주얼,스트릿,베이직"
        },
        {
            "name": "데님 팬츠",
            "brand": "H&M",
            "category": "하의",
            "price": 45000,
            "description": "편안한 핏의 데님 팬츠",
            "image_url": "https://via.placeholder.com/300x400/10b981/ffffff?text=데님+팬츠",
            "style_tags": "캐주얼,데일리,베이직"
        },
        {
            "name": "화이트 스니커즈",
            "brand": "NIKE",
            "category": "신발",
            "price": 89000,
            "description": "깔끔한 화이트 스니커즈",
            "image_url": "https://via.placeholder.com/300x400/f59e0b/ffffff?text=화이트+스니커즈",
            "style_tags": "스포티,캐주얼,베이직"
        },
        {
            "name": "미니 백",
            "brand": "COACH",
            "category": "가방",
            "price": 120000,
            "description": "엘레간트한 미니 백",
            "image_url": "https://via.placeholder.com/300x400/8b5cf6/ffffff?text=미니+백",
            "style_tags": "오피스,데이트,엘레간트"
        },
        {
            "name": "블라우스",
            "brand": "UNIQLO",
            "category": "상의",
            "price": 28000,
            "description": "오피스룩에 어울리는 블라우스",
            "image_url": "https://via.placeholder.com/300x400/ef4444/ffffff?text=블라우스",
            "style_tags": "오피스,엘레간트,포멀"
        },
        {
            "name": "와이드 팬츠",
            "brand": "COS",
            "category": "하의",
            "price": 65000,
            "description": "트렌디한 와이드 팬츠",
            "image_url": "https://via.placeholder.com/300x400/06b6d4/ffffff?text=와이드+팬츠",
            "style_tags": "트렌디,캐주얼,모던"
        },
        {
            "name": "니트 베스트",
            "brand": "MASSIMO DUTTI",
            "category": "상의",
            "price": 75000,
            "description": "레이어드에 완벽한 니트 베스트",
            "image_url": "https://via.placeholder.com/300x400/84cc16/ffffff?text=니트+베스트",
            "style_tags": "캐주얼,레이어드,베이직"
        },
        {
            "name": "플랫 슈즈",
            "brand": "SAM EDELMAN",
            "category": "신발",
            "price": 95000,
            "description": "편안하고 스타일리시한 플랫 슈즈",
            "image_url": "https://via.placeholder.com/300x400/f97316/ffffff?text=플랫+슈즈",
            "style_tags": "엘레간트,데이트,포멀"
        },
        {
            "name": "크로스백",
            "brand": "FOSSIL",
            "category": "가방",
            "price": 85000,
            "description": "실용적인 크로스백",
            "image_url": "https://via.placeholder.com/300x400/ec4899/ffffff?text=크로스백",
            "style_tags": "캐주얼,데일리,실용적"
        },
        {
            "name": "후드 집업",
            "brand": "ADIDAS",
            "category": "상의",
            "price": 55000,
            "description": "스포티한 후드 집업",
            "image_url": "https://via.placeholder.com/300x400/3b82f6/ffffff?text=후드+집업",
            "style_tags": "스포티,캐주얼,스트릿"
        }
    ]
    
    try:
        with engine.connect() as conn:
            for product in sample_products:
                conn.execute(text("""
                    INSERT INTO products (name, brand, category, price, description, image_url, style_tags)
                    VALUES (:name, :brand, :category, :price, :description, :image_url, :style_tags)
                """), product)
            conn.commit()
            print(f"✅ {len(sample_products)}개의 샘플 상품 데이터 삽입 완료")
    except Exception as e:
        print(f"❌ 샘플 상품 데이터 삽입 오류: {e}")

def insert_sample_reviews():
    """샘플 리뷰 데이터 삽입"""
    sample_reviews = [
        {"product_id": 1, "rating": 5, "review_text": "소재가 부드럽고 핏이 좋아요!", "reviewer_name": "김패션"},
        {"product_id": 1, "rating": 4, "review_text": "색상이 예쁘고 가격도 합리적이에요", "reviewer_name": "이스타일"},
        {"product_id": 2, "rating": 5, "review_text": "편안하고 스타일리시해요", "reviewer_name": "박코디"},
        {"product_id": 2, "rating": 4, "review_text": "핏이 정말 좋아요", "reviewer_name": "최유저"},
        {"product_id": 3, "rating": 5, "review_text": "깔끔하고 편안해요", "reviewer_name": "정신발"},
        {"product_id": 3, "rating": 4, "review_text": "색상이 예쁘고 가격도 괜찮아요", "reviewer_name": "한스니커"},
        {"product_id": 4, "rating": 5, "review_text": "엘레간트하고 실용적이에요", "reviewer_name": "임백팬"},
        {"product_id": 4, "rating": 4, "review_text": "크기가 딱 좋아요", "reviewer_name": "강미니"},
        {"product_id": 5, "rating": 5, "review_text": "오피스에 완벽해요", "reviewer_name": "윤오피스"},
        {"product_id": 5, "rating": 4, "review_text": "소재가 좋고 핏이 예뻐요", "reviewer_name": "조블라우스"}
    ]
    
    try:
        with engine.connect() as conn:
            for review in sample_reviews:
                conn.execute(text("""
                    INSERT INTO reviews (product_id, rating, review_text, reviewer_name)
                    VALUES (:product_id, :rating, :review_text, :reviewer_name)
                """), review)
            conn.commit()
            print(f"✅ {len(sample_reviews)}개의 샘플 리뷰 데이터 삽입 완료")
    except Exception as e:
        print(f"❌ 샘플 리뷰 데이터 삽입 오류: {e}")

def create_sample_data():
    """전체 샘플 데이터 생성"""
    print("🚀 샘플 데이터 생성 시작...")
    
    # 테이블 생성
    create_products_table()
    create_reviews_table()
    
    # 샘플 데이터 삽입
    insert_sample_products()
    insert_sample_reviews()
    
    print("✅ 샘플 데이터 생성 완료!")

if __name__ == "__main__":
    create_sample_data() 