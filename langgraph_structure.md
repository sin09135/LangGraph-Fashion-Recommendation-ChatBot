# LangGraph 구조 시각화

## 🏗️ 전체 시스템 아키텍처

```mermaid
graph TD
    A[사용자 입력] --> B[intent_router]
    B --> C{의도 분류}
    
    C -->|recommendation| D[recommendation_generator]
    C -->|coordination| E[coordination_finder]
    C -->|similar_product_finder| F[similar_product_finder]
    C -->|review_search| G[review_search_node]
    C -->|image_search| H[image_similarity_search]
    C -->|chat| I[conversation_agent]
    C -->|filter_existing| J[filter_existing_recommendations]
    
    D --> K[output_node]
    E --> K
    F --> K
    G --> L[review_based_recommendation]
    L --> K
    H --> K
    I --> K
    J --> K
    
    K --> M[최종 응답]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style K fill:#f3e5f5
    style M fill:#e8f5e8
```

## 🔄 상세 워크플로우

### 1. Intent Router (의도 분류)
```mermaid
graph LR
    A[사용자 입력] --> B[LLM 의도 분석]
    B --> C{의도 결정}
    
    C -->|상품 추천| D[recommendation]
    C -->|코디 추천| E[coordination]
    C -->|유사 상품| F[similar_product_finder]
    C -->|리뷰 검색| G[review_search]
    C -->|이미지 검색| H[image_search]
    C -->|일반 대화| I[chat]
    C -->|필터링| J[filter_existing]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#fff8e1
```

### 2. Recommendation Generator (일반 추천)
```mermaid
graph TD
    A[사용자 입력] --> B[text_filter_parser]
    B --> C[조건 추출]
    C --> D[PostgreSQL 검색]
    D --> E[상품 필터링]
    E --> F[결과 정렬]
    F --> G[output_node]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style G fill:#f3e5f5
```

### 3. Coordination Finder (코디 추천)
```mermaid
graph TD
    A[상품 번호] --> B[상품 정보 조회]
    B --> C[이미지 임베딩]
    C --> D[다른 카테고리 상품 검색]
    D --> E[코디네이션 점수 계산]
    E --> F[상위 상품 선택]
    F --> G[output_node]
    
    style A fill:#e1f5fe
    style E fill:#fff3e0
    style G fill:#f3e5f5
```

### 4. Review Search Node (리뷰 검색)
```mermaid
graph TD
    A[상품 번호/키워드] --> B[PostgreSQL 리뷰 검색]
    B --> C[리뷰 데이터 수집]
    C --> D[LLM 리뷰 요약]
    D --> E[review_based_recommendation]
    E --> F[output_node]
    
    style A fill:#e1f5fe
    style D fill:#fff3e0
    style F fill:#f3e5f5
```

### 5. Image Similarity Search (이미지 검색)
```mermaid
graph TD
    A[이미지 업로드] --> B[CLIP 임베딩]
    B --> C[ChromaDB 벡터 검색]
    C --> D[유사도 계산]
    D --> E[상위 결과 선택]
    E --> F[output_node]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style F fill:#f3e5f5
```

## 📊 데이터 흐름

### PostgreSQL 데이터베이스
```mermaid
erDiagram
    products {
        int product_id PK
        string product_name
        string category
        float price
        string brand_kr
        string image_url
        string tags
    }
    
    sessions {
        string session_id PK
        timestamp created_at
        timestamp updated_at
    }
    
    liked_products {
        int id PK
        string session_id FK
        int product_id FK
        timestamp created_at
    }
    
    product_reviews {
        int id PK
        int product_id FK
        float rating
        string content
        string user_name
        timestamp review_date
    }
    
    products ||--o{ liked_products : "사용자가 좋아요"
    products ||--o{ product_reviews : "상품 리뷰"
    sessions ||--o{ liked_products : "세션별 좋아요"
```

### ChromaDB 벡터 데이터베이스
```mermaid
graph LR
    A[상품 이미지] --> B[CLIP 임베딩]
    C[리뷰 텍스트] --> D[텍스트 임베딩]
    B --> E[ChromaDB - product_images]
    D --> F[ChromaDB - reviews]
    
    style A fill:#e1f5fe
    style C fill:#e1f5fe
    style E fill:#fff3e0
    style F fill:#fff3e0
```

## 🎯 주요 노드 설명

### 1. Intent Router
- **기능**: 사용자 입력의 의도를 분석
- **입력**: 사용자 메시지
- **출력**: 의도 분류 결과 (recommendation/coordination/similar_product_finder/review_search/image_search/chat)
- **알고리즘**: LLM 기반 의도 분석

### 2. Recommendation Generator
- **기능**: 일반적인 상품 추천
- **입력**: 상품 조건 (카테고리, 가격, 브랜드 등)
- **출력**: 조건에 맞는 상품 목록
- **알고리즘**: PostgreSQL 쿼리 기반 필터링

### 3. Coordination Finder
- **기능**: 특정 상품과 코디하기 좋은 상품 추천
- **입력**: 상품 번호
- **출력**: 코디네이션 점수가 높은 상품들
- **알고리즘**: 이미지 유사도 + 가격 호환성 + 브랜드 호환성 + 카테고리 조합 점수

### 4. Review Search Node
- **기능**: 상품 리뷰 검색 및 요약
- **입력**: 상품 번호 또는 키워드
- **출력**: 리뷰 요약 및 평점 분석
- **알고리즘**: PostgreSQL 검색 + LLM 요약

### 5. Image Similarity Search
- **기능**: 이미지 기반 유사 상품 검색
- **입력**: 업로드된 이미지
- **출력**: 유사한 상품 목록
- **알고리즘**: CLIP 임베딩 + ChromaDB 벡터 검색

### 6. Output Node
- **기능**: 최종 응답 포맷팅
- **입력**: 각 노드의 결과
- **출력**: 사용자에게 전달할 응답
- **포맷**: 상품 정보 + 설명 + 추천 이유

## 🔧 설정 및 환경 변수

### 필수 환경 변수
```bash
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/fashion_db
CHROMA_DB_PATH=./chroma_db
```

### 주요 설정값
```python
# config.py
MAX_RECOMMENDATIONS = 10
DEFAULT_DISPLAY_COUNT = 10
SIMILARITY_THRESHOLD = 0.7
COORDINATION_WEIGHTS = {
    'image_similarity': 0.4,
    'price_compatibility': 0.3,
    'brand_compatibility': 0.2,
    'category_combination': 0.1
}
```

## 🚀 성능 최적화

### 1. 캐싱 전략
- CLIP 모델 임베딩 캐싱
- PostgreSQL 쿼리 결과 캐싱
- LLM 응답 캐싱

### 2. 비동기 처리
- 이미지 업로드 비동기 처리
- LLM 호출 비동기 처리
- 데이터베이스 쿼리 비동기 처리

### 3. 벡터 검색 최적화
- ChromaDB 인덱싱
- 임베딩 차원 최적화
- 검색 범위 제한

## 🔍 디버깅 및 모니터링

### 로그 레벨
```python
# 각 노드별 상세 로그
🔍 intent_router - 의도 분석 결과
🤖 의도 분석: {intent}
📝 추천 상품 {count}개 생성
💾 output_node에서 previous_recommendations 저장
```

### 성능 메트릭
- 응답 시간 측정
- 각 노드별 처리 시간
- 데이터베이스 쿼리 성능
- LLM 호출 성능 