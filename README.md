# AI 패션 추천 챗봇 (MCP 통합)

AI 기반 패션 상품 추천 챗봇입니다. 자연어 대화를 통해 개인화된 패션 추천을 제공하며, 이미지 업로드, 텍스트 기반 검색, 유사 상품 추천, 코디네이션 추천, 리뷰 분석 등의 기능을 제공합니다.

** MCP(Model Context Protocol) 통합**: 이제 모든 추천 기능이 표준화된 MCP 도구로 제공됩니다.

## 🎯 주요 기능

#### **자연어 대화 기반 추천**
- "버뮤다 팬츠 4만원 미만으로 추천해줘"
- "1번 상품과 코디하기 좋은 상품 추천해줘"
- "이 상품과 비슷한 스타일 추천해줘"
- "1번 상품 리뷰는 어때?"

#### **이미지 기반 검색**
- 사진 업로드로 유사한 패션 상품 검색
- CLIP 모델을 사용한 정확한 이미지 분석

#### **코디네이션 추천**
- 특정 상품과 어울리는 조합 추천
- 스타일, 가격, 브랜드 호환성 고려

#### **리뷰 분석**
- 상품별 리뷰 요약 및 분석
- 사용자 평점 기반 추천

#### **개인화 기능**
- 좋아하는 상품 저장 및 관리
- 개인 취향 기반 추천

### 🔧 **MCP 도구들**
- `fashion_recommend` - 상품 추천
- `fashion_coordination` - 코디네이션 추천
- `fashion_similar_search` - 유사 상품 검색
- `fashion_review_analysis` - 리뷰 분석
- `fashion_image_search` - 이미지 기반 검색
- `fashion_user_preferences` - 사용자 선호도 관리

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 환경 변수 설정
export OPENAI_API_KEY=your_openai_api_key
export DATABASE_URL=postgresql://username:password@localhost:5432/fashion_db

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행 방법

#### 🎯 **전체 시스템 실행 (추천)**
```bash
./run_all.sh
```
- MCP 서버, 백엔드, 프론트엔드를 모두 자동으로 실행
- 프로세스 모니터링 및 자동 종료 포함

#### 🔧 **개발 모드 실행**
```bash
./run_dev.sh
```
- 필요한 서비스만 선택적으로 실행 가능
- 개발 중 유용한 옵션들 제공

#### 🧪 **MCP 테스트**
```bash
./test_mcp.sh
```
- MCP 서버 연결 및 도구 테스트
- 각 기능별 개별 테스트 가능

#### 📦 **개별 서비스 실행**
```bash
# MCP 서버만 실행
./run_mcp.sh

# 백엔드만 실행
./run_backend.sh

# 프론트엔드만 실행
./run_frontend.sh
```

### 3. 브라우저에서 접속
```
http://localhost:3000
```

## 시스템 아키텍처

### MCP 통합 아키텍처
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   MCP Client    │    │   MCP Server    │
│   (React)       │◄──►│   (Python)      │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   FastAPI       │    │   LangGraph     │
                       │   Backend       │    │   Nodes         │
                       └─────────────────┘    └─────────────────┘
```

### LangGraph 워크플로우
```
사용자 입력 → Intent Router → 특화 노드 → Output Node
```

#### **주요 노드들:**
- **`intent_router`**: 사용자 의도 분석 (추천/코디/유사상품/리뷰/이미지)
- **`recommendation_generator`**: 일반 상품 추천
- **`coordination_finder`**: 코디네이션 추천
- **`similar_product_finder`**: 유사 상품 검색
- **`review_search_node`**: 리뷰 검색 및 분석
- **`image_similarity_search`**: 이미지 기반 검색
- **`conversation_agent`**: 일반 대화 처리

### 데이터베이스 구조
- **PostgreSQL**: 상품 정보, 사용자 세션, 좋아요 관리
- **ChromaDB**: 리뷰 임베딩, 이미지 임베딩 벡터 저장

## 프로젝트 구조

```
fashion_rec_system/
├── backend/                 # FastAPI 백엔드
│   ├── main.py             # 서버 엔트리 포인트 (MCP API 포함)
│   ├── mcp_api.py          # MCP API 엔드포인트
│   ├── nodes.py            # LangGraph 노드들
│   ├── config.py           # 설정 파일
│   ├── llm_service.py      # LLM 서비스
│   └── core/               # 핵심 모듈
├── frontend/               # React 프론트엔드
│   ├── src/
│   │   ├── App.js          # 메인 앱 컴포넌트
│   │   └── components/     # React 컴포넌트들
│   │       ├── ChatInterface.js    # 채팅 인터페이스
│   │       ├── MCPInterface.js    # MCP 인터페이스
│   │       ├── ProductGrid.js      # 상품 그리드
│   │       ├── LikedProductsView.js # 좋아요 상품 보기
│   │       └── Header.js           # 헤더 컴포넌트
│   └── package.json        # Node.js 의존성
├── utils/                  # 유틸리티 함수들
├── chroma_db/             # 벡터 데이터베이스
├── mcp_server.py          # MCP 서버 메인 파일
├── mcp_client.py          # MCP 클라이언트
├── mcp_config.json        # MCP 설정 파일
├── run_all.sh             # 전체 시스템 실행 스크립트
├── run_dev.sh             # 개발 모드 실행 스크립트
├── run_mcp.sh             # MCP 서버 실행 스크립트
├── test_mcp.sh            # MCP 테스트 스크립트
├── run_backend.sh         # 백엔드 실행 스크립트
├── run_frontend.sh        # 프론트엔드 실행 스크립트
├── requirements.txt        # Python 의존성
└── README.md              # 프로젝트 문서
```

## 기술 스택

### 백엔드
- **FastAPI**: 웹 프레임워크
- **LangGraph**: 대화형 AI 워크플로우
- **MCP**: Model Context Protocol
- **OpenAI GPT**: 자연어 처리
- **CLIP**: 이미지 임베딩 모델
- **PostgreSQL**: 관계형 데이터베이스
- **ChromaDB**: 벡터 데이터베이스
- **SQLAlchemy**: ORM

### 프론트엔드
- **React**: 사용자 인터페이스
- **Styled Components**: CSS-in-JS 스타일링
- **Lucide React**: 아이콘 라이브러리

## 사용 예시

### 1. 일반 상품 추천
```
사용자: "버뮤다 팬츠 4만원 미만으로 추천해줘"
시스템: 버뮤다 팬츠 중 4만원 미만 상품 10개 추천
```

### 2. 코디네이션 추천
```
사용자: "1번 상품과 코디하기 좋은 상품 추천해줘"
시스템: 1번 상품과 어울리는 상의, 신발, 액세서리 추천
```

### 3. 유사 상품 검색
```
사용자: "이 상품과 비슷한 스타일 추천해줘"
시스템: 이미지나 상품과 유사한 스타일의 다른 상품 추천
```

### 4. 리뷰 분석
```
사용자: "1번 상품 리뷰는 어때?"
시스템: 해당 상품의 리뷰 요약 및 평점 분석 제공
```
## 📋 API 엔드포인트

### 기존 API
- `POST /chat`: 메인 채팅 엔드포인트
- `POST /upload-image`: 이미지 업로드
- `POST /like`: 상품 좋아요 추가
- `GET /likes/{session_id}`: 좋아요 상품 조회
- `GET /session/{session_id}`: 세션 정보 조회
- `GET /health`: 서버 상태 확인

### MCP API
- `GET /api/mcp/tools`: 사용 가능한 도구 목록
- `GET /api/mcp/resources`: 사용 가능한 리소스 목록
- `POST /api/mcp/call-tool`: 도구 호출
- `GET /api/mcp/health`: MCP 서버 상태 확인
- `WebSocket /api/mcp/ws`: 실시간 MCP 통신

## 개발 워크플로우

1. **MCP 도구 개발**: `mcp_server.py`에 새로운 도구 추가
2. **백엔드 개발**: `backend/` 폴더에서 Python 코드 수정
3. **프론트엔드 개발**: `frontend/` 폴더에서 React 코드 수정
4. **LangGraph 노드 추가**: `backend/nodes.py`에 새로운 노드 구현
5. **테스트**: 각 기능별 테스트 실행
6. **배포**: Git을 통한 버전 관리

- [MCP 상세 문서](./MCP_README.md)
- [LangGraph 구조 문서](./langgraph_structure.md)
=======
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
>>>>>>> 381b0d5c8dc4d64b5406c70c20a5d07cfb54bbea
