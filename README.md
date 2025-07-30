# AI 패션 추천 시스템

AI 기반 패션 상품 추천 시스템입니다. 이미지 업로드, 텍스트 기반 검색, 유사 상품 추천 등의 기능을 제공합니다.

## 🚀 빠른 시작

### 1. 백엔드 실행
```bash
# 실행 권한 부여
chmod +x run_backend.sh

# 백엔드 서버 시작
./run_backend.sh
```

### 2. 프론트엔드 실행
```bash
# 실행 권한 부여
chmod +x run_frontend.sh

# 프론트엔드 서버 시작
./run_frontend.sh
```

## 📁 프로젝트 구조

```
fashion_rec_system/
├── backend/                 # FastAPI 백엔드
│   ├── main.py             # 서버 엔트리 포인트
│   ├── nodes.py            # LangGraph 노드들
│   ├── config.py           # 설정 파일
│   └── llm_service.py      # LLM 서비스
├── frontend/               # React 프론트엔드
│   ├── src/
│   │   ├── App.js          # 메인 앱 컴포넌트
│   │   └── components/     # React 컴포넌트들
│   └── package.json        # Node.js 의존성
├── utils/                  # 유틸리티 함수들
├── chroma_db/             # 벡터 데이터베이스
├── requirements.txt        # Python 의존성
├── run_backend.sh         # 백엔드 실행 스크립트
├── run_frontend.sh        # 프론트엔드 실행 스크립트
└── README.md              # 프로젝트 문서
```

## 🔧 주요 기능

### 1. 이미지 기반 추천
- CLIP 모델을 사용한 이미지 임베딩
- 유사한 패션 상품 추천
- 카테고리 기반 필터링

### 2. 텍스트 기반 추천
- 자연어 처리 기반 상품 검색
- 스타일, 색상, 브랜드 등 조건별 필터링

### 3. 유사 상품 추천
- 이전 추천 결과에서 특정 상품과 유사한 상품 검색
- "1번 상품과 유사한 상품" 형태의 요청 처리

### 4. 가격 필터링
- "4만원 미만으로 적용" 형태의 가격 기반 필터링

## 🛠️ 기술 스택

### 백엔드
- **FastAPI**: 웹 프레임워크
- **LangGraph**: 대화형 AI 워크플로우
- **CLIP**: 이미지 임베딩 모델
- **PostgreSQL**: 관계형 데이터베이스
- **ChromaDB**: 벡터 데이터베이스

### 프론트엔드
- **React**: 사용자 인터페이스
- **JavaScript**: 클라이언트 로직

## 📋 환경 설정

### 1. Python 환경
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. Node.js 환경
```bash
# Node.js 16+ 설치 필요
cd frontend
npm install
```

### 3. 데이터베이스 설정
- PostgreSQL 설치 및 실행
- `.env` 파일에 데이터베이스 연결 정보 설정

## 🔄 개발 워크플로우

1. **백엔드 개발**: `backend/` 폴더에서 Python 코드 수정
2. **프론트엔드 개발**: `frontend/` 폴더에서 React 코드 수정
3. **테스트**: 각 기능별 테스트 파일 실행
4. **배포**: Git을 통한 버전 관리

## 📝 API 엔드포인트

- `POST /chat`: 메인 채팅 엔드포인트
- `GET /health`: 서버 상태 확인

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.