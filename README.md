# AI 패션 추천 시스템 (AI Fashion Recommendation System)

> **LangGraph 기반 멀티턴 대화형 추천 챗봇**  
> 이커머스 추천 시스템 특화 LLM 개발


## 프로젝트 개요

자연어 처리와 이미지 유사도 분석을 통해 개인화된 패션 상품을 추천하는 AI 챗봇입니다. LangGraph와 MCP(Model Context Protocol)를 활용하여 대화형 AI 워크플로우를 구현했습니다. 본 프로젝트의 상세내용은 **portfolio_chabot.pdf**에서 확인할 수 있습니다.


## 무엇을 해결하는가?

### 기존 패션 이커머스의 한계

- **획일적 추천**: 구매 기록 기반의 단순한 추천 알고리즘
- **소통 부재**: 사용자 취향과 상황을 고려하지 못하는 일방향적 서비스
- **복잡한 검색**: 원하는 스타일을 언어로 표현하기 어려운 UI/UX

AI 패션 추천 시스템은 **LangGraph**와 **MCP**를 활용하여 사용자와 자연스럽게 대화하며 취향을 파악하고, 상황에 맞는 개인화된 패션 추천을 제공합니다.

### 핵심 기능

- **대화형 추천**: "따뜻한 느낌으로 데이트하기 좋은 코디" 같은 자연어 요청 이해
- **이미지 유사도 검색**: CLIP 모델 기반 시각적 상품 검색
- **스타일 코디네이션**: 상황별 맞춤 스타일링 조합 제안
- **리뷰 감정 분석**: 상품 평가의 신뢰도 향상

---

## 시스템 아키텍처

### 3계층 구조

#### Frontend Layer

- **React.js**: 사용자 인터페이스
- **실시간 채팅**: WebSocket 기반 대화형 UI

#### Backend Layer  

- **FastAPI**: 고성능 비동기 API 서버
- **LangGraph Workflow**: AI 에이전트 조정 및 대화 흐름 관리
- **OpenAI GPT-4**: 자연어 처리 및 추천 생성

#### Data Layer

- **PostgreSQL**: 사용자/상품 데이터 저장
- **ChromaDB**: 벡터 임베딩 저장소
- **CLIP Embeddings**: 이미지-텍스트 멀티모달 검색

---

## LangGraph 워크플로우

### 지능형 라우팅 시스템

```
사용자 입력 → intent_router → 의도 분석
                    ↓
        ┌──────────────────────────────┐
        ↓                              ↓
  일상대화 Agent                    추천 Agent
(conversation_agent)     (recommendation_generator,
                        coordination_finder,
                        similar_product_finder)
        ↓                              ↓
    output_node ←────── 최종 응답 ←────────
```

| 핵심 기능                    | 설명                       | 기술                        |
| ---------------------------- | -------------------------- | --------------------------- |
| **Intent Router**            | 사용자 의도 분석 및 라우팅 | GPT-4 + 프롬프트 엔지니어링 |
| **Recommendation Generator** | 상황별 상품 추천           | SQL 쿼리 + LLM 추천 생성    |
| **Coordination Finder**      | 스타일링 코디 추천         | 이미지 임베딩 + 스타일 규칙 |
| **Similar Product Finder**   | 유사 상품 검색             | CLIP 벡터 유사도 검색       |

---

## 데이터 수집 및 처리

### 무신사 패션 데이터 크롤링

- **수집 도구**: Selenium, BeautifulSoup, Pandas
- **수집 규모**: 1,200개 상품, 1,400개 이미지, 5개 카테고리
- **처리 과정**: 3단계 파이프라인 (기본정보 → 상세정보 → 리뷰데이터)

### 수집 결과

| 데이터 유형 | 수량    | 활용 목적          |
| ----------- | ------- | ------------------ |
| 상품 정보   | 1,200개 | 추천 알고리즘 학습 |
| 상품 이미지 | 1,400개 | CLIP 임베딩 생성   |
| 사용자 리뷰 | 다수    | 감정 분석 및 평가  |

---

## 한계점 및 향후 개선

### 한계점

- **감성 키워드 해석**: "따뜻한", "시크한" 등 추상적 표현의 정확도 개선 필요
- **복잡한 스타일링**: 다양한 아이템 조합 시 일관성 있는 코디 제안의 어려움
- **실시간 트렌드 반영**: 빠르게 변화하는 패션 트렌드 대응 부족

### 개선 계획

1. **패션 도메인 특화 모델** 파인튜닝
2. **실시간 트렌드 API** 연동
3. **사용자 피드백 학습** 시스템 구축

---

## 기술 스택

### 주요 기술

- **AI/ML**: LangGraph, OpenAI GPT-4, CLIP
- **Backend**: Python 3.11, FastAPI, PostgreSQL  
- **Frontend**: React.js, WebSocket
- **Vector DB**: ChromaDB
- **Data Processing**: Selenium, BeautifulSoup, Pandas
