# MCP 패션 추천 시스템

이 프로젝트는 기존의 패션 추천 시스템을 **MCP(Model Context Protocol)** 방식으로 변환한 것입니다. MCP는 AI 모델이 다양한 도구와 서버에 접근할 수 있도록 하는 표준화된 프로토콜입니다.

## 🎯 MCP 변환의 장점

### 1. **표준화된 인터페이스**
- 모든 도구가 일관된 방식으로 정의됨
- 새로운 도구 추가가 용이함
- 다른 MCP 호환 시스템과 연동 가능

### 2. **확장성**
- 새로운 추천 알고리즘을 도구로 쉽게 추가
- 다양한 AI 모델과의 연동
- 마이크로서비스 아키텍처 지원

### 3. **유연성**
- 동적 도구 로딩
- 실시간 도구 상태 모니터링
- 세션 기반 상태 관리

## 🏗️ MCP 아키텍처

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

## 📁 MCP 관련 파일 구조

```
fashion_rec_system/
├── mcp_server.py          # MCP 서버 메인 파일
├── mcp_client.py          # MCP 클라이언트
├── backend/mcp_api.py     # MCP API 엔드포인트
├── frontend/src/components/
│   └── MCPInterface.js    # MCP 프론트엔드 인터페이스
├── mcp_config.json        # MCP 설정 파일
├── run_mcp.sh            # MCP 서버 실행 스크립트
└── MCP_README.md         # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 환경 변수 설정
export OPENAI_API_KEY=your_openai_api_key
export DATABASE_URL=postgresql://username:password@localhost:5432/fashion_db

# 의존성 설치
pip install -r requirements.txt
```

### 2. MCP 서버 실행

```bash
# 방법 1: 스크립트 사용
./run_mcp.sh

# 방법 2: 직접 실행
python mcp_server.py
```

### 3. 백엔드 서버 실행 (MCP API 포함)

```bash
# 기존 백엔드 실행 (MCP API 포함)
./run_backend.sh
```

### 4. 프론트엔드 실행

```bash
# 프론트엔드 실행
./run_frontend.sh
```

## 🛠️ 사용 가능한 MCP 도구들

### 1. **fashion_recommend** - 상품 추천
```json
{
  "category": "상의",
  "price_range": "1만원-5만원",
  "brand": "유니클로",
  "style": "캐주얼",
  "limit": 10
}
```

### 2. **fashion_coordination** - 코디네이션 추천
```json
{
  "product_id": 1,
  "categories": ["하의", "신발"],
  "budget": "10만원 이하",
  "limit": 5
}
```

### 3. **fashion_similar_search** - 유사 상품 검색
```json
{
  "product_id": 1,
  "image_url": "https://example.com/image.jpg",
  "similarity_threshold": 0.7,
  "limit": 10
}
```

### 4. **fashion_review_analysis** - 리뷰 분석
```json
{
  "product_id": 1,
  "keyword": "품질",
  "rating_filter": 4.0,
  "limit": 20
}
```

### 5. **fashion_image_search** - 이미지 기반 검색
```json
{
  "image_data": "base64_encoded_image_data",
  "category_filter": "상의",
  "similarity_threshold": 0.7,
  "limit": 10
}
```

### 6. **fashion_user_preferences** - 사용자 선호도 관리
```json
{
  "action": "set",
  "preferences": {
    "preferred_brands": ["유니클로", "ZARA"],
    "price_range": "1만원-10만원",
    "style": "캐주얼"
  },
  "session_id": "user123"
}
```

## 🔧 MCP 서버 개발

### 새로운 도구 추가하기

1. **MCP 서버에 도구 정의 추가** (`mcp_server.py`):

```python
Tool(
    name="fashion_new_tool",
    description="새로운 패션 도구",
    inputSchema={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "매개변수 1"},
            "param2": {"type": "number", "description": "매개변수 2"}
        },
        "required": ["param1"]
    }
)
```

2. **도구 처리 메서드 추가**:

```python
async def _handle_new_tool(self, arguments: Dict[str, Any], state: FashionRecommendationState) -> List[TextContent]:
    """새로운 도구 처리"""
    # 도구 로직 구현
    result = await your_new_function(arguments)
    
    return [TextContent(
        type="text",
        text=result
    )]
```

3. **도구 호출 매핑 추가**:

```python
if name == "fashion_new_tool":
    return await self._handle_new_tool(arguments, state)
```

### MCP 클라이언트 확장

```python
async def new_tool_method(self, **kwargs) -> str:
    """새로운 도구 메서드"""
    request = FashionRecommendationRequest(
        tool_name="fashion_new_tool",
        arguments=kwargs,
        session_id="default"
    )
    
    return await self.call_tool(request)
```

## 📊 MCP 모니터링

### 서버 상태 확인

```bash
# 헬스 체크
curl http://localhost:8001/api/mcp/health

# 도구 목록 조회
curl http://localhost:8001/api/mcp/tools

# 리소스 목록 조회
curl http://localhost:8001/api/mcp/resources
```

### WebSocket 연결 테스트

```javascript
const ws = new WebSocket('ws://localhost:8001/api/mcp/ws');

ws.onopen = () => {
    console.log('MCP WebSocket 연결됨');
    
    // 도구 목록 요청
    ws.send(JSON.stringify({
        type: 'get_tools'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('MCP 메시지:', data);
};
```

## 🔍 디버깅

### 로그 확인

```bash
# MCP 서버 로그
tail -f mcp_server.log

# 백엔드 로그
tail -f backend.log
```

### 연결 상태 확인

```python
# MCP 클라이언트에서
async with FashionRecommendationAPI() as api:
    tools = await api.get_tools_info()
    print(f"사용 가능한 도구: {len(tools)}개")
    
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")
```

## 🚀 성능 최적화

### 1. **연결 풀링**
- MCP 클라이언트 연결 재사용
- 세션 상태 캐싱

### 2. **비동기 처리**
- 모든 MCP 호출을 비동기로 처리
- 동시 요청 처리 최적화

### 3. **캐싱 전략**
- 도구 결과 캐싱
- 사용자 선호도 캐싱

## 🔐 보안 고려사항

### 1. **인증 및 권한**
```python
# 세션 기반 인증
async def authenticate_session(session_id: str) -> bool:
    # 세션 검증 로직
    pass
```

### 2. **입력 검증**
```python
# Pydantic 모델을 통한 입력 검증
class ToolInput(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    session_id: str
```

### 3. **API 키 관리**
```bash
# 환경 변수로 민감한 정보 관리
export OPENAI_API_KEY=your_secure_api_key
export DATABASE_URL=your_secure_database_url
```

## 📈 확장 계획

### 1. **추가 도구들**
- `fashion_trend_analysis` - 트렌드 분석
- `fashion_size_recommendation` - 사이즈 추천
- `fashion_color_coordination` - 컬러 코디네이션

### 2. **다중 MCP 서버**
- 이미지 처리 전용 서버
- 리뷰 분석 전용 서버
- 추천 엔진 전용 서버

### 3. **클라우드 배포**
- Docker 컨테이너화
- Kubernetes 오케스트레이션
- Auto-scaling 설정

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 MCP 도구를 개발합니다
3. 테스트를 작성합니다
4. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 있거나 질문이 있으시면 이슈를 생성해 주세요. 