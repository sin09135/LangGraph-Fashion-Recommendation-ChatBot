#%%
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import base64
import io
import json
from datetime import datetime
from sqlalchemy import text
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.database import engine

from config import HOST, PORT, CORS_ORIGINS
from nodes import intent_router, conversation_agent, output_node, text_filter_parser, recommendation_generator, feedback_analyzer, image_processor, image_similarity_search, similar_product_finder

# ==================== 세션 테이블 생성 ====================

def create_sessions_table():
    """세션 테이블 생성"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    state_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("✅ 세션 테이블 생성 완료")
    except Exception as e:
        print(f"❌ 세션 테이블 생성 오류: {e}")

# ==================== 세션 관리 함수 ====================

def save_session_to_db(session_id: str, state: dict):
    """세션을 데이터베이스에 저장"""
    try:
        # 메시지 객체를 직렬화 가능한 형태로 변환
        serializable_state = {}
        for key, value in state.items():
            if key == "messages":
                # 메시지 객체를 딕셔너리로 변환
                serializable_messages = []
                for msg in value:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        serializable_messages.append({
                            'content': msg.content,
                            'type': msg.type
                        })
                serializable_state[key] = serializable_messages
            else:
                serializable_state[key] = value
        
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sessions (session_id, state_data, updated_at)
                VALUES (:session_id, :state_data, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id) 
                DO UPDATE SET 
                    state_data = :state_data,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                'session_id': session_id,
                'state_data': json.dumps(serializable_state, ensure_ascii=False)
            })
            conn.commit()
    except Exception as e:
        print(f"❌ 세션 저장 오류: {e}")

def load_session_from_db(session_id: str) -> Optional[dict]:
    """데이터베이스에서 세션 로드"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT state_data FROM sessions 
                WHERE session_id = :session_id
            """), {'session_id': session_id})
            
            row = result.fetchone()
            if row and row.state_data:
                # state_data가 이미 dict인지 확인
                if isinstance(row.state_data, dict):
                    state_data = row.state_data
                else:
                    # 문자열인 경우 JSON 파싱
                    state_data = json.loads(row.state_data)
                
                # 메시지 객체 복원
                if 'messages' in state_data:
                    restored_messages = []
                    for msg_data in state_data['messages']:
                        if msg_data['type'] == 'human':
                            restored_messages.append(HumanMessage(content=msg_data['content']))
                        # AI 메시지는 나중에 다시 생성되므로 HumanMessage만 복원
                    state_data['messages'] = restored_messages
                
                return state_data
    except Exception as e:
        print(f"❌ 세션 로드 오류: {e}")
    
    return None

def delete_session_from_db(session_id: str):
    """데이터베이스에서 세션 삭제"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM sessions WHERE session_id = :session_id
            """), {'session_id': session_id})
            conn.commit()
    except Exception as e:
        print(f"❌ 세션 삭제 오류: {e}")

# ==================== FastAPI 앱 생성 ====================

app = FastAPI(title="AI 패션 추천 시스템", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 요청/응답 모델 ====================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    input_image: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    recommendations: List[Dict] = []
    session_id: str

class ImageUploadResponse(BaseModel):
    message: str
    image_url: str
    session_id: str

# ==================== 간단한 워크플로우 ====================

def create_simple_graph():
    """간단한 워크플로우 생성"""
    workflow = StateGraph(dict)  # 간단한 dict 타입 사용
    
    # 노드 추가
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("text_filter_parser", text_filter_parser)
    workflow.add_node("image_processor", image_processor)
    workflow.add_node("image_similarity_search", image_similarity_search)
    workflow.add_node("feedback_analyzer", feedback_analyzer)
    workflow.add_node("recommendation_generator", recommendation_generator)
    workflow.add_node("conversation_agent", conversation_agent)
    workflow.add_node("similar_product_finder", similar_product_finder)
    workflow.add_node("output_node", output_node)
    
    # 시작점 설정
    workflow.set_entry_point("intent_router")
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "intent_router",
        lambda state: state.get("intent", "chat"),
        {
            "recommendation": "text_filter_parser",
            "image_search": "image_processor",
            "similar_product_finder": "similar_product_finder",
            "feedback": "feedback_analyzer",
            "chat": "conversation_agent"
        }
    )
    
    # 추천 플로우
    workflow.add_edge("text_filter_parser", "recommendation_generator")
    workflow.add_edge("recommendation_generator", "output_node")
    workflow.add_edge("output_node", END)
    
    # 이미지 검색 플로우
    workflow.add_edge("image_processor", "image_similarity_search")
    workflow.add_edge("image_similarity_search", "recommendation_generator")
    
    # 특정 상품 유사 상품 찾기 플로우
    workflow.add_edge("similar_product_finder", END)
    
    # 피드백 플로우
    workflow.add_edge("feedback_analyzer", "recommendation_generator")
    
    # 대화 플로우
    workflow.add_edge("conversation_agent", END)
    
    return workflow.compile()

# ==================== 세션 관리 ====================

def create_initial_state():
    """초기 상태 생성"""
    return {
        "messages": [], "intent": None, "slots": {}, "youtuber_context": {},
        "input_image": None, "image_results": [], "recommendations": [], "feedback": None
    }

# LangGraph 앱 생성
graph_app = create_simple_graph()

# ==================== API 엔드포인트 ====================

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), session_id: Optional[str] = None):
    """이미지 업로드 API"""
    try:
        # 이미지 파일 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
        # 이미지 데이터 읽기
        image_data = await file.read()
        
        # Base64로 인코딩 (임시 저장)
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        # 세션 생성 또는 업데이트
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 기존 세션 로드 또는 새 세션 생성
        state = load_session_from_db(session_id) or create_initial_state()
        
        # 세션에 이미지 URL 저장
        state["input_image"] = image_url
        
        # 세션 저장
        save_session_to_db(session_id, state)
        
        return ImageUploadResponse(
            message="이미지 업로드 성공! 이제 이미지와 유사한 상품을 찾아드릴게요.",
            image_url=image_url,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"이미지 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail="이미지 업로드에 실패했습니다.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 엔드포인트"""
    try:
        # 세션 ID 생성
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 세션 상태 가져오기 또는 초기화
        state = load_session_from_db(session_id) or create_initial_state()
        
        # 사용자 메시지 추가
        user_message = HumanMessage(content=request.message)
        state["messages"].append(user_message)
        
        # 이미지 데이터가 있으면 state에 추가
        if request.input_image:
            state["input_image"] = request.input_image
            print(f"🖼️ 이미지 데이터 추가됨: {len(request.input_image)} 문자")
        
        # LangGraph 실행
        result = graph_app.invoke(state)
        
        # 세션 저장
        save_session_to_db(session_id, result)
        
        # API 응답 처리
        if result.get("intent") == "chat":
            # 대화 응답은 conversation_agent에서 생성된 메시지 사용
            ai_messages = [m for m in result["messages"] if getattr(m, "type", None) == "ai"]
            response_text = ai_messages[-1].content if ai_messages else "죄송합니다. 응답을 생성할 수 없습니다."
        else:
            # 추천/피드백/이미지 검색 응답은 output_node에서 생성된 메시지 사용
            last_message = result["messages"][-1] if result["messages"] else None
            response_text = last_message.content if last_message else "죄송합니다. 응답을 생성할 수 없습니다."

        return ChatResponse(
            response=response_text,
            recommendations=result.get("recommendations", []),
            session_id=session_id
        )
        
    except Exception as e:
        print(f"채팅 오류: {e}")
        raise HTTPException(status_code=500, detail="채팅 처리에 실패했습니다.")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "AI 패션 추천 시스템 API", "version": "1.0.0"}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """세션 정보 조회"""
    state = load_session_from_db(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"session_id": session_id, "state": state}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    delete_session_from_db(session_id)
    return {"message": "세션이 삭제되었습니다.", "session_id": session_id}

# ==================== 서버 시작 ====================

if __name__ == "__main__":
    # 세션 테이블 생성
    create_sessions_table()
    
    print("🚀 AI 패션 추천 시스템 서버 시작...")
    uvicorn.run(app, host=HOST, port=PORT) 