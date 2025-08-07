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
from backend.core.database import engine

from config import HOST, PORT, CORS_ORIGINS
from nodes import intent_router, conversation_agent, output_node, text_filter_parser, recommendation_generator, feedback_analyzer, image_processor, image_similarity_search, similar_product_finder, coordination_finder, review_search_node, review_based_recommendation, review_analyzer, filter_existing_recommendations

# MCP API 통합
try:
    from mcp_api import mcp_router, register_mcp_events
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP API를 불러올 수 없습니다. MCP 기능이 비활성화됩니다.")

# ==================== 세션 테이블 생성 ====================

def create_sessions_table():
    """세션 테이블 생성"""
    try:
        with engine.connect() as conn:
            # SQLite용 테이블 생성 (JSONB 대신 TEXT 사용)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("✅ 세션 테이블 생성 완료")
    except Exception as e:
        print(f"❌ 세션 테이블 생성 오류: {e}")

def create_likes_table():
    """좋아요 테이블 생성"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS likes (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    product_id INTEGER NOT NULL,
                    product_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, product_id)
                )
            """))
            conn.commit()
            print("✅ 좋아요 테이블 생성 완료")
    except Exception as e:
        print(f"❌ 좋아요 테이블 생성 오류: {e}")
        import traceback
        traceback.print_exc()

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

# ==================== 좋아요 관리 함수 ====================

def add_like(session_id: str, product_id: int, product_data: dict):
    """좋아요 추가"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO likes (session_id, product_id, product_data, created_at)
                VALUES (:session_id, :product_id, :product_data, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id, product_id) 
                DO UPDATE SET 
                    product_data = EXCLUDED.product_data,
                    created_at = CURRENT_TIMESTAMP
            """), {
                'session_id': session_id,
                'product_id': product_id,
                'product_data': json.dumps(product_data, ensure_ascii=False)
            })
            conn.commit()
            print(f"✅ 좋아요 추가: 세션 {session_id}, 상품 {product_id}")
            print(f"📝 저장된 상품 데이터: {product_data}")
    except Exception as e:
        print(f"❌ 좋아요 추가 오류: {e}")
        raise e

def remove_like(session_id: str, product_id: int):
    """좋아요 제거"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM likes WHERE session_id = :session_id AND product_id = :product_id
            """), {
                'session_id': session_id,
                'product_id': product_id
            })
            conn.commit()
            print(f"✅ 좋아요 제거: 세션 {session_id}, 상품 {product_id}")
    except Exception as e:
        print(f"❌ 좋아요 제거 오류: {e}")

def get_liked_products(session_id: str) -> List[Dict]:
    """좋아요한 상품 목록 조회"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT product_data FROM likes 
                WHERE session_id = :session_id 
                ORDER BY created_at DESC
            """), {'session_id': session_id})
            
            liked_products = []
            for row in result.fetchall():
                if row.product_data:
                    try:
                        product_data = json.loads(row.product_data)
                        liked_products.append(product_data)
                        print(f"📖 로드된 상품: {product_data.get('product_name', 'Unknown')}")
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON 파싱 오류: {e}, 데이터: {row.product_data}")
            
            print(f"✅ 좋아요 상품 조회: 세션 {session_id}, {len(liked_products)}개")
            return liked_products
    except Exception as e:
        print(f"❌ 좋아요 상품 조회 오류: {e}")
        return []

def is_liked(session_id: str, product_id: int) -> bool:
    """상품이 좋아요되었는지 확인"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 1 FROM likes 
                WHERE session_id = :session_id AND product_id = :product_id
            """), {
                'session_id': session_id,
                'product_id': product_id
            })
            return result.fetchone() is not None
    except Exception as e:
        print(f"❌ 좋아요 확인 오류: {e}")
        return False

# ==================== FastAPI 앱 생성 ====================

app = FastAPI(
    title="AI 패션 추천 시스템 (MCP 통합)",
    description="AI 기반 패션 상품 추천 챗봇 API (MCP 통합)",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP API 라우터 추가 (가능한 경우)
if MCP_AVAILABLE:
    app.include_router(mcp_router)
    register_mcp_events(app)
    print("✅ MCP API가 활성화되었습니다.")
else:
    print("❌ MCP API가 비활성화되었습니다.")

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

class LikeRequest(BaseModel):
    session_id: str
    product_id: int
    action: str  # "like" or "unlike"

class LikeResponse(BaseModel):
    message: str
    liked_products: List[Dict] = []

# ==================== 워크플로우 ====================

def create_simple_graph():
    """워크플로우 생성"""
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
    workflow.add_node("coordination_finder", coordination_finder)
    workflow.add_node("review_search_node", review_search_node)
    workflow.add_node("review_based_recommendation", review_based_recommendation)
    workflow.add_node("review_analyzer", review_analyzer)
    workflow.add_node("filter_existing_recommendations", filter_existing_recommendations)
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
            "coordination": "coordination_finder",
            "feedback": "feedback_analyzer",
            "filter_existing": "filter_existing_recommendations",
            "review_search": "review_search_node",
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
    
    # 코디 추천 플로우
    workflow.add_edge("coordination_finder", END)
    
    # 리뷰 검색 플로우
    workflow.add_edge("review_search_node", "review_analyzer")
    workflow.add_edge("review_analyzer", "review_based_recommendation")
    workflow.add_edge("review_based_recommendation", "output_node")
    
    # 기존 추천 결과 필터링 플로우
    workflow.add_edge("filter_existing_recommendations", "output_node")
    
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

@app.post("/like", response_model=LikeResponse)
async def toggle_like(request: LikeRequest):
    """좋아요 토글"""
    try:
        if request.action == "like":
            # 상품 정보를 가져와서 좋아요 추가
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT product_id, product_name, category, image_url, price, brand_kr
                    FROM products WHERE product_id = :product_id
                """), {'product_id': request.product_id})
                
                product = result.fetchone()
                if product:
                    product_data = {
                        'product_id': product.product_id,
                        'product_name': product.product_name,
                        'category': product.category,
                        'image_url': product.image_url,
                        'price': product.price,
                        'brand_kr': product.brand_kr
                    }
                    add_like(request.session_id, request.product_id, product_data)
                    message = f"상품 '{product.product_name}'을 좋아요에 추가했습니다."
                else:
                    raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다.")
        elif request.action == "unlike":
            remove_like(request.session_id, request.product_id)
            message = "좋아요를 취소했습니다."
        else:
            raise HTTPException(status_code=400, detail="잘못된 액션입니다.")
        
        # 좋아요한 상품 목록 반환
        liked_products = get_liked_products(request.session_id)
        
        return LikeResponse(
            message=message,
            liked_products=liked_products
        )
        
    except Exception as e:
        print(f"좋아요 토글 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"좋아요 처리에 실패했습니다: {str(e)}")

@app.get("/likes/{session_id}", response_model=LikeResponse)
async def get_likes(session_id: str):
    """좋아요한 상품 목록 조회"""
    try:
        liked_products = get_liked_products(session_id)
        return LikeResponse(
            message=f"좋아요한 상품 {len(liked_products)}개를 찾았습니다.",
            liked_products=liked_products
        )
    except Exception as e:
        print(f"좋아요 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="좋아요 목록 조회에 실패했습니다.")

@app.get("/is-liked/{session_id}/{product_id}")
async def check_liked(session_id: str, product_id: int):
    """상품이 좋아요되었는지 확인"""
    try:
        is_liked_status = is_liked(session_id, product_id)
        return {"is_liked": is_liked_status}
    except Exception as e:
        print(f"좋아요 확인 오류: {e}")
        raise HTTPException(status_code=500, detail="좋아요 확인에 실패했습니다.")

# ==================== 서버 시작 ====================

if __name__ == "__main__":
    # 세션 테이블 생성
    create_sessions_table()
    create_likes_table() # 좋아요 테이블 생성
    
    print("🚀 AI 패션 추천 시스템 서버 시작...")
    uvicorn.run(app, host=HOST, port=PORT) 