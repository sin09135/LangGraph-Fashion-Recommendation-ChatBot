#!/usr/bin/env python3
"""
MCP API 엔드포인트
프론트엔드와 MCP 서버 간의 통신을 처리
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mcp_client import FashionRecommendationAPI

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 라우터 생성
mcp_router = APIRouter(prefix="/api/mcp", tags=["mcp"])

# Pydantic 모델들
class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    session_id: str = "default"

class ToolCallResponse(BaseModel):
    success: bool
    data: str = ""
    error: str = ""

# MCP API 인스턴스
mcp_api: FashionRecommendationAPI = None

async def get_mcp_api() -> FashionRecommendationAPI:
    """MCP API 인스턴스 가져오기 (싱글톤 패턴)"""
    global mcp_api
    if mcp_api is None:
        mcp_api = FashionRecommendationAPI()
        await mcp_api.connect()
    return mcp_api

@mcp_router.get("/tools")
async def get_tools():
    """사용 가능한 도구 목록 조회"""
    try:
        api = await get_mcp_api()
        tools = await api.get_tools_info()
        return JSONResponse(content=tools)
    except Exception as e:
        logger.error(f"도구 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"도구 목록 조회 실패: {str(e)}")

@mcp_router.get("/resources")
async def get_resources():
    """사용 가능한 리소스 목록 조회"""
    try:
        api = await get_mcp_api()
        resources = await api.get_resources_info()
        return JSONResponse(content=resources)
    except Exception as e:
        logger.error(f"리소스 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"리소스 목록 조회 실패: {str(e)}")

@mcp_router.post("/call-tool")
async def call_tool(request: ToolCallRequest):
    """도구 호출"""
    try:
        api = await get_mcp_api()
        
        # 도구별 호출 메서드 매핑
        tool_methods = {
            "fashion_recommend": api.recommend,
            "fashion_coordination": api.coordinate,
            "fashion_similar_search": api.find_similar,
            "fashion_review_analysis": api.analyze_reviews,
            "fashion_image_search": api.search_by_image,
            "fashion_user_preferences": api.manage_preferences
        }
        
        if request.tool_name not in tool_methods:
            raise HTTPException(status_code=400, detail=f"알 수 없는 도구: {request.tool_name}")
        
        # 도구 호출
        method = tool_methods[request.tool_name]
        result = await method(**request.arguments)
        
        return ToolCallResponse(
            success=True,
            data=result
        )
        
    except Exception as e:
        logger.error(f"도구 호출 실패: {e}")
        return ToolCallResponse(
            success=False,
            error=str(e)
        )

@mcp_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 엔드포인트"""
    await websocket.accept()
    
    try:
        api = await get_mcp_api()
        
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 메시지 타입에 따른 처리
            if message.get("type") == "call_tool":
                try:
                    tool_name = message.get("tool_name")
                    arguments = message.get("arguments", {})
                    session_id = message.get("session_id", "default")
                    
                    # 도구 호출
                    tool_methods = {
                        "fashion_recommend": api.recommend,
                        "fashion_coordination": api.coordinate,
                        "fashion_similar_search": api.find_similar,
                        "fashion_review_analysis": api.analyze_reviews,
                        "fashion_image_search": api.search_by_image,
                        "fashion_user_preferences": api.manage_preferences
                    }
                    
                    if tool_name in tool_methods:
                        method = tool_methods[tool_name]
                        result = await method(**arguments)
                        
                        # 결과 전송
                        await websocket.send_text(json.dumps({
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "result": result,
                            "success": True
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "result": f"알 수 없는 도구: {tool_name}",
                            "success": False
                        }))
                        
                except Exception as e:
                    logger.error(f"WebSocket 도구 호출 실패: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "tool_result",
                        "tool_name": message.get("tool_name", "unknown"),
                        "result": f"오류가 발생했습니다: {str(e)}",
                        "success": False
                    }))
            
            elif message.get("type") == "get_tools":
                try:
                    tools = await api.get_tools_info()
                    await websocket.send_text(json.dumps({
                        "type": "tools_list",
                        "tools": tools
                    }))
                except Exception as e:
                    logger.error(f"도구 목록 조회 실패: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"도구 목록 조회 실패: {str(e)}"
                    }))
            
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong"
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket 연결이 종료되었습니다.")
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"WebSocket 오류: {str(e)}"
            }))
        except:
            pass

@mcp_router.get("/health")
async def health_check():
    """헬스 체크"""
    try:
        api = await get_mcp_api()
        return {"status": "healthy", "connected": True}
    except Exception as e:
        return {"status": "unhealthy", "connected": False, "error": str(e)}

# MCP 서버 상태 관리
class MCPServerManager:
    """MCP 서버 관리자"""
    
    def __init__(self):
        self.api: FashionRecommendationAPI = None
        self._lock = asyncio.Lock()
    
    async def get_api(self) -> FashionRecommendationAPI:
        """API 인스턴스 가져오기"""
        async with self._lock:
            if self.api is None:
                self.api = FashionRecommendationAPI()
                await self.api.connect()
            return self.api
    
    async def close(self):
        """연결 종료"""
        if self.api:
            await self.api.disconnect()
            self.api = None

# 전역 MCP 서버 관리자
mcp_manager = MCPServerManager()

# 애플리케이션 종료 시 정리
async def cleanup_mcp():
    """MCP 연결 정리"""
    await mcp_manager.close()

# FastAPI 이벤트 핸들러 등록을 위한 함수
def register_mcp_events(app):
    """FastAPI 앱에 MCP 이벤트 핸들러 등록"""
    @app.on_event("shutdown")
    async def shutdown_event():
        await cleanup_mcp() 