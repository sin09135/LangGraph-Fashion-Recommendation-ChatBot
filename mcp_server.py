#!/usr/bin/env python3
"""
MCP Fashion Recommendation Server
패션 추천 시스템을 MCP(Model Context Protocol) 방식으로 구현
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
from pathlib import Path

import mcp.server as mcp_server
import mcp.server.stdio
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
)

# 기존 백엔드 모듈들 import
from backend.nodes import (
    intent_router,
    recommendation_generator,
    coordination_finder,
    similar_product_finder,
    review_search_node,
    image_similarity_search,
    conversation_agent,
    output_node
)
from backend.config import get_settings
from backend.core.database import get_database_session

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FashionRecommendationState:
    """패션 추천 시스템의 상태 관리"""
    session_id: str
    previous_recommendations: List[Dict] = None
    user_preferences: Dict = None
    
    def __post_init__(self):
        if self.previous_recommendations is None:
            self.previous_recommendations = []
        if self.user_preferences is None:
            self.user_preferences = {}

class FashionMCPServer:
    """패션 추천 MCP 서버"""
    
    def __init__(self):
        self.settings = get_settings()
        self.states: Dict[str, FashionRecommendationState] = {}
        
    async def initialize(self, client: mcp_server.ClientSession) -> None:
        """MCP 서버 초기화"""
        logger.info("패션 추천 MCP 서버 초기화 중...")
        
        # 리소스 등록
        await client.list_resources()
        
        # 도구 등록
        await client.list_tools()
        
        logger.info("패션 추천 MCP 서버 초기화 완료")
    
    async def get_resources(self) -> List[Resource]:
        """사용 가능한 리소스 목록 반환"""
        return [
            Resource(
                uri="fashion://products",
                name="패션 상품 데이터베이스",
                description="패션 상품 정보 및 리뷰 데이터베이스",
                mimeType="application/json"
            ),
            Resource(
                uri="fashion://recommendations",
                name="추천 시스템",
                description="AI 기반 패션 상품 추천 시스템",
                mimeType="application/json"
            ),
            Resource(
                uri="fashion://images",
                name="이미지 데이터베이스",
                description="상품 이미지 및 CLIP 임베딩 데이터",
                mimeType="application/json"
            )
        ]
    
    async def get_tools(self) -> List[Tool]:
        """사용 가능한 도구 목록 반환"""
        return [
            Tool(
                name="fashion_recommend",
                description="패션 상품 추천 - 카테고리, 가격, 브랜드 등 조건으로 상품 추천",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "상품 카테고리 (상의, 하의, 신발, 액세서리 등)"},
                        "price_range": {"type": "string", "description": "가격 범위 (예: 1만원-5만원)"},
                        "brand": {"type": "string", "description": "브랜드명"},
                        "style": {"type": "string", "description": "스타일 (캐주얼, 정장, 스포츠 등)"},
                        "limit": {"type": "integer", "description": "추천 상품 개수 (기본값: 10)"}
                    },
                    "required": []
                }
            ),
            Tool(
                name="fashion_coordination",
                description="코디네이션 추천 - 특정 상품과 어울리는 상품 추천",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "integer", "description": "기준 상품 ID"},
                        "categories": {"type": "array", "items": {"type": "string"}, "description": "추천할 카테고리들"},
                        "budget": {"type": "string", "description": "예산 범위"},
                        "limit": {"type": "integer", "description": "추천 상품 개수"}
                    },
                    "required": ["product_id"]
                }
            ),
            Tool(
                name="fashion_similar_search",
                description="유사 상품 검색 - 이미지나 상품과 유사한 스타일 검색",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_url": {"type": "string", "description": "참조 이미지 URL"},
                        "product_id": {"type": "integer", "description": "참조 상품 ID"},
                        "similarity_threshold": {"type": "number", "description": "유사도 임계값 (0-1)"},
                        "limit": {"type": "integer", "description": "검색 결과 개수"}
                    },
                    "required": []
                }
            ),
            Tool(
                name="fashion_review_analysis",
                description="상품 리뷰 분석 및 요약",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "integer", "description": "상품 ID"},
                        "keyword": {"type": "string", "description": "검색 키워드"},
                        "rating_filter": {"type": "number", "description": "평점 필터 (1-5)"},
                        "limit": {"type": "integer", "description": "분석할 리뷰 개수"}
                    },
                    "required": []
                }
            ),
            Tool(
                name="fashion_image_search",
                description="이미지 기반 상품 검색",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_data": {"type": "string", "description": "Base64 인코딩된 이미지 데이터"},
                        "category_filter": {"type": "string", "description": "카테고리 필터"},
                        "similarity_threshold": {"type": "number", "description": "유사도 임계값"},
                        "limit": {"type": "integer", "description": "검색 결과 개수"}
                    },
                    "required": ["image_data"]
                }
            ),
            Tool(
                name="fashion_user_preferences",
                description="사용자 선호도 관리",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["get", "set", "update"], "description": "동작 타입"},
                        "preferences": {"type": "object", "description": "선호도 데이터"},
                        "session_id": {"type": "string", "description": "세션 ID"}
                    },
                    "required": ["action"]
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any], session_id: str) -> List[TextContent]:
        """도구 호출 처리"""
        logger.info(f"도구 호출: {name}, 인수: {arguments}")
        
        # 세션 상태 관리
        if session_id not in self.states:
            self.states[session_id] = FashionRecommendationState(session_id=session_id)
        
        state = self.states[session_id]
        
        try:
            if name == "fashion_recommend":
                return await self._handle_recommendation(arguments, state)
            elif name == "fashion_coordination":
                return await self._handle_coordination(arguments, state)
            elif name == "fashion_similar_search":
                return await self._handle_similar_search(arguments, state)
            elif name == "fashion_review_analysis":
                return await self._handle_review_analysis(arguments, state)
            elif name == "fashion_image_search":
                return await self._handle_image_search(arguments, state)
            elif name == "fashion_user_preferences":
                return await self._handle_user_preferences(arguments, state)
            else:
                raise ValueError(f"알 수 없는 도구: {name}")
                
        except Exception as e:
            logger.error(f"도구 호출 오류: {e}")
            return [TextContent(
                type="text",
                text=f"오류가 발생했습니다: {str(e)}"
            )]
    
    async def _handle_recommendation(self, arguments: Dict[str, Any], state: FashionRecommendationState) -> List[TextContent]:
        """일반 상품 추천 처리"""
        # LangGraph 노드 호출
        result = await recommendation_generator(
            user_input=json.dumps(arguments),
            session_id=state.session_id,
            previous_recommendations=state.previous_recommendations
        )
        
        # 결과 업데이트
        if "recommendations" in result:
            state.previous_recommendations = result["recommendations"]
        
        return [TextContent(
            type="text",
            text=result.get("response", "추천 결과를 생성할 수 없습니다.")
        )]
    
    async def _handle_coordination(self, arguments: Dict[str, Any], state: FashionRecommendationState) -> List[TextContent]:
        """코디네이션 추천 처리"""
        result = await coordination_finder(
            user_input=json.dumps(arguments),
            session_id=state.session_id,
            previous_recommendations=state.previous_recommendations
        )
        
        if "recommendations" in result:
            state.previous_recommendations = result["recommendations"]
        
        return [TextContent(
            type="text",
            text=result.get("response", "코디네이션 추천을 생성할 수 없습니다.")
        )]
    
    async def _handle_similar_search(self, arguments: Dict[str, Any], state: FashionRecommendationState) -> List[TextContent]:
        """유사 상품 검색 처리"""
        result = await similar_product_finder(
            user_input=json.dumps(arguments),
            session_id=state.session_id,
            previous_recommendations=state.previous_recommendations
        )
        
        if "recommendations" in result:
            state.previous_recommendations = result["recommendations"]
        
        return [TextContent(
            type="text",
            text=result.get("response", "유사 상품을 찾을 수 없습니다.")
        )]
    
    async def _handle_review_analysis(self, arguments: Dict[str, Any], state: FashionRecommendationState) -> List[TextContent]:
        """리뷰 분석 처리"""
        result = await review_search_node(
            user_input=json.dumps(arguments),
            session_id=state.session_id,
            previous_recommendations=state.previous_recommendations
        )
        
        return [TextContent(
            type="text",
            text=result.get("response", "리뷰 분석을 수행할 수 없습니다.")
        )]
    
    async def _handle_image_search(self, arguments: Dict[str, Any], state: FashionRecommendationState) -> List[TextContent]:
        """이미지 검색 처리"""
        result = await image_similarity_search(
            user_input=json.dumps(arguments),
            session_id=state.session_id,
            previous_recommendations=state.previous_recommendations
        )
        
        if "recommendations" in result:
            state.previous_recommendations = result["recommendations"]
        
        return [TextContent(
            type="text",
            text=result.get("response", "이미지 검색을 수행할 수 없습니다.")
        )]
    
    async def _handle_user_preferences(self, arguments: Dict[str, Any], state: FashionRecommendationState) -> List[TextContent]:
        """사용자 선호도 관리"""
        action = arguments.get("action")
        
        if action == "get":
            return [TextContent(
                type="text",
                text=f"현재 사용자 선호도: {json.dumps(state.user_preferences, ensure_ascii=False)}"
            )]
        elif action == "set":
            state.user_preferences = arguments.get("preferences", {})
            return [TextContent(
                type="text",
                text="사용자 선호도가 설정되었습니다."
            )]
        elif action == "update":
            state.user_preferences.update(arguments.get("preferences", {}))
            return [TextContent(
                type="text",
                text="사용자 선호도가 업데이트되었습니다."
            )]
        else:
            return [TextContent(
                type="text",
                text="잘못된 동작입니다. 'get', 'set', 'update' 중 하나를 선택하세요."
            )]

async def main():
    """MCP 서버 실행"""
    # MCP 서버 생성
    server = FashionMCPServer()
    
    # stdio 서버로 실행
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await mcp.server.stdio.run_server(
            server,
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fashion-recommendation-server",
                server_version="1.0.0",
                capabilities=mcp.server.stdio.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 