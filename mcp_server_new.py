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

import mcp.server
from mcp.server import FastMCP
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
)

# 기존 백엔드 모듈들 import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from nodes import (
    intent_router,
    recommendation_generator,
    coordination_finder,
    similar_product_finder,
    review_search_node,
    image_similarity_search,
    conversation_agent,
    output_node
)
from config import get_settings
from core.database import get_database_session

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

# MCP 서버 생성
server = FastMCP(
    name="fashion-recommendation-server",
    instructions="패션 상품 추천 시스템입니다. 다양한 도구를 사용하여 패션 상품을 추천하고 분석할 수 있습니다."
)

# 상태 관리
states: Dict[str, FashionRecommendationState] = {}

@server.tool()
async def fashion_recommend(
    category: str = None,
    price_range: str = None,
    brand: str = None,
    style: str = None,
    limit: int = 10
) -> str:
    """패션 상품 추천 - 카테고리, 가격, 브랜드 등 조건으로 상품 추천"""
    arguments = {
        "category": category,
        "price_range": price_range,
        "brand": brand,
        "style": style,
        "limit": limit
    }
    
    # 세션 ID 생성 (간단한 구현)
    session_id = "default_session"
    if session_id not in states:
        states[session_id] = FashionRecommendationState(session_id=session_id)
    
    state = states[session_id]
    
    result = await recommendation_generator(
        user_input=json.dumps(arguments),
        session_id=session_id,
        previous_recommendations=state.previous_recommendations
    )
    
    if "recommendations" in result:
        state.previous_recommendations = result["recommendations"]
    
    return result.get("response", "추천을 생성할 수 없습니다.")

@server.tool()
async def fashion_coordination(
    product_id: int,
    category: str = None,
    limit: int = 5
) -> str:
    """코디네이션 추천 - 특정 상품과 어울리는 상품 추천"""
    arguments = {
        "product_id": product_id,
        "category": category,
        "limit": limit
    }
    
    session_id = "default_session"
    if session_id not in states:
        states[session_id] = FashionRecommendationState(session_id=session_id)
    
    state = states[session_id]
    
    result = await coordination_finder(
        user_input=json.dumps(arguments),
        session_id=session_id,
        previous_recommendations=state.previous_recommendations
    )
    
    if "recommendations" in result:
        state.previous_recommendations = result["recommendations"]
    
    return result.get("response", "코디네이션 추천을 생성할 수 없습니다.")

@server.tool()
async def fashion_similar_search(
    product_id: int,
    limit: int = 10
) -> str:
    """유사 상품 검색 - 특정 상품과 유사한 상품 검색"""
    arguments = {
        "product_id": product_id,
        "limit": limit
    }
    
    session_id = "default_session"
    if session_id not in states:
        states[session_id] = FashionRecommendationState(session_id=session_id)
    
    state = states[session_id]
    
    result = await similar_product_finder(
        user_input=json.dumps(arguments),
        session_id=session_id,
        previous_recommendations=state.previous_recommendations
    )
    
    if "recommendations" in result:
        state.previous_recommendations = result["recommendations"]
    
    return result.get("response", "유사 상품을 찾을 수 없습니다.")

@server.tool()
async def fashion_review_analysis(
    product_id: int,
    keyword: str = None
) -> str:
    """리뷰 분석 - 상품 리뷰 분석 및 요약"""
    arguments = {
        "product_id": product_id,
        "keyword": keyword
    }
    
    session_id = "default_session"
    if session_id not in states:
        states[session_id] = FashionRecommendationState(session_id=session_id)
    
    state = states[session_id]
    
    result = await review_search_node(
        user_input=json.dumps(arguments),
        session_id=session_id,
        previous_recommendations=state.previous_recommendations
    )
    
    return result.get("response", "리뷰 분석을 수행할 수 없습니다.")

@server.tool()
async def fashion_image_search(
    image_url: str,
    category: str = None,
    limit: int = 10
) -> str:
    """이미지 검색 - 이미지와 유사한 상품 검색"""
    arguments = {
        "image_url": image_url,
        "category": category,
        "limit": limit
    }
    
    session_id = "default_session"
    if session_id not in states:
        states[session_id] = FashionRecommendationState(session_id=session_id)
    
    state = states[session_id]
    
    result = await image_similarity_search(
        user_input=json.dumps(arguments),
        session_id=session_id,
        previous_recommendations=state.previous_recommendations
    )
    
    if "recommendations" in result:
        state.previous_recommendations = result["recommendations"]
    
    return result.get("response", "이미지 검색을 수행할 수 없습니다.")

@server.tool()
async def fashion_user_preferences(
    action: str,
    preferences: Dict = None
) -> str:
    """사용자 선호도 관리 - 사용자 선호도 설정 및 조회"""
    session_id = "default_session"
    if session_id not in states:
        states[session_id] = FashionRecommendationState(session_id=session_id)
    
    state = states[session_id]
    
    if action == "get":
        return f"현재 사용자 선호도: {json.dumps(state.user_preferences, ensure_ascii=False)}"
    elif action == "set":
        state.user_preferences = preferences or {}
        return "사용자 선호도가 설정되었습니다."
    elif action == "update":
        state.user_preferences.update(preferences or {})
        return "사용자 선호도가 업데이트되었습니다."
    else:
        return "잘못된 동작입니다. 'get', 'set', 'update' 중 하나를 선택하세요."

if __name__ == "__main__":
    # stdio로 서버 실행
    server.run(transport="stdio") 