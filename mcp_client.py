#!/usr/bin/env python3
"""
MCP Fashion Recommendation Client
패션 추천 MCP 서버와 통신하는 클라이언트
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import mcp.client as mcp_client
import mcp.client.stdio
from mcp.types import TextContent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FashionRecommendationRequest:
    """패션 추천 요청 데이터"""
    tool_name: str
    arguments: Dict[str, Any]
    session_id: str

class FashionMCPClient:
    """패션 추천 MCP 클라이언트"""
    
    def __init__(self, server_path: str = "./mcp_server.py"):
        self.server_path = server_path
        self.client: Optional[mcp_client.ClientSession] = None
        
    async def connect(self):
        """MCP 서버에 연결"""
        try:
            # stdio 클라이언트 생성
            self.client = await mcp_client.stdio.stdio_client(
                server_path=self.server_path
            )
            logger.info("MCP 서버에 연결되었습니다.")
            
            # 서버 초기화
            await self.client.initialize()
            
        except Exception as e:
            logger.error(f"MCP 서버 연결 실패: {e}")
            raise
    
    async def disconnect(self):
        """MCP 서버 연결 해제"""
        if self.client:
            await self.client.close()
            logger.info("MCP 서버 연결이 해제되었습니다.")
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록 조회"""
        if not self.client:
            raise RuntimeError("MCP 서버에 연결되지 않았습니다.")
        
        tools = await self.client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in tools
        ]
    
    async def get_available_resources(self) -> List[Dict[str, Any]]:
        """사용 가능한 리소스 목록 조회"""
        if not self.client:
            raise RuntimeError("MCP 서버에 연결되지 않았습니다.")
        
        resources = await self.client.list_resources()
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mimeType
            }
            for resource in resources
        ]
    
    async def call_tool(self, request: FashionRecommendationRequest) -> str:
        """도구 호출"""
        if not self.client:
            raise RuntimeError("MCP 서버에 연결되지 않았습니다.")
        
        try:
            # 도구 호출
            result = await self.client.call_tool(
                name=request.tool_name,
                arguments=request.arguments
            )
            
            # 결과 텍스트 추출
            if result and len(result) > 0:
                return result[0].text
            else:
                return "결과가 없습니다."
                
        except Exception as e:
            logger.error(f"도구 호출 실패: {e}")
            return f"오류가 발생했습니다: {str(e)}"
    
    async def recommend_products(self, 
                               category: Optional[str] = None,
                               price_range: Optional[str] = None,
                               brand: Optional[str] = None,
                               style: Optional[str] = None,
                               limit: int = 10,
                               session_id: str = "default") -> str:
        """상품 추천"""
        arguments = {}
        if category:
            arguments["category"] = category
        if price_range:
            arguments["price_range"] = price_range
        if brand:
            arguments["brand"] = brand
        if style:
            arguments["style"] = style
        if limit:
            arguments["limit"] = limit
        
        request = FashionRecommendationRequest(
            tool_name="fashion_recommend",
            arguments=arguments,
            session_id=session_id
        )
        
        return await self.call_tool(request)
    
    async def recommend_coordination(self,
                                  product_id: int,
                                  categories: Optional[List[str]] = None,
                                  budget: Optional[str] = None,
                                  limit: int = 10,
                                  session_id: str = "default") -> str:
        """코디네이션 추천"""
        arguments = {"product_id": product_id}
        if categories:
            arguments["categories"] = categories
        if budget:
            arguments["budget"] = budget
        if limit:
            arguments["limit"] = limit
        
        request = FashionRecommendationRequest(
            tool_name="fashion_coordination",
            arguments=arguments,
            session_id=session_id
        )
        
        return await self.call_tool(request)
    
    async def search_similar_products(self,
                                    product_id: Optional[int] = None,
                                    image_url: Optional[str] = None,
                                    similarity_threshold: float = 0.7,
                                    limit: int = 10,
                                    session_id: str = "default") -> str:
        """유사 상품 검색"""
        arguments = {}
        if product_id:
            arguments["product_id"] = product_id
        if image_url:
            arguments["image_url"] = image_url
        if similarity_threshold:
            arguments["similarity_threshold"] = similarity_threshold
        if limit:
            arguments["limit"] = limit
        
        request = FashionRecommendationRequest(
            tool_name="fashion_similar_search",
            arguments=arguments,
            session_id=session_id
        )
        
        return await self.call_tool(request)
    
    async def analyze_reviews(self,
                            product_id: Optional[int] = None,
                            keyword: Optional[str] = None,
                            rating_filter: Optional[float] = None,
                            limit: int = 20,
                            session_id: str = "default") -> str:
        """리뷰 분석"""
        arguments = {}
        if product_id:
            arguments["product_id"] = product_id
        if keyword:
            arguments["keyword"] = keyword
        if rating_filter:
            arguments["rating_filter"] = rating_filter
        if limit:
            arguments["limit"] = limit
        
        request = FashionRecommendationRequest(
            tool_name="fashion_review_analysis",
            arguments=arguments,
            session_id=session_id
        )
        
        return await self.call_tool(request)
    
    async def search_by_image(self,
                             image_data: str,
                             category_filter: Optional[str] = None,
                             similarity_threshold: float = 0.7,
                             limit: int = 10,
                             session_id: str = "default") -> str:
        """이미지 기반 검색"""
        arguments = {"image_data": image_data}
        if category_filter:
            arguments["category_filter"] = category_filter
        if similarity_threshold:
            arguments["similarity_threshold"] = similarity_threshold
        if limit:
            arguments["limit"] = limit
        
        request = FashionRecommendationRequest(
            tool_name="fashion_image_search",
            arguments=arguments,
            session_id=session_id
        )
        
        return await self.call_tool(request)
    
    async def manage_user_preferences(self,
                                    action: str,
                                    preferences: Optional[Dict[str, Any]] = None,
                                    session_id: str = "default") -> str:
        """사용자 선호도 관리"""
        arguments = {"action": action}
        if preferences:
            arguments["preferences"] = preferences
        if session_id:
            arguments["session_id"] = session_id
        
        request = FashionRecommendationRequest(
            tool_name="fashion_user_preferences",
            arguments=arguments,
            session_id=session_id
        )
        
        return await self.call_tool(request)

class FashionRecommendationAPI:
    """패션 추천 API 래퍼"""
    
    def __init__(self, server_path: str = "./mcp_server.py"):
        self.client = FashionMCPClient(server_path)
        self._connected = False
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.disconnect()
    
    async def connect(self):
        """MCP 서버에 연결"""
        await self.client.connect()
        self._connected = True
    
    async def disconnect(self):
        """MCP 서버 연결 해제"""
        await self.client.disconnect()
        self._connected = False
    
    async def get_tools_info(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 정보 조회"""
        return await self.client.get_available_tools()
    
    async def get_resources_info(self) -> List[Dict[str, Any]]:
        """사용 가능한 리소스 정보 조회"""
        return await self.client.get_available_resources()
    
    # 편의 메서드들
    async def recommend(self, **kwargs) -> str:
        """상품 추천 (편의 메서드)"""
        return await self.client.recommend_products(**kwargs)
    
    async def coordinate(self, **kwargs) -> str:
        """코디네이션 추천 (편의 메서드)"""
        return await self.client.recommend_coordination(**kwargs)
    
    async def find_similar(self, **kwargs) -> str:
        """유사 상품 검색 (편의 메서드)"""
        return await self.client.search_similar_products(**kwargs)
    
    async def analyze_reviews(self, **kwargs) -> str:
        """리뷰 분석 (편의 메서드)"""
        return await self.client.analyze_reviews(**kwargs)
    
    async def search_by_image(self, **kwargs) -> str:
        """이미지 검색 (편의 메서드)"""
        return await self.client.search_by_image(**kwargs)
    
    async def manage_preferences(self, **kwargs) -> str:
        """선호도 관리 (편의 메서드)"""
        return await self.client.manage_user_preferences(**kwargs)

# 사용 예시
async def example_usage():
    """사용 예시"""
    async with FashionRecommendationAPI() as api:
        # 도구 정보 조회
        tools = await api.get_tools_info()
        print("사용 가능한 도구들:")
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")
        
        # 상품 추천
        result = await api.recommend(
            category="상의",
            price_range="1만원-5만원",
            limit=5
        )
        print(f"\n추천 결과:\n{result}")
        
        # 코디네이션 추천
        coord_result = await api.coordinate(
            product_id=1,
            categories=["하의", "신발"],
            limit=3
        )
        print(f"\n코디네이션 추천:\n{coord_result}")

if __name__ == "__main__":
    asyncio.run(example_usage()) 