#!/bin/bash

# MCP 서버 테스트 스크립트

echo "🧪 MCP 서버를 테스트합니다..."

# 현재 디렉토리를 스크립트 위치로 변경
cd "$(dirname "$0")"

# 환경 변수 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
    echo "export OPENAI_API_KEY=your_api_key 를 실행하세요."
    exit 1
fi

if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL 환경 변수가 설정되지 않았습니다."
    echo "export DATABASE_URL=postgresql://username:password@localhost:5432/fashion_db 를 실행하세요."
    exit 1
fi

# Python 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo "📦 가상환경을 활성화합니다..."
    source venv/bin/activate
fi

# 의존성 설치 확인
echo "📋 의존성을 확인합니다..."
pip install -r requirements.txt

echo ""
echo "🧪 테스트 옵션:"
echo "1. MCP 서버 연결 테스트"
echo "2. 도구 목록 테스트"
echo "3. 상품 추천 테스트"
echo "4. 코디네이션 추천 테스트"
echo "5. 전체 테스트"
echo ""

read -p "실행할 테스트를 선택하세요 (1-5): " choice

case $choice in
    1)
        echo "🔧 MCP 서버 연결을 테스트합니다..."
        python -c "
import asyncio
from mcp_client import FashionRecommendationAPI

async def test_connection():
    try:
        async with FashionRecommendationAPI() as api:
            print('✅ MCP 서버 연결 성공!')
            tools = await api.get_tools_info()
            print(f'📋 사용 가능한 도구: {len(tools)}개')
            for tool in tools:
                print(f'  - {tool[\"name\"]}: {tool[\"description\"]}')
    except Exception as e:
        print(f'❌ MCP 서버 연결 실패: {e}')

asyncio.run(test_connection())
"
        ;;
    2)
        echo "🔧 도구 목록을 테스트합니다..."
        python -c "
import asyncio
from mcp_client import FashionRecommendationAPI

async def test_tools():
    try:
        async with FashionRecommendationAPI() as api:
            tools = await api.get_tools_info()
            print('📋 사용 가능한 도구 목록:')
            for i, tool in enumerate(tools, 1):
                print(f'{i}. {tool[\"name\"]}')
                print(f'   설명: {tool[\"description\"]}')
                print(f'   스키마: {tool[\"input_schema\"]}')
                print()
    except Exception as e:
        print(f'❌ 도구 목록 테스트 실패: {e}')

asyncio.run(test_tools())
"
        ;;
    3)
        echo "🔧 상품 추천을 테스트합니다..."
        python -c "
import asyncio
from mcp_client import FashionRecommendationAPI

async def test_recommendation():
    try:
        async with FashionRecommendationAPI() as api:
            print('🔍 상품 추천 테스트...')
            result = await api.recommend(
                category='상의',
                price_range='1만원-5만원',
                limit=3
            )
            print('✅ 추천 결과:')
            print(result)
    except Exception as e:
        print(f'❌ 상품 추천 테스트 실패: {e}')

asyncio.run(test_recommendation())
"
        ;;
    4)
        echo "🔧 코디네이션 추천을 테스트합니다..."
        python -c "
import asyncio
from mcp_client import FashionRecommendationAPI

async def test_coordination():
    try:
        async with FashionRecommendationAPI() as api:
            print('🔍 코디네이션 추천 테스트...')
            result = await api.coordinate(
                product_id=1,
                categories=['하의', '신발'],
                limit=3
            )
            print('✅ 코디네이션 결과:')
            print(result)
    except Exception as e:
        print(f'❌ 코디네이션 추천 테스트 실패: {e}')

asyncio.run(test_coordination())
"
        ;;
    5)
        echo "🔧 전체 테스트를 실행합니다..."
        python -c "
import asyncio
from mcp_client import FashionRecommendationAPI

async def run_all_tests():
    try:
        async with FashionRecommendationAPI() as api:
            print('🧪 전체 테스트 시작...')
            
            # 1. 연결 테스트
            print('\\n1. 연결 테스트...')
            tools = await api.get_tools_info()
            print(f'✅ 연결 성공! 도구 {len(tools)}개 발견')
            
            # 2. 상품 추천 테스트
            print('\\n2. 상품 추천 테스트...')
            rec_result = await api.recommend(category='상의', limit=2)
            print('✅ 상품 추천 성공')
            
            # 3. 코디네이션 테스트
            print('\\n3. 코디네이션 추천 테스트...')
            coord_result = await api.coordinate(product_id=1, limit=2)
            print('✅ 코디네이션 추천 성공')
            
            # 4. 리뷰 분석 테스트
            print('\\n4. 리뷰 분석 테스트...')
            review_result = await api.analyze_reviews(product_id=1, limit=5)
            print('✅ 리뷰 분석 성공')
            
            # 5. 사용자 선호도 테스트
            print('\\n5. 사용자 선호도 테스트...')
            pref_result = await api.manage_preferences(
                action='set',
                preferences={'preferred_brands': ['유니클로']}
            )
            print('✅ 사용자 선호도 설정 성공')
            
            print('\\n🎉 모든 테스트 통과!')
            
    except Exception as e:
        print(f'❌ 테스트 실패: {e}')

asyncio.run(run_all_tests())
"
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "✅ 테스트가 완료되었습니다." 