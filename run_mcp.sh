#!/bin/bash

# MCP 패션 추천 서버 실행 스크립트

echo "🚀 MCP 패션 추천 서버를 시작합니다..."

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

# MCP 서버 실행
echo "🔧 MCP 서버를 시작합니다..."
python mcp_server.py

echo "✅ MCP 서버가 종료되었습니다." 