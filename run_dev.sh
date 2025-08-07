#!/bin/bash

# 개발용 실행 스크립트
# 개발 중에 필요한 서비스만 실행

echo "🔧 개발 모드로 패션 추천 시스템을 시작합니다..."

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

# 프론트엔드 의존성 설치 확인
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 프론트엔드 의존성을 설치합니다..."
    cd frontend && npm install && cd ..
fi

echo ""
echo "🎯 개발 모드 옵션:"
echo "1. 백엔드만 실행 (포트 8001)"
echo "2. 프론트엔드만 실행 (포트 3000)"
echo "3. MCP 서버만 실행"
echo "4. 백엔드 + 프론트엔드"
echo "5. MCP + 백엔드"
echo "6. 전체 시스템 (MCP + 백엔드 + 프론트엔드)"
echo ""

read -p "실행할 옵션을 선택하세요 (1-6): " choice

case $choice in
    1)
        echo "🔧 백엔드 서버를 시작합니다..."
        python backend/main.py
        ;;
    2)
        echo "🔧 프론트엔드 서버를 시작합니다..."
        cd frontend && npm start
        ;;
    3)
        echo "🔧 MCP 서버를 시작합니다..."
        python mcp_server.py
        ;;
    4)
        echo "🔧 백엔드와 프론트엔드를 시작합니다..."
        python backend/main.py &
        BACKEND_PID=$!
        echo "백엔드 PID: $BACKEND_PID"
        sleep 3
        cd frontend && npm start &
        FRONTEND_PID=$!
        echo "프론트엔드 PID: $FRONTEND_PID"
        echo "🛑 종료하려면 Ctrl+C를 누르세요."
        wait
        ;;
    5)
        echo "🔧 MCP 서버와 백엔드를 시작합니다..."
        python mcp_server.py &
        MCP_PID=$!
        echo "MCP 서버 PID: $MCP_PID"
        sleep 3
        python backend/main.py &
        BACKEND_PID=$!
        echo "백엔드 PID: $BACKEND_PID"
        echo "🛑 종료하려면 Ctrl+C를 누르세요."
        wait
        ;;
    6)
        echo "🔧 전체 시스템을 시작합니다..."
        python mcp_server.py &
        MCP_PID=$!
        echo "MCP 서버 PID: $MCP_PID"
        sleep 3
        python backend/main.py &
        BACKEND_PID=$!
        echo "백엔드 PID: $BACKEND_PID"
        sleep 3
        cd frontend && npm start &
        FRONTEND_PID=$!
        echo "프론트엔드 PID: $FRONTEND_PID"
        echo "🛑 종료하려면 Ctrl+C를 누르세요."
        wait
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac 