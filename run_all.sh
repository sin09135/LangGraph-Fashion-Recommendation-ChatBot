#!/bin/bash

# 패션 추천 시스템 전체 실행 스크립트
# MCP 서버, 백엔드, 프론트엔드를 순차적으로 실행

echo "🚀 패션 추천 시스템을 시작합니다..."

# 현재 디렉토리를 스크립트 위치로 변경
cd "$(dirname "$0")"

# 상위 디렉토리의 .env 파일 로드
if [ -f "../.env" ]; then
    echo "📋 .env 파일을 로드합니다..."
    source ../.env
    # DB_URL을 DATABASE_URL로 매핑
    if [ -n "$DB_URL" ]; then
        export DATABASE_URL="$DB_URL"
    fi
fi

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

# 프로세스 ID 저장 파일
PID_FILE="running_processes.pid"

# 프로세스 종료 함수
cleanup() {
    echo "🛑 모든 프로세스를 종료합니다..."
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "프로세스 $pid 종료 중..."
                kill "$pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    exit 0
}

# 시그널 핸들러 설정
trap cleanup SIGINT SIGTERM

# 기존 PID 파일 정리
rm -f "$PID_FILE"

echo "🔧 MCP 서버를 시작합니다..."
/Users/kimsinwoo/anaconda3/envs/llm_rec_310/bin/python mcp_server.py &
MCP_PID=$!
echo $MCP_PID >> "$PID_FILE"
echo "MCP 서버 PID: $MCP_PID"

# MCP 서버 시작 대기
sleep 3

echo "🔧 백엔드 서버를 시작합니다..."
/Users/kimsinwoo/anaconda3/envs/llm_rec_310/bin/python backend/main.py &
BACKEND_PID=$!
echo $BACKEND_PID >> "$PID_FILE"
echo "백엔드 서버 PID: $BACKEND_PID"

# 백엔드 서버 시작 대기
sleep 5

echo "🔧 프론트엔드 서버를 시작합니다..."
cd frontend && npm start &
FRONTEND_PID=$!
echo $FRONTEND_PID >> "$PID_FILE"
echo "프론트엔드 서버 PID: $FRONTEND_PID"

echo ""
echo "✅ 모든 서비스가 시작되었습니다!"
echo ""
echo "🌐 접속 URL:"
echo "   - 프론트엔드: http://localhost:3000"
echo "   - 백엔드 API: http://localhost:8001"
echo "   - MCP 서버: stdio (백그라운드)"
echo ""
echo "📊 서비스 상태:"
echo "   - MCP 서버: http://localhost:8001/api/mcp/health"
echo "   - 백엔드: http://localhost:8001/"
echo ""
echo "🛑 종료하려면 Ctrl+C를 누르세요."

# 모든 프로세스가 실행 중인지 모니터링
while true; do
    if [ -f "$PID_FILE" ]; then
        all_running=true
        while read -r pid; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "⚠️ 프로세스 $pid가 종료되었습니다."
                all_running=false
            fi
        done < "$PID_FILE"
        
        if [ "$all_running" = false ]; then
            echo "❌ 일부 서비스가 종료되었습니다."
            cleanup
        fi
    fi
    sleep 10
done 