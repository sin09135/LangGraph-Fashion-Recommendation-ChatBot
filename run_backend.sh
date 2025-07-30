#!/bin/bash

echo "🚀 AI 패션 추천 시스템 백엔드를 시작합니다..."

# Python 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo "📦 가상환경을 활성화합니다..."
    source venv/bin/activate
fi

# 필요한 패키지 설치
echo "📦 필요한 패키지를 설치합니다..."
pip install -r requirements.txt

# FastAPI 서버 시작
echo "🌐 FastAPI 서버를 시작합니다..."
cd backend
python main.py 