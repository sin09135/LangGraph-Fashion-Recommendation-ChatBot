from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage
from llm_service import llm_service
import json
from sqlalchemy import create_engine, text
from config import (
    DB_URL, IMAGE_BASE_URL, MAX_IMAGE_RESULTS, MAX_RECOMMENDATIONS, 
    DEFAULT_PRICE_RANGES, DEFAULT_DISPLAY_COUNT, DEFAULT_SIMILARITY_SCORE,
    AVAILABLE_STYLES, FEEDBACK_KEYWORDS, SIMILAR_PRODUCT_KEYWORDS
)
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import io
import base64
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# 상태 타입 정의
AgentState = Dict[str, Any] # Simplified for flat structure

# 데이터베이스 연결
engine = create_engine(DB_URL)

# 이미지 임베딩 캐시 (메모리 기반)
image_embedding_cache = {}

# CLIP 모델 초기화
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    print("✅ CLIP 모델 로드 완료")
except Exception as e:
    print(f"⚠️ CLIP 모델 로드 실패: {e}")
    clip_model = None
    clip_processor = None

def get_image_embedding(image_path: str, image_url: str = None) -> Optional[np.ndarray]:
    """이미지 임베딩을 가져오거나 생성 (캐싱 포함 + 오류 처리 강화)"""
    cache_key = image_path or image_url
    
    if not cache_key:
        return None
    
    if cache_key in image_embedding_cache:
        print(f"📦 캐시에서 이미지 임베딩 로드: {os.path.basename(cache_key)}")
        return image_embedding_cache[cache_key]
    
    try:
        image = None

        # ✅ 로컬 이미지 처리
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
        
        # ✅ URL 이미지 처리
        elif image_url:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                print(f"⚠️ 이미지가 아닌 응답: {content_type}")
                return None

            try:
                image = Image.open(io.BytesIO(response.content))
            except Exception as e:
                print(f"⚠️ URL 이미지 로드 실패: {e}")
                return None

        if image is None:
            return None

        # RGB 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # CLIP 임베딩 생성
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)

        embedding = features.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)

        # 캐시에 저장
        image_embedding_cache[cache_key] = embedding
        print(f"💾 이미지 임베딩 캐시 저장: {os.path.basename(cache_key)}")

        return embedding

    except Exception as e:
        print(f"⚠️ 이미지 임베딩 생성 실패: {e}")
        return None

def calculate_weighted_similarity(product_embedding, similar_embedding, price_diff=None, brand_match=False):
    """가중치가 적용된 유사도 계산"""
    # 기본 이미지 유사도 (0.7 가중치)
    image_similarity = np.dot(product_embedding, similar_embedding)
    
    # 가격 유사도 (0.2 가중치)
    price_similarity = 0.0
    if price_diff is not None:
        # 가격 차이가 적을수록 높은 유사도
        price_similarity = max(0, 1 - abs(price_diff) / 50000)  # 5만원 차이를 기준
    
    # 브랜드 일치 보너스 (0.1 가중치)
    brand_bonus = 0.1 if brand_match else 0.0
    
    # 가중 평균 계산
    weighted_similarity = (image_similarity * 0.7 + price_similarity * 0.2 + brand_bonus)
    
    return weighted_similarity

def get_similar_products_by_id(product_id: int, limit: int = 10) -> list:
    """특정 상품 ID로 유사한 상품들을 찾는 함수"""
    try:
        with engine.connect() as conn:
            # 해당 상품의 정보 가져오기
            product_query = text("""
                SELECT product_id, product_name, category, image_path, image_url, price, brand_kr
                FROM products 
                WHERE product_id = :product_id
            """)
            product_result = conn.execute(product_query, {"product_id": product_id}).fetchone()
            
            if not product_result:
                print(f"⚠️ 상품 ID {product_id}를 찾을 수 없습니다.")
                return []
            
            # Row 객체를 딕셔너리로 변환
            product = {
                'product_id': product_result.product_id,
                'product_name': product_result.product_name,
                'category': product_result.category,
                'image_path': product_result.image_path,
                'image_url': product_result.image_url,
                'price': product_result.price,
                'brand_kr': product_result.brand_kr
            }
            print(f"🎯 대상 상품: {product['product_name']} (카테고리: {product['category']})")
            
            # 해당 상품의 이미지 임베딩 가져오기
            product_embedding = get_image_embedding(product['image_path'], product['image_url'])
            
            if product_embedding is None:
                print(f"⚠️ 상품 ID {product_id}의 이미지 임베딩을 생성할 수 없습니다.")
                return []
            
            # 같은 카테고리의 다른 상품들 가져오기
            similar_query = text("""
                SELECT product_id, product_name, category, image_path, image_url, price, brand_kr
                FROM products 
                WHERE category = :category 
                AND product_id != :product_id
                AND image_path IS NOT NULL AND image_path != ''
                LIMIT :limit
            """)
            
            similar_products = conn.execute(similar_query, {
                "category": product['category'],
                "product_id": product_id,
                "limit": limit * 2  # 더 많은 상품을 가져와서 필터링
            }).fetchall()
            
            if not similar_products:
                print(f"⚠️ 카테고리 '{product['category']}'에서 다른 상품을 찾을 수 없습니다.")
                return []
            
            # 유사도 계산
            products_with_similarity = []
            for similar_product in similar_products:
                # Row 객체를 딕셔너리로 변환
                similar_product_dict = {
                    'product_id': similar_product.product_id,
                    'product_name': similar_product.product_name,
                    'category': similar_product.category,
                    'image_path': similar_product.image_path,
                    'image_url': similar_product.image_url,
                    'price': similar_product.price,
                    'brand_kr': similar_product.brand_kr
                }
                
                similar_embedding = get_image_embedding(
                    similar_product_dict['image_path'], 
                    similar_product_dict['image_url']
                )
                
                if similar_embedding is not None:
                    # 가격 차이 계산
                    price_diff = None
                    if product.get('price') and similar_product_dict.get('price'):
                        price_diff = abs(float(product['price']) - float(similar_product_dict['price']))
                    
                    # 브랜드 일치 여부
                    brand_match = (product.get('brand_kr') and similar_product_dict.get('brand_kr') and 
                                 product['brand_kr'] == similar_product_dict['brand_kr'])
                    
                    # 가중 유사도 계산
                    weighted_similarity = calculate_weighted_similarity(
                        product_embedding, 
                        similar_embedding, 
                        price_diff, 
                        brand_match
                    )
                    
                    similar_product_dict['similarity_score'] = float(weighted_similarity)
                    similar_product_dict['image_similarity'] = float(np.dot(product_embedding, similar_embedding))
                    similar_product_dict['price_diff'] = price_diff
                    similar_product_dict['brand_match'] = brand_match
                    products_with_similarity.append(similar_product_dict)
            
            # 유사도 순으로 정렬
            products_with_similarity.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # 상위 결과만 반환
            result = products_with_similarity[:limit]
            
            print(f"✅ {product['category']} 카테고리에서 {len(result)}개 유사 상품 찾음")
            for i, product in enumerate(result[:3], 1):
                print(f"   {i}. {product['product_name']} (유사도: {product['similarity_score']:.3f})")
            
            return result
            
    except Exception as e:
        print(f"⚠️ 유사 상품 검색 오류: {e}")
        return []

def extract_product_number(user_input: str) -> int:
    """사용자 입력에서 상품 번호를 추출하는 함수"""
    import re
    
    # 다양한 패턴 매칭
    patterns = [
        r'(\d+)번',  # "1번", "2번" 등
        r'(\d+)번째',  # "1번째", "2번째" 등
        r'(\d+)번 상품',  # "1번 상품" 등
        r'(\d+)번째 상품',  # "1번째 상품" 등
        r'상품 (\d+)',  # "상품 1" 등
        r'(\d+)',  # 단순 숫자 (마지막에 체크)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input)
        if match:
            number = int(match.group(1))
            # 1-10 범위 체크 (추천 결과는 보통 10개 이하)
            if 1 <= number <= 10:
                print(f"🔍 상품 번호 추출: {number}번")
                return number
    
    print(f"⚠️ 상품 번호를 찾을 수 없습니다: {user_input}")
    return 0

def intent_router(state: AgentState) -> AgentState:
    """사용자 의도 분류"""
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    if state.get("input_image"):
        print("🖼️ 이미지가 감지되어 image_search로 분류합니다.")
        state['intent'] = "image_search"
        return state
    
    # Route to similar_product_finder if product number is mentioned and previous recommendations exist
    previous_recommendations = state.get("previous_recommendations", [])
    print(f"🔍 intent_router - previous_recommendations 개수: {len(previous_recommendations)}")
    
    if previous_recommendations:
        product_number = extract_product_number(user_input)
        print(f"🔍 intent_router - 추출된 상품 번호: {product_number}")
        
        if product_number > 0 and product_number <= len(previous_recommendations):
            print(f"🎯 {product_number}번 상품과 유사한 상품 검색으로 분류합니다.")
            state['intent'] = "similar_product_finder"
            return state
    
    # 프롬프팅 기반 의도 분류 (키워드 체크 대신)
    intent_prompt = f"""
    사용자 입력: "{user_input}"
    
    다음 중 하나로 분류해주세요:
    - "image_search": 유사한 상품, 비슷한 상품, 같은 스타일, 이미지 기반 검색
    - "recommendation": 상품 추천 요청
    - "feedback": 이전 추천에 대한 피드백 (비싸요, 싸요, 마음에 안 들어요 등)
    - "chat": 일반 대화
    
    분류 규칙:
    1. 이미지 검색: "유사한", "비슷한", "같은 스타일", "이런 느낌" 등
    2. 추천 요청: "추천해줘", "보여줘", "찾아줘" 등
    3. 피드백: "비싸요", "싸요", "마음에 안 들어요", "다른 거", "더 저렴한", "4만원 미만" 등
    4. 일반 대화: 인사, 감사, 기타 잡담
    
    다음 형식으로 JSON을 반환하세요:
    {{
        "intent": "image_search/recommendation/feedback/chat",
        "confidence": "high/medium/low",
        "reason": "분류 이유"
    }}
    """
    
    try:
        response = llm_service.invoke(intent_prompt).strip()
        result = json.loads(response)
        intent = result.get('intent', 'chat')
        confidence = result.get('confidence', 'low')
        
        print(f"🤖 의도 분석: {intent}")
        state['intent'] = intent
        
    except Exception as e:
        print(f"⚠️ 의도 분류 오류: {e}")
        state['intent'] = 'chat'
    
    return state

def youtuber_style_analyzer(state: AgentState) -> AgentState:
    """유튜버 스타일 분석"""
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    # 유튜버 이름 감지 및 스타일 추론
    youtuber_prompt = f"""
    사용자 입력: "{user_input}"
    
    다음 유튜버들의 스타일을 분석해주세요:
    
    유튜버 스타일 데이터베이스:
    - "핏더사이즈": 스트릿, 와이드핏, 오버핏, 캐주얼, 스포티
    - "김여운": 미니멀, 베이직, 모노톤, 심플, 클린
    - "디즈니": 컬러풀, 귀여운, 캐릭터, 팝, 플레이풀
    - "김민지": 트렌디, 스트릿, 힙합, 오버핏, 스포티
    - "아이브": 걸크러시, 스트릿, 힙합, 오버핏, 스포티
    - "뉴진스": Y2K, 레트로, 컬러풀, 귀여운, 팝
    - "블랙핑크": 걸크러시, 스트릿, 힙합, 오버핏, 스포티
    - "르세라핌": 미니멀, 베이직, 모노톤, 심플, 클린
    - "아이유": 미니멀, 베이직, 모노톤, 심플, 클린
    - "태연": 미니멀, 베이직, 모노톤, 심플, 클린
    
    만약 위 유튜버가 언급되지 않았다면 빈 결과를 반환하세요.
    
    다음 형식으로 JSON을 반환하세요:
    {{
        "youtuber_detected": true/false,
        "youtuber_name": "유튜버명 또는 null",
        "style_keywords": ["키워드1", "키워드2", "키워드3"],
        "style_description": "스타일 설명",
        "confidence": "high/medium/low"
    }}
    
    스타일 키워드는 다음 중에서 선택하세요:
    - 스트릿, 와이드핏, 오버핏, 캐주얼, 스포티
    - 미니멀, 베이직, 모노톤, 심플, 클린
    - 컬러풀, 귀여운, 캐릭터, 팝, 플레이풀
    - 트렌디, 힙합, Y2K, 레트로, 걸크러시
    """
    
    try:
        response = llm_service.invoke(youtuber_prompt).strip()
        result = json.loads(response)
        
        youtuber_detected = result.get('youtuber_detected', False)
        youtuber_name = result.get('youtuber_name')
        style_keywords = result.get('style_keywords', [])
        style_description = result.get('style_description', '')
        confidence = result.get('confidence', 'low')
        
        if youtuber_detected and youtuber_name and confidence in ['high', 'medium']:
            print(f"🎬 유튜버 감지: {youtuber_name}")
            print(f"🎨 스타일 키워드: {style_keywords}")
            
            # 상태에 유튜버 정보 저장
            state['youtuber_context'] = {
                'name': youtuber_name,
                'style_keywords': style_keywords,
                'style_description': style_description,
                'confidence': confidence
            }
        else:
            print("🎬 유튜버 감지되지 않음")
            state['youtuber_context'] = {}
            
    except Exception as e:
        print(f"⚠️ 유튜버 스타일 분석 오류: {e}")
        state['youtuber_context'] = {}
    
    return state

def image_processor(state: AgentState) -> AgentState:
    """이미지 처리 및 임베딩 생성"""
    if not clip_model or not clip_processor:
        print("⚠️ CLIP 모델이 로드되지 않았습니다.")
        state['image_results'] = []
        return state
    
    try:
        # 이미지 데이터 가져오기 (업로드된 이미지 또는 이전 추천 상품의 이미지)
        image_data = state.get('input_image')
        
        # 이전 추천 결과에서 선택된 상품의 이미지 사용
        if not image_data and state.get('selected_product_index') is not None:
            previous_recommendations = state.get("recommendations", [])
            selected_index = state['selected_product_index']
            
            if 0 <= selected_index < len(previous_recommendations):
                selected_product = previous_recommendations[selected_index]
                image_data = selected_product.get('image_url')
                print(f"🖼️ 이전 추천 상품 {selected_index + 1}번의 이미지를 사용합니다: {selected_product.get('product_name')}")
            else:
                print("⚠️ 선택된 상품 인덱스가 유효하지 않습니다.")
                state['image_results'] = []
                return state
        
        if not image_data:
            print("⚠️ 이미지 데이터가 없습니다.")
            state['image_results'] = []
            return state
        
        image = None
        
        # Base64 데이터 URL인지 확인
        if image_data.startswith('data:image/'):
            # Base64 데이터 URL에서 이미지 추출
            try:
                # "data:image/jpeg;base64," 부분 제거
                if ';base64,' in image_data:
                    base64_data = image_data.split(';base64,')[1]
                else:
                    base64_data = image_data
                
                # Base64 패딩 추가 (필요한 경우)
                missing_padding = len(base64_data) % 4
                if missing_padding:
                    base64_data += '=' * (4 - missing_padding)
                
                # 추가 검증: Base64 문자열이 유효한지 확인
                try:
                    # 테스트 디코딩
                    test_decode = base64.b64decode(base64_data)
                    print(f"✅ Base64 검증 완료: {len(test_decode)} 바이트")
                except Exception as e:
                    print(f"⚠️ Base64 검증 실패: {e}")
                    # 패딩을 다시 조정
                    base64_data = base64_data.rstrip('=')
                    missing_padding = len(base64_data) % 4
                    if missing_padding:
                        base64_data += '=' * (4 - missing_padding)
                
                # Base64 디코딩
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes))
                print("✅ Base64 이미지 데이터 처리 완료")
            except Exception as e:
                print(f"⚠️ Base64 이미지 처리 오류: {e}")
                state['image_results'] = []
                return state
        else:
            # 일반 URL인 경우 HTTP 요청
            try:
                print(f"🌐 URL 이미지 다운로드: {image_data}")
                response = requests.get(image_data, timeout=15)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                print("✅ URL 이미지 다운로드 완료")
            except Exception as e:
                print(f"⚠️ URL 이미지 다운로드 오류: {e}")
                state['image_results'] = []
                return state
        
        if image is None:
            print("⚠️ 이미지를 로드할 수 없습니다.")
            state['image_results'] = []
            return state
        
        # 이미지 전처리 (RGB 변환)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("🔄 이미지를 RGB 모드로 변환했습니다.")
        
        # CLIP 모델로 이미지 임베딩 생성
        try:
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            
            # 임베딩을 numpy 배열로 변환
            image_embedding = image_features.cpu().numpy().flatten()
            
            # 정규화
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            
            state['image_embedding'] = image_embedding.tolist()
            print(f"✅ 이미지 임베딩 생성 완료: {len(image_embedding)} 차원")
            print(f"📊 임베딩 범위: {image_embedding.min():.4f} ~ {image_embedding.max():.4f}")
            
        except Exception as e:
            print(f"⚠️ CLIP 모델 임베딩 생성 오류: {e}")
            state['image_results'] = []
            return state
        
    except Exception as e:
        print(f"⚠️ 이미지 처리 오류: {e}")
        state['image_results'] = []
    
    return state

def predict_category_from_image(image_embedding: np.ndarray) -> str:
    """이미지 임베딩을 기반으로 카테고리 예측"""
    try:
        # 간단한 카테고리 예측 로직 (실제로는 더 정교한 모델 사용 가능)
        # 이미지 임베딩의 특정 차원들을 분석하여 카테고리 예측
        
        # 임베딩의 평균값과 표준편차를 기반으로 카테고리 예측
        embedding_mean = np.mean(image_embedding)
        embedding_std = np.std(image_embedding)
        
        # 간단한 규칙 기반 카테고리 예측
        if embedding_mean > 0.1:
            return "상의"
        elif embedding_std > 0.15:
            return "바지"
        elif embedding_mean < -0.1:
            return "신발"
        else:
            return "아우터"
            
    except Exception as e:
        print(f"⚠️ 카테고리 예측 실패: {e}")
        return "상의"  # 기본값

def process_product_embedding(row, user_embedding, base_url):
    """단일 상품의 이미지 임베딩을 처리하는 함수 (병렬 처리용)"""
    try:
        image = None
        image_source = None
        absolute_path = None
        
        # 1. 로컬 이미지 파일 처리
        if row.image_path and row.image_path.strip():
            image_path = row.image_path.strip()
            
            # 상대경로인 경우 절대경로로 변환
            if image_path.startswith('./'):
                relative_path = image_path[2:]
                absolute_path = os.path.join(base_url, relative_path)
            elif image_path.startswith('.'):
                relative_path = image_path.lstrip('.')
                absolute_path = os.path.join(base_url, relative_path.lstrip('/'))
            else:
                absolute_path = image_path
            
            if os.path.exists(absolute_path):
                try:
                    image = Image.open(absolute_path)
                    image_source = "local"
                except Exception as e:
                    print(f"⚠️ 로컬 이미지 로드 실패: {e}")
                    image = None
        
        # 2. URL 이미지 처리 (로컬 이미지가 없는 경우)
        if image is None and row.image_url and row.image_url.strip():
            try:
                response = requests.get(row.image_url, timeout=10)  # 타임아웃 단축
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                image_source = "url"
            except Exception as e:
                print(f"⚠️ URL 이미지 다운로드 실패: {e}")
        
        # 3. 이미지 처리 및 임베딩 생성
        if image is not None:
            try:
                # 캐싱된 임베딩 사용 또는 새로 생성
                product_embedding = get_image_embedding(absolute_path, row.image_url)
                
                if product_embedding is not None:
                    # 코사인 유사도 계산
                    similarity = np.dot(user_embedding, product_embedding)
                    
                    return {
                        'product_id': row.product_id,
                        'product_name': row.product_name,
                        'category': row.category,
                        'price': float(row.price) if row.price else 0,
                        'description': row.description,
                        'image_url': row.image_url,
                        'image_path': row.image_path,
                        'brand_kr': row.brand_kr,
                        'tags': row.tags if row.tags else [],
                        'similarity_score': float(similarity),
                        'image_source': image_source
                    }
            except Exception as e:
                print(f"⚠️ 임베딩 처리 실패: {e}")
        
        return None
    except Exception as e:
        print(f"⚠️ 상품 처리 오류: {e}")
        return None

def image_similarity_search(state: AgentState) -> AgentState:
    """이미지 유사도 기반 상품 검색 (카테고리 우선순위 적용)"""
    if not state.get('image_embedding'):
        print("⚠️ 이미지 임베딩이 없습니다.")
        state['image_results'] = []
        return state
    
    # 이미지 파일 base URL 설정
    base_url = IMAGE_BASE_URL
    
    try:
        # 데이터베이스에서 상품 이미지들과 유사도 계산
        with engine.connect() as conn:
            # 최적화된 쿼리: 이미지가 있는 상품만 가져오기
            query = f"""
                SELECT product_id, product_name, category, price, description, 
                       image_url, image_path, brand_kr, tags
                FROM products
                WHERE (image_path IS NOT NULL AND image_path != '' AND image_path != 'NULL') 
                   OR (image_url IS NOT NULL AND image_url != '' AND image_url != 'NULL')
                ORDER BY product_id DESC  -- 최신 상품 우선
                LIMIT {MAX_IMAGE_RESULTS * 3}  -- 더 많은 상품을 가져와서 카테고리별 필터링
            """
            
            result = conn.execute(text(query))
            products = []
            processed_count = 0
            error_count = 0
            
            # 사용자 이미지 임베딩
            user_embedding = np.array(state['image_embedding'])
            
            # 이미지에서 카테고리 예측
            predicted_category = predict_category_from_image(user_embedding)
            print(f"🔍 이미지 분석 결과 예측 카테고리: {predicted_category}")
            
            # 병렬 처리로 이미지 임베딩 계산
            with ThreadPoolExecutor(max_workers=4) as executor:
                # 각 상품에 대해 병렬로 임베딩 처리
                future_to_row = {
                    executor.submit(process_product_embedding, row, user_embedding, base_url): row 
                    for row in result
                }
                
                # 결과 수집
                for future in as_completed(future_to_row):
                    try:
                        product_data = future.result()
                        if product_data:
                            products.append(product_data)
                            processed_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                        print(f"⚠️ 병렬 처리 오류: {e}")
            
            # 카테고리별 상품 그룹화
            category_products = {}
            all_products = []
            
            for product in products:
                category = product['category']
                if category not in category_products:
                    category_products[category] = []
                category_products[category].append(product)
                all_products.append(product)
            
            # 카테고리별로 정렬하고 우선순위 적용
            final_products = []
            
            # 각 카테고리별로 유사도 순으로 정렬
            for category, category_items in category_products.items():
                category_items.sort(key=lambda x: x['similarity_score'], reverse=True)
                print(f"📂 카테고리 '{category}': {len(category_items)}개 상품 (최고 유사도: {category_items[0]['similarity_score']:.3f})")
            
            # 가장 유사한 상품의 카테고리 찾기
            if all_products:
                all_products.sort(key=lambda x: x['similarity_score'], reverse=True)
                best_match = all_products[0]
                best_category = best_match['category']
                print(f"🎯 가장 유사한 상품: {best_match['product_name']} (카테고리: {best_category}, 유사도: {best_match['similarity_score']:.3f})")
                
                # 디버깅: 상위 5개 상품의 카테고리 출력
                print("🔍 상위 5개 상품:")
                for i, product in enumerate(all_products[:5], 1):
                    print(f"   {i}. {product['product_name']} ({product['category']}, 유사도: {product['similarity_score']:.3f})")
                
                # 가장 유사한 카테고리의 상품들만 추천 (강제)
                if best_category in category_products:
                    best_category_products = category_products[best_category]
                    best_category_products.sort(key=lambda x: x['similarity_score'], reverse=True)
                    
                    # 같은 카테고리 상품들만 추천 (강제)
                    final_products = best_category_products[:MAX_IMAGE_RESULTS]
                    print(f"✅ 같은 카테고리 '{best_category}'에서 {len(final_products)}개 상품 추천")
                    
                    # 디버깅: 추천된 상품들 출력
                    print("📋 추천된 상품들:")
                    for i, product in enumerate(final_products[:5], 1):
                        print(f"   {i}. {product['product_name']} ({product['category']}, 유사도: {product['similarity_score']:.3f})")
                    
                    # 카테고리 검증
                    categories_in_result = set(p['category'] for p in final_products)
                    if len(categories_in_result) > 1:
                        print(f"⚠️ 경고: 결과에 여러 카테고리가 포함됨: {categories_in_result}")
                    else:
                        print(f"✅ 성공: 모든 추천 상품이 같은 카테고리 '{best_category}'입니다.")
                else:
                    # 예상치 못한 경우: 전체 유사도 순으로 정렬
                    final_products = all_products[:MAX_IMAGE_RESULTS]
            else:
                print("⚠️ 유사한 상품을 찾지 못했습니다.")
                final_products = []
            
            # 최종 결과 수 제한
            final_products = final_products[:MAX_IMAGE_RESULTS]
            
            state['image_results'] = final_products
            if final_products:
                print(f"✅ 이미지 유사 상품 {len(final_products)}개 찾음 (최고 유사도: {final_products[0]['similarity_score']:.3f})")
                print(f"📊 처리 통계: 성공 {processed_count}개, 실패 {error_count}개")
                category_distribution = []
                for cat in set(p['category'] for p in final_products):
                    count = len([p for p in final_products if p['category'] == cat])
                    category_distribution.append(f'{cat}({count})')
                print(f"📂 카테고리 분포: {', '.join(category_distribution)}")
            else:
                print("⚠️ 이미지 유사 상품을 찾지 못했습니다.")
            
    except Exception as e:
        print(f"⚠️ 이미지 유사도 검색 오류: {e}")
        state['image_results'] = []
    
    return state

def feedback_analyzer(state: AgentState) -> AgentState:
    """피드백 분석 및 슬롯 업데이트"""
    user_input = state['messages'][-1].content if state['messages'] else ""
    current_slots = state.get('slots', {})
    
    feedback_prompt = f"""
    다음 사용자 피드백을 분석하여 추천 조건을 업데이트해주세요:
    
    현재 조건: {current_slots}
    사용자 피드백: "{user_input}"
    
    다음 형식으로 JSON을 반환하세요:
    {{
        "category": "상의/바지/신발/가방/패션소품",
        "price_range": "low/medium/high",
        "style": "캐주얼/스포티/오피스/데이트/스트릿",
        "color": "색상명",
        "brand": "브랜드명",
        "material": "소재명",
        "size": "사이즈",
        "additional_keywords": ["추가 키워드들"]
    }}
    
    피드백에 따라 적절히 조건을 변경하세요:
    - "비싸요" → price_range를 낮춤
    - "싸요" → price_range를 높임
    - "다른 스타일로" → style을 변경
    - "다른 색으로" → color를 변경
    - "다른 브랜드로" → brand를 변경
    - "좋아요" → 현재 조건 유지하되 더 다양하게
    
    해당하는 정보가 없으면 null로 설정하세요.
    """
    
    # 프롬프팅 기반 피드백 분석
    feedback_prompt = f"""
    다음 사용자 피드백을 분석하여 추천 조건을 업데이트해주세요:
    
    현재 조건: {current_slots}
    사용자 피드백: "{user_input}"
    
    다음 형식으로 JSON을 반환하세요:
    {{
        "feedback_type": "positive/negative/neutral",
        "updated_slots": {{
            "category": "상의/바지/신발/가방/패션소품",
            "price_range": "low/medium/high",
            "style": "캐주얼/스포티/오피스/데이트/스트릿",
            "color": "색상명",
            "brand": "브랜드명",
            "material": "소재명",
            "size": "사이즈",
            "additional_keywords": ["추가 키워드들"]
        }}
    }}
    
    피드백 분석 규칙:
    - "비싸요", "4만원 미만", "저렴한" → price_range를 "low"로 변경 (기존 조건 유지)
    - "싸요", "비싼" → price_range를 "high"로 변경 (기존 조건 유지)
    - "다른 스타일로" → style을 다른 스타일로 변경 (기존 조건 유지)
    - "다른 색으로" → color를 null로 설정 (기존 조건 유지)
    - "다른 브랜드로" → brand를 null로 설정 (기존 조건 유지)
    - "좋아요" → 현재 조건 유지하되 더 다양하게
    
    중요: 기존 조건(category, additional_keywords 등)은 그대로 유지하고, 
    피드백에 해당하는 조건만 업데이트하세요!
    
    예시: 현재 조건에 category="바지", additional_keywords=["버뮤다"]가 있으면
    "4만원 미만으로 추천해줘" → price_range="low"로 변경하되, 
    category와 additional_keywords는 그대로 유지
    """
    
    try:
        response = llm_service.invoke(feedback_prompt)
        feedback_result = json.loads(response)
        
        feedback_type = feedback_result.get('feedback_type', 'neutral')
        updated_slots = feedback_result.get('updated_slots', {})
        
        # 기존 슬롯과 병합 (null이 아닌 값만 업데이트)
        for key, value in updated_slots.items():
            if value is not None:
                current_slots[key] = value
        
        state['slots'] = current_slots
        state['feedback'] = feedback_type
        print(f"피드백 분석 완료: {current_slots}")
        
    except Exception as e:
        print(f"피드백 분석 오류: {e}")
        state['feedback'] = 'neutral'
    
    return state

def text_filter_parser(state: AgentState) -> AgentState:
    """텍스트 기반 조건 추출 (슬롯)"""
    user_input = state['messages'][-1].content if state['messages'] else ""
    
    # 의도가 recommendation이 아닌 경우 기본 슬롯만 설정
    if state.get('intent') != 'recommendation':
        state['slots'] = {
            "category": None,
            "price_range": None,
            "style": None,
            "color": None,
            "brand": None,
            "material": None,
            "size": None,
            "additional_keywords": []
        }
        return state
    
    # 이전 슬롯 가져오기 (멀티턴 대화를 위해)
    previous_slots = state.get('slots', {})
    if not previous_slots:
        previous_slots = {
            "category": None,
            "price_range": None,
            "style": None,
            "color": None,
            "brand": None,
            "material": None,
            "size": None,
            "additional_keywords": []
        }
    
    filter_prompt = f"""
    다음 입력에서 상품 추천에 필요한 조건들을 추출해주세요:
    
    이전 조건: {previous_slots}
    현재 입력: "{user_input}"
    
    다음 형식으로 JSON을 반환하세요:
    {{
        "category": "상의/바지/신발/가방/패션소품",
        "price_range": "low/medium/high",
        "style": "캐주얼/스포티/오피스/데이트/스트릿",
        "color": "색상명",
        "brand": "브랜드명",
        "material": "소재명",
        "size": "사이즈",
        "additional_keywords": ["추가 키워드들"]
    }}
    
    카테고리 매핑 규칙 (중요!):
    - "팬츠", "바지", "진", "슬랙스", "트레이닝팬츠" → "바지"
    - "티셔츠", "상의", "셔츠", "니트", "후드", "맨투맨" → "상의"
    - "운동화", "스니커즈", "신발", "구두", "샌들" → "신발"
    - "가방", "백팩", "크로스백", "토트백" → "가방"
    
    가격대 매핑:
    - "싼", "저렴한", "4만원 미만", "3만원 미만", "2만원 미만" → "low"
    - "보통", "적당한", "5만원 정도" → "medium"  
    - "비싼", "고급", "프리미엄", "10만원 이상" → "high"
    
    스타일 매핑:
    - "캐주얼", "일상", "편한", "데일리" → "캐주얼"
    - "스포티", "운동", "액티브", "피트니스" → "스포티"
    - "오피스", "비즈니스", "정장", "깔끔한" → "오피스"
    - "데이트", "로맨틱", "여성스러운" → "데이트"
    - "스트릿", "힙합", "스케이트", "힙한" → "스트릿"
    
    색상 매핑:
    - "검정", "블랙", "검은색" → "검정"
    - "흰색", "화이트", "하얀색" → "흰색"
    - "파란색", "블루", "청색" → "파란색"
    - "빨간색", "레드", "적색" → "빨간색"
    - "초록색", "그린", "녹색" → "초록색"
    - "노란색", "옐로우", "황색" → "노란색"
    - "보라색", "퍼플", "자주색" → "보라색"
    - "주황색", "오렌지", "주황" → "주황색"
    - "분홍색", "핑크", "연분홍" → "분홍색"
    - "회색", "그레이", "회색" → "회색"
    - "베이지", "크림", "아이보리" → "베이지"
    - "네이비", "다크블루" → "네이비"
    
    소재 매핑:
    - "면", "코튼", "순면" → "면"
    - "니트", "울", "스웨터" → "니트"
    - "가죽", "레더", "가죽" → "가죽"
    - "데님", "청", "데님" → "데님"
    - "실크", "비단", "실크" → "실크"
    - "린넨", "마" → "린넨"
    - "폴리에스터", "폴리" → "폴리에스터"
    - "나일론", "나일론" → "나일론"
    - "스웨트", "후리스" → "스웨트"
    
    브랜드 매핑 (실제 브랜드명만):
    - "나이키", "아디다스", "유니클로", "ZARA", "H&M", "무신사", "크림소다랩", "버던트", "난데", "구찌", "샤넬", "루이비통" 등
    
    additional_keywords 매핑 (상품 특징):
    - "버뮤다", "데님", "카고", "와이드", "스키니", "반팔", "라운드넥", "오버핏", "하프팬츠", "7부", "8부", "밴딩", "원턱", "핀턱", "크롭", "롱", "숏", "미니", "맥시" 등
    
    중요: 카테고리 키워드는 additional_keywords에 포함하지 마세요!
    - 카테고리 키워드: "팬츠", "바지", "티셔츠", "신발", "가방" 등
    - additional_keywords에 포함할 키워드: "버뮤다", "데님", "카고", "와이드", "스키니", "반팔", "라운드넥", "오버핏" 등
    
    필수: 사용자가 요청한 구체적인 키워드는 반드시 additional_keywords에 포함하세요!
    - "버뮤다 팬츠" → additional_keywords: ["버뮤다"] (brand가 아님!)
    - "데님 바지" → additional_keywords: ["데님"] (brand가 아님!)
    - "카고 팬츠" → additional_keywords: ["카고"] (brand가 아님!)
    - "와이드 팬츠" → additional_keywords: ["와이드"] (brand가 아님!)
    - "스키니 진" → additional_keywords: ["스키니"] (brand가 아님!)
    
    주의: "버뮤다", "데님", "카고", "와이드", "스키니" 등은 브랜드명이 아닌 상품 특징입니다!
    브랜드명은 "나이키", "아디다스", "유니클로" 등 실제 브랜드명만 해당됩니다.
    
    절대 금지: "버뮤다", "데님", "카고", "와이드", "스키니"를 brand 필드에 넣지 마세요!
    이들은 모두 additional_keywords에만 들어가야 합니다!
    
    강제 규칙:
    - "버뮤다" → 반드시 additional_keywords에만 추가
    - "데님" → 반드시 additional_keywords에만 추가  
    - "카고" → 반드시 additional_keywords에만 추가
    - "와이드" → 반드시 additional_keywords에만 추가
    - "스키니" → 반드시 additional_keywords에만 추가
    
    이 키워드들을 brand 필드에 넣으면 안 됩니다!
    
    additional_keywords 추출 예시:
    - "버뮤다 팬츠" → ["버뮤다"] (팬츠는 카테고리로 처리)
    - "데님 바지" → ["데님"] (바지는 카테고리로 처리)
    - "카고 팬츠" → ["카고"] (팬츠는 카테고리로 처리)
    - "와이드 팬츠" → ["와이드"] (팬츠는 카테고리로 처리)
    - "스키니 진" → ["스키니"] (진은 카테고리로 처리)
    - "반팔 티셔츠" → ["반팔"] (티셔츠는 카테고리로 처리)
    - "라운드넥 티셔츠" → ["라운드넥"] (티셔츠는 카테고리로 처리)
    - "오버핏 티셔츠" → ["오버핏"] (티셔츠는 카테고리로 처리)
    
    멀티턴 대화 규칙:
    - 이전 조건이 있으면 그대로 유지하세요
    - 새로운 정보만 업데이트하세요
    - 예: 이전에 category="바지", additional_keywords=["버뮤다"]가 있었는데
      "4만원 미만으로 추천해줘"라고 하면
      → category="바지", additional_keywords=["버뮤다"], price_range="low"로 설정
    
    해당하는 정보가 없으면 null로 설정하세요.
    """
    
    try:
        response = llm_service.invoke(filter_prompt)
        slots = json.loads(response)
        
        print(f"🔍 LLM 원본 응답: {response}")
        print(f"🔍 LLM 파싱 결과: {slots}")
        
        # Slot 검증 및 정규화
        slots = validate_and_normalize_slots(slots)
        
        state['slots'] = slots
        print(f"조건 추출: {slots}")
    except Exception as e:
        print(f"조건 추출 오류: {e}")
        state['slots'] = {
            "category": None,
            "price_range": None,
            "style": None,
            "color": None,
            "brand": None,
            "material": None,
            "size": None,
            "additional_keywords": []
        }
    
    return state

def validate_and_normalize_slots(slots: dict) -> dict:
    """Slot 검증 및 정규화"""
    # 카테고리 정규화
    if slots.get('category'):
        category_mapping = {
            '팬츠': '바지', '진': '바지', '슬랙스': '바지', '트레이닝팬츠': '바지',
            '상의': '상의', '셔츠': '상의', '니트': '상의', '후드': '상의', '맨투맨': '상의',
            '운동화': '신발', '스니커즈': '신발', '구두': '신발', '샌들': '신발',
            '가방': '가방', '백팩': '가방', '크로스백': '가방', '토트백': '가방'
        }
        slots['category'] = category_mapping.get(slots['category'], slots['category'])
    
    # 가격대 정규화
    if slots.get('price_range'):
        price_mapping = {
            'low': 'low', 'medium': 'medium', 'high': 'high',
            '싼': 'low', '저렴한': 'low', '보통': 'medium', '비싼': 'high', '고급': 'high'
        }
        slots['price_range'] = price_mapping.get(slots['price_range'], slots['price_range'])
    
    # 색상 정규화 (하드코딩)
    if slots.get('color'):
        print(f"🔍 색상 정규화 전: {slots['color']}")
        color_mapping = {
            '검정': ['검정', '블랙', '검은색'],
            '흰색': ['흰색', '화이트', '하얀색'],
            '파란색': ['파란색', '블루', '청색'],
            '빨간색': ['빨간색', '레드', '적색'],
            '초록색': ['초록색', '그린', '녹색'],
            '노란색': ['노란색', '옐로우', '황색'],
            '보라색': ['보라색', '퍼플', '자주색'],
            '주황색': ['주황색', '오렌지', '주황'],
            '분홍색': ['분홍색', '핑크', '연분홍'],
            '회색': ['회색', '그레이'],
            '베이지': ['베이지', '크림', '아이보리'],
            '네이비': ['네이비', '다크블루']
        }
        original_color = slots['color']
        slots['color'] = color_mapping.get(slots['color'], [slots['color']])
        print(f"🎨 색상 정규화 후: {original_color} → {slots['color']}")
    
    # 스타일 정규화
    if slots.get('style'):
        style_mapping = {
            '일상': '캐주얼', '편한': '캐주얼', '데일리': '캐주얼',
            '운동': '스포티', '액티브': '스포티', '피트니스': '스포티',
            '비즈니스': '오피스', '정장': '오피스', '깔끔한': '오피스',
            '로맨틱': '데이트', '여성스러운': '데이트',
            '힙합': '스트릿', '스케이트': '스트릿', '힙한': '스트릿'
        }
        slots['style'] = style_mapping.get(slots['style'], slots['style'])
    
    # 소재 정규화
    if slots.get('material'):
        material_mapping = {
            '코튼': '면', '순면': '면', '울': '니트', '스웨터': '니트',
            '레더': '가죽', '청': '데님', '비단': '실크', '마': '린넨',
            '폴리에스터': '폴리에스터', '후리스': '스웨트'
        }
        slots['material'] = material_mapping.get(slots['material'], slots['material'])
    
    # additional_keywords 정리 (빈 문자열 제거, 중복 제거)
    if slots.get('additional_keywords'):
        keywords = [kw.strip() for kw in slots['additional_keywords'] if kw.strip()]
        slots['additional_keywords'] = list(set(keywords))  # 중복 제거
    
    return slots

def recommendation_generator(state: AgentState) -> AgentState:
    """추천 알고리즘 수행 (텍스트 + 이미지 기반)"""
    slots = state.get('slots', {})
    image_results = state.get('image_results', [])
    
    # 이미지 결과가 있으면 우선 사용
    if image_results:
        state['recommendations'] = image_results
        print(f"이미지 기반 추천 상품 {len(image_results)}개 생성")
        return state
    
    # 텍스트 기반 추천
    try:
        with engine.connect() as conn:
            query = """
                SELECT product_id, product_name, category, price, description, 
                       image_url, brand_kr, tags
                FROM products
                WHERE 1=1
            """
            params = {}
            
            # 카테고리 필터
            if slots.get('category'):
                query += " AND category = :category"
                params['category'] = slots['category']
            
            # 가격 필터
            if slots.get('price_range') == 'low':
                query += f" AND price <= {DEFAULT_PRICE_RANGES['low']}"
            elif slots.get('price_range') == 'medium':
                query += f" AND price BETWEEN {DEFAULT_PRICE_RANGES['low']} AND {DEFAULT_PRICE_RANGES['medium']}"
            elif slots.get('price_range') == 'high':
                query += f" AND price > {DEFAULT_PRICE_RANGES['medium']}"
            
            # 브랜드 필터
            if slots.get('brand'):
                query += " AND brand_kr ILIKE :brand"
                params['brand'] = f"%{slots['brand']}%"
            
            # 스타일 필터 (description에서 검색)
            if slots.get('style'):
                query += " AND COALESCE(description, '') ILIKE :style"
                params['style'] = f"%{slots['style']}%"
            
            # 색상 필터 (product_name과 description에서 검색)
            if slots.get('color'):
                if isinstance(slots['color'], list):
                    # 여러 색상 용어로 검색
                    color_conditions = []
                    for i, color_term in enumerate(slots['color']):
                        color_conditions.append(f"(product_name ILIKE :color_{i} OR COALESCE(description, '') ILIKE :color_{i})")
                        params[f'color_{i}'] = f"%{color_term}%"
                    query += f" AND ({' OR '.join(color_conditions)})"
                else:
                    # 단일 색상으로 검색
                    query += " AND (product_name ILIKE :color OR COALESCE(description, '') ILIKE :color)"
                    params['color'] = f"%{slots['color']}%"
            
            # 소재 필터 (product_name과 description에서 검색)
            if slots.get('material'):
                query += " AND (product_name ILIKE :material OR COALESCE(description, '') ILIKE :material)"
                params['material'] = f"%{slots['material']}%"
            
            # 추가 키워드 필터 (tags 필드 + 상품명 + 설명 활용)
            if slots.get('additional_keywords') and len(slots['additional_keywords']) > 0:
                # 빈 문자열이 아닌 키워드만 필터링
                valid_keywords = [kw.strip() for kw in slots['additional_keywords'] if kw.strip()]
                
                if valid_keywords:
                    # 각 키워드에 대해 tags, 상품명, 설명에서 검색
                    keyword_conditions = []
                    for i, keyword in enumerate(valid_keywords):
                        # tags, 상품명, 설명에서 키워드 검색 (NULL 처리 포함)
                        keyword_conditions.append(f"""
                            (tags::text ILIKE :keyword_{i} OR 
                             product_name ILIKE :keyword_{i} OR 
                             COALESCE(description, '') ILIKE :keyword_{i})
                        """)
                        params[f'keyword_{i}'] = f"%{keyword}%"
                    
                    if keyword_conditions:
                        query += f" AND ({' OR '.join(keyword_conditions)})"
            
            query += f" ORDER BY RANDOM() LIMIT {MAX_RECOMMENDATIONS}"
            
            result = conn.execute(text(query), params)
            products = []
            
            for row in result:
                product = {
                    'product_id': row.product_id,
                    'product_name': row.product_name,
                    'category': row.category,
                    'price': float(row.price) if row.price else 0,
                    'description': row.description,
                    'image_url': row.image_url,
                    'brand_kr': row.brand_kr,
                    'tags': row.tags if row.tags else []
                }
                products.append(product)
            
            state['recommendations'] = products
            print(f"텍스트 기반 추천 상품 {len(products)}개 생성")
            
    except Exception as e:
        print(f"추천 생성 오류: {e}")
        state['recommendations'] = []
    
    return state

def similar_product_finder(state: AgentState) -> AgentState:
    """특정 상품과 유사한 상품을 찾는 노드"""
    user_input = state['messages'][-1].content if state['messages'] else ""
    
    # 상품 번호 추출
    product_number = extract_product_number(user_input)
    
    if product_number <= 0:
        # 상품 번호를 찾을 수 없으면 일반 대화로 처리
        return conversation_agent(state)
    
    # 이전 추천 결과에서 해당 번호의 상품 찾기
    previous_recommendations = state.get('previous_recommendations', [])
    print(f"🔍 이전 추천 결과 개수: {len(previous_recommendations)}")
    print(f"🔍 요청한 상품 번호: {product_number}")
    
    if not previous_recommendations or product_number > len(previous_recommendations):
        response_text = f"죄송합니다. {product_number}번 상품을 찾을 수 없습니다. 먼저 상품을 추천받아주세요."
        ai_message = AIMessage(content=response_text)
        state['messages'].append(ai_message)
        return state
    
    target_product = previous_recommendations[product_number - 1]
    product_id = target_product.get('product_id')
    
    print(f"🎯 대상 상품 ID: {product_id}")
    print(f"🎯 대상 상품명: {target_product.get('product_name', 'N/A')}")
    
    if not product_id:
        response_text = f"죄송합니다. {product_number}번 상품의 정보를 찾을 수 없습니다."
        ai_message = AIMessage(content=response_text)
        state['messages'].append(ai_message)
        return state
    
    print(f"🎯 {product_number}번 상품과 유사한 상품을 찾습니다...")
    print(f"   대상 상품: {target_product['product_name']}")
    
    # 유사 상품 검색
    similar_products = get_similar_products_by_id(product_id, MAX_RECOMMENDATIONS)
    
    if similar_products:
        # 결과를 state에 저장
        state['recommendations'] = similar_products
        state['previous_recommendations'] = similar_products  # 다음 요청을 위해 저장
        
        response_text = f"{product_number}번 상품 '{target_product['product_name']}'과 유사한 상품 {len(similar_products)}개를 찾았습니다:\n\n"
        
        for i, product in enumerate(similar_products, 1):
            response_text += f"{i}. {product['product_name']}\n"
            if product.get('price'):
                response_text += f"   가격: {product['price']:,}원\n"
            if product.get('brand_kr'):
                response_text += f"   브랜드: {product['brand_kr']}\n"
            response_text += "\n"
        
        ai_message = AIMessage(content=response_text)
        state['messages'].append(ai_message)
        print(f"✅ {product_number}번 상품 유사 상품 {len(similar_products)}개 찾음")
        print(f"💾 previous_recommendations 저장됨: {len(similar_products)}개")
    else:
        response_text = f"죄송합니다. {product_number}번 상품과 유사한 상품을 찾을 수 없습니다."
        ai_message = AIMessage(content=response_text)
        state['messages'].append(ai_message)
    
    return state

def conversation_agent(state: AgentState) -> AgentState:
    """일반 잡담, 감성 응대 처리"""
    user_input = state['messages'][-1].content if state['messages'] else ""
    
    # 일반 대화 응답 생성
    chat_prompt = f"""
    다음 사용자 입력에 대한 친근하고 도움이 되는 응답을 생성해주세요:
    
    사용자: "{user_input}"
    
    패션 추천 챗봇의 입장에서 자연스럽고 친근하게 응답해주세요.
    """
    
    try:
        response_text = llm_service.invoke(chat_prompt)
        ai_message = AIMessage(content=response_text)
        state['messages'].append(ai_message)
        print(f"대화 응답 생성: {response_text[:50]}...")
    except Exception as e:
        print(f"대화 응답 생성 오류: {e}")
        ai_message = AIMessage(content="죄송합니다. 다시 말씀해주세요.")
        state['messages'].append(ai_message)
    
    # 대화는 여기서 끝남 - output_node를 거치지 않음
    return state

def output_node(state: AgentState) -> AgentState:
    """최종 응답 포맷 구성 및 출력"""
    recommendations = state.get('recommendations', [])
    feedback = state.get('feedback')
    intent = state.get('intent')
    
    if recommendations:
        # 추천 결과를 previous_recommendations에 저장 (다음 요청을 위해)
        state['previous_recommendations'] = recommendations
        print(f"💾 output_node에서 previous_recommendations 저장: {len(recommendations)}개")
        
        if intent == "image_search":
            response_text = f"이미지와 유사한 상품 {len(recommendations)}개를 찾았습니다:\n\n"
        else:
            response_text = f"추천 상품 {len(recommendations)}개를 찾았습니다:\n\n"
        
        # 사용자 기억에 따라 상품 표시 (설정 가능)
        display_count = min(DEFAULT_DISPLAY_COUNT, len(recommendations))
        for i, product in enumerate(recommendations[:display_count], 1):
            response_text += f"{i}. {product['product_name']} ({product['brand_kr']})\n"
            response_text += f"   가격: {product['price']:,}원\n"
            response_text += f"   카테고리: {product['category']}\n"
            if product.get('similarity_score'):
                response_text += f"   유사도: {product['similarity_score']:.2f}\n"
            response_text += "\n"
        
        if feedback == "positive":
            response_text += "좋아하신다니 기뻐요! 더 많은 추천이 필요하시면 언제든 말씀해주세요. 😊"
        elif feedback == "negative":
            response_text += "아쉽네요. 다른 조건으로 다시 추천해드릴게요!"
        else:
            response_text += "이런 상품들은 어떠세요? 더 구체적인 요청이 있으시면 말씀해주세요!"
    else:
        response_text = "죄송합니다. 조건에 맞는 상품을 찾지 못했습니다. 다른 조건으로 다시 시도해보세요."
    
    ai_message = AIMessage(content=response_text)
    state['messages'].append(ai_message)
    print(f"최종 응답 생성: {response_text[:50]}...")
    
    return state 