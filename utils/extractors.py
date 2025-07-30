"""
필터 추출 관련 유틸리티 함수들
"""
import json
from typing import Dict, List, Any, Optional, Tuple

# LLM import는 나중에 main에서 가져올 예정
llm = None

def set_llm(llm_instance):
    """LLM 인스턴스를 설정"""
    global llm
    llm = llm_instance

def extract_seasonal_keywords(user_input: str) -> List[str]:
    """사용자 입력에서 계절이나 조건을 유추해서 관련 키워드들 생성"""
    prompt = f"""
    다음 사용자 입력에서 계절, 상황, 스타일 등을 유추하여 관련 키워드들을 추출해주세요:
    사용자 입력: "{user_input}"
    
    추출할 키워드 유형:
    1. 계절 관련: 봄, 여름, 가을, 겨울, 시원한, 따뜻한 등
    2. 상황 관련: 데이트, 회사, 운동, 여행, 파티 등
    3. 스타일 관련: 캐주얼, 정장, 스포티, 엘레간트 등
    4. 소재 관련: 데님, 코튼, 니트, 가죽 등
    5. 색상 관련: 검은색, 흰색, 파란색 등
    
    반드시 문자열 배열 형태로 답변해주세요.
    예시: ["여름", "시원한", "캐주얼", "데님"]
    
    JSON 배열 형태로 답변해주세요.
    """
    
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        keywords = json.loads(response.content)
        
        # 딕셔너리 리스트인 경우 문자열 리스트로 변환
        if keywords and isinstance(keywords[0], dict):
            converted_keywords = []
            for item in keywords:
                if isinstance(item, dict):
                    # 딕셔너리의 값들을 추출
                    for value in item.values():
                        if value and isinstance(value, str):
                            converted_keywords.append(value)
                else:
                    converted_keywords.append(str(item))
            keywords = converted_keywords
        
        print(f"유추된 키워드들: {keywords}")
        return keywords
    except Exception as e:
        print(f"키워드 유추 오류: {e}")
        return []

def extract_price_info(user_input: str) -> Dict[str, Optional[int]]:
    """사용자 입력에서 가격 정보 추출"""
    prompt = f"""
    다음 사용자 입력에서 가격 정보를 추출해주세요:
    사용자 입력: "{user_input}"
    
    JSON 형태로 다음 정보를 추출해주세요:
    {{
        "max_price": 숫자 (최대 가격, 없으면 null),
        "min_price": 숫자 (최소 가격, 없으면 null)
    }}
    
    예시:
    - "3만원 미만" → {{"max_price": 30000, "min_price": null}}
    - "5만원 이상" → {{"max_price": null, "min_price": 50000}}
    - "2만원~5만원" → {{"max_price": 50000, "min_price": 20000}}
    - "저렴한" → {{"max_price": 30000, "min_price": null}}
    - "비싼" → {{"max_price": null, "min_price": 100000}}
    - "가격대가 4만원 이상" → {{"max_price": null, "min_price": 40000}}
    - "4만원 이상" → {{"max_price": null, "min_price": 40000}}
    - "4만원 이하" → {{"max_price": 40000, "min_price": null}}
    
    주의사항:
    - "X만원 이상" = 최소 가격 (min_price)
    - "X만원 이하" = 최대 가격 (max_price)
    - "X만원 미만" = 최대 가격 (max_price)
    - "X만원 초과" = 최소 가격 (min_price)
    
    JSON만 답변해주세요.
    """
    
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        price_info = json.loads(response.content)
        print(f"추출된 가격 정보: {price_info}")
        return price_info
    except Exception as e:
        print(f"가격 정보 추출 오류: {e}")
        return {}

def get_category_keyword_mapping() -> Dict[str, List[str]]:
    """카테고리별 키워드 매핑 생성"""
    return {
        '상의': ['티셔츠', '셔츠', '니트', '후드', '맨투맨', '폴로', '블라우스', '탑', '상의'],
        '바지': ['청바지', '슬랙스', '트레이닝', '쇼츠', '팬츠', '버뮤다', '와이드', '스키니', '바지'],
        '아우터': ['자켓', '코트', '패딩', '후드', '바람막이', '트렌치', '블레이저', '아우터'],
        '신발': ['운동화', '스니커즈', '로퍼', '옥스포드', '샌들', '부츠', '신발'],
        '가방': ['백팩', '토트', '크로스백', '클러치', '숄더백', '가방'],
        '패션소품': ['모자', '양말', '벨트', '시계', '주얼리', '선글라스', '스카프', '소품']
    }

def find_best_category(user_input: str, category_mapping: Dict[str, List[str]], previous_context: Optional[Dict] = None) -> Tuple[Optional[str], int]:
    """사용자 입력에서 가장 적합한 카테고리 찾기"""
    best_category = None
    best_score = 0
    
    # 이전 컨텍스트에서 카테고리 정보 가져오기
    previous_category = None
    if previous_context and previous_context.get('products'):
        previous_products = previous_context.get('products', [])
        if previous_products:
            categories = set()
            for product in previous_products:
                categories.add(product.get('category', ''))
            if categories:
                previous_category = list(categories)[0]  # 첫 번째 카테고리 사용
                print(f"이전 대화 카테고리: {previous_category}")
    
    for category, keywords in category_mapping.items():
        score = 0
        
        # 키워드 매칭
        for keyword in keywords:
            if keyword.lower() in user_input.lower():
                score += 1
        
        # 이전 카테고리와 일치하면 보너스
        if previous_category and category == previous_category:
            score += 3
            print(f"이전 카테고리 보너스 적용: {category} (+3점)")
        
        # 특별한 키워드 매칭 (브랜드명 등)
        if category == '바지' and any(word in user_input.lower() for word in ['뮤다팬츠', '버뮤다', '팬츠', '바지', '버뮤다팬츠']):
            score += 5  # 버뮤다 팬츠는 확실히 바지 카테고리
            print(f"버뮤다 팬츠 키워드 매칭: {category} (+5점)")
        elif category == '가방' and any(word in user_input.lower() for word in ['가방', '백팩', '토트']):
            score += 2
        elif category == '상의' and any(word in user_input.lower() for word in ['티셔츠', '셔츠', '상의']):
            score += 2
        elif category == '신발' and any(word in user_input.lower() for word in ['운동화', '신발', '스니커즈']):
            score += 2
        elif category == '아우터' and any(word in user_input.lower() for word in ['자켓', '코트', '아우터']):
            score += 2
        
        if score > best_score:
            best_score = score
            best_category = category
    
    return best_category, best_score

def extract_filters(user_input: str, user_preferences: Optional[Dict] = None, previous_context: Optional[Dict] = None) -> Dict[str, Any]:
    """사용자 입력에서 필터 정보 추출 (이전 조건 누적)"""
    print(f"\n=== extract_filters 시작: '{user_input}' ===")
    
    # 동적으로 카테고리 매핑 생성
    category_mapping = get_category_keyword_mapping()
    print(f"카테고리 매핑: {list(category_mapping.keys())}")
    
    # 사용자 입력에서 가장 적합한 카테고리 찾기
    best_category, score = find_best_category(user_input, category_mapping, previous_context)
    print(f"동적 카테고리 분류 결과: '{best_category}' (점수: {score})")
    
    # 계절이나 조건을 유추해서 관련 키워드들 생성
    seasonal_keywords = extract_seasonal_keywords(user_input)
    
    # 가격 정보 추출
    price_info = extract_price_info(user_input)
    
    # 이전 컨텍스트에서 조건들 추출 (첫 번째 요청이 아닐 때만)
    previous_conditions = {}
    if previous_context and previous_context.get('products'):
        previous_products = previous_context.get('products', [])
        if previous_products is None:
            previous_products = []
        previous_method = previous_context.get('recommendation_method', '')
        
        # 이전 추천에서 언급된 조건들 추출
        if previous_products:
            # 이전 상품들의 특징을 분석하여 조건 추출
            categories = set()
            keywords = set()
            
            for product in previous_products:
                categories.add(product.get('category', ''))
                # 상품명, 설명, 태그에서 키워드 추출
                product_name = product.get('product_name', '').lower()
                description = product.get('description', '').lower()
                tags = product.get('tags', [])
                
                # 키워드 추출
                for tag in tags:
                    keywords.add(tag.lower())
                
                # 상품명에서 키워드 추출
                words = product_name.split()
                for word in words:
                    if len(word) >= 2:
                        keywords.add(word.lower())
            
            previous_conditions = {
                'categories': list(categories),
                'keywords': list(keywords),
                'method': previous_method
            }
            print(f"이전 조건 발견: {previous_conditions}")
    
    # LLM을 사용한 추가 분석 (이전 조건 누적)
    filter_prompt = f"""
    다음 사용자 입력에서 상품 필터 정보를 추출해주세요:
    현재 입력: "{user_input}"
    
    유추된 키워드들: {seasonal_keywords}
    가격 정보: {price_info}
    
    데이터베이스에 존재하는 카테고리들: {list(category_mapping.keys())}
    
    현재 입력을 분석하여 JSON 형태로 다음 정보를 추출해주세요:
    {{
        "category": "위 카테고리 중 하나 (없으면 null)",
        "brand": "브랜드명 (없으면 null)",
        "user_keywords": ["사용자가 직접 입력한 키워드1", "키워드2", ...] (사용자가 명시적으로 요청한 키워드들),
        "inferred_tags": ["유추된 태그1", "태그2", ...] (시스템이 유추한 키워드들),
        "keyword": "사용자가 요청한 구체적인 상품명 (사용자 키워드 우선)",
        "max_price": 숫자 (최대 가격, 없으면 null),
        "min_price": 숫자 (최소 가격, 없으면 null)
    }}
    
    카테고리 분류 규칙:
    - "버뮤다 팬츠", "팬츠", "바지" → 반드시 "바지" 카테고리
    - "티셔츠", "상의", "셔츠" → 반드시 "상의" 카테고리
    - "운동화", "신발" → 반드시 "신발" 카테고리
    - "가방", "백팩" → 반드시 "가방" 카테고리
    - "자켓", "코트" → 반드시 "아우터" 카테고리
    
    keyword 추출 방법:
    1. 사용자가 요청한 구체적인 상품명을 그대로 넣어주세요
    2. 소재, 스타일, 디자인 등의 구체적인 요구사항도 keyword에 포함하세요
    3. 사용자가 입력한 모든 중요한 키워드를 그대로 유지하세요
    4. 조사나 접미사(을, 를, 이, 에, 의, 과, 와, 로, 추천, 해줘 등)는 제외하세요
    5. "~인데 ~이면 좋겠어" 형태에서는 앞부분과 뒷부분 모두 keyword에 포함하세요
    
    user_keywords 분리 규칙 (중요!):
    - "버뮤다 팬츠" → ["버뮤다", "팬츠"] (반드시 개별 단어로 분리)
    - "데님 소재" → ["데님", "소재"] (반드시 개별 단어로 분리)
    - "검은색 티셔츠" → ["검은색", "티셔츠"] (반드시 개별 단어로 분리)
    - "버뮤다 팬츠인데 데님 소재" → ["버뮤다", "팬츠", "데님", "소재"] (모든 단어를 개별적으로 분리)
    
    절대 하지 말 것:
    - ["버뮤다 팬츠", "데님"] (X) - 단어를 묶지 말 것
    - ["데님 소재"] (X) - 단어를 묶지 말 것
    
    예시:
    - "버뮤다 팬츠 추천해줘" → {{"category": "바지", "user_keywords": ["버뮤다", "팬츠"], "inferred_tags": ["여름", "시원한"], "keyword": "버뮤다 팬츠"}}
    - "버뮤다 팬츠인데 데님 소재로 된걸로" → {{"category": "바지", "user_keywords": ["버뮤다", "팬츠", "데님", "소재"], "inferred_tags": ["여름", "시원한"], "keyword": "버뮤다 팬츠 데님 소재"}}
    - "티셔츠 추천해줘" → {{"category": "상의", "user_keywords": ["티셔츠"], "inferred_tags": ["캐주얼"], "keyword": "티셔츠"}}
    - "운동화 추천해줘" → {{"category": "신발", "user_keywords": ["운동화"], "inferred_tags": ["편안한"], "keyword": "운동화"}}
    - "캐주얼한 상의 추천해줘" → {{"category": "상의", "user_keywords": ["캐주얼", "상의"], "inferred_tags": ["편안한"], "keyword": "캐주얼 상의"}}
    - "버뮤다 팬츠인데 데님 소재가 들어간" → {{"category": "바지", "user_keywords": ["버뮤다", "데님"], "inferred_tags": ["여름", "시원한"], "keyword": "버뮤다 팬츠 데님 소재"}}
    - "티셔츠인데 검은색이면 좋겠어" → {{"category": "상의", "user_keywords": ["티셔츠", "검은색"], "inferred_tags": ["캐주얼"], "keyword": "티셔츠 검은색"}}
    
    JSON만 답변해주세요.
    """
    
    try:
        response = llm.invoke([{"role": "user", "content": filter_prompt}])
        filters = json.loads(response.content)
        print(f"LLM 카테고리 분류 결과: {filters.get('category', 'null')}")
        
        # 동적으로 찾은 카테고리가 있으면 우선 적용
        if best_category and score > 0:
            original_category = filters.get('category')
            filters['category'] = best_category
            print(f"동적 카테고리 우선 적용: LLM '{original_category}' → 동적 '{best_category}' (점수: {score})")
        
        # 유추된 키워드들을 inferred_tags에 추가
        if seasonal_keywords:
            current_inferred_tags = filters.get('inferred_tags', [])
            filters['inferred_tags'] = list(set(current_inferred_tags + seasonal_keywords))
            print(f"유추된 키워드를 inferred_tags에 추가: {filters['inferred_tags']}")
        
        # 기존 tags 필드를 inferred_tags로 통합 (하위 호환성)
        if 'tags' not in filters:
            filters['tags'] = filters.get('inferred_tags', [])
        
        # 가격 정보 추가
        if price_info:
            filters.update(price_info)
            print(f"가격 정보 추가: {price_info}")
        
        # 이전 조건과 현재 조건을 결합 (이전 조건이 있을 때만)
        if previous_conditions.get('keywords'):
            # 현재 키워드가 구체적인 상품명이나 브랜드명을 포함하면 이전 키워드 누적을 제한
            current_keyword = filters.get('keyword', '')
            current_input_lower = user_input.lower()
            
            # 구체적인 상품명이나 브랜드가 있으면 이전 키워드 누적을 최소화
            if any(specific in current_input_lower for specific in ['버뮤다', '팬츠', '바지', '상의', '신발', '가방']):
                # 현재 키워드를 우선하고, 이전 키워드는 최소한만 추가
                if current_keyword:
                    filters['keyword'] = current_keyword
                    print(f"구체적인 상품명이 있어서 현재 키워드 우선: '{current_keyword}'")
            else:
                # 일반적인 경우에만 이전 키워드 누적
                previous_keywords = previous_conditions['keywords'][:5]  # 상위 5개만
                current_tags = filters.get('tags', [])
                filters['tags'] = list(set(current_tags + previous_keywords))
                print(f"이전 조건 태그 누적: {filters['tags']}")
                
                if current_keyword:
                    filters['keyword'] = f"{current_keyword} {' '.join(previous_keywords[:2])}".strip()
                    print(f"조건 누적: 이전 조건 + 현재 조건 = '{filters['keyword']}'")
        
        print(f"=== extract_filters 최종 결과: {filters} ===\n")
        return filters
    except Exception as e:
        print(f"필터 추출 오류: {e}")
        return {"category": best_category, "brand": None, "tags": seasonal_keywords, "keyword": None, "max_price": None, "min_price": None} 