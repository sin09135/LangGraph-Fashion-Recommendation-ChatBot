import React from 'react';
import styled from 'styled-components';
import { ShoppingBag, Heart, ExternalLink } from 'lucide-react';

const ProductContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  height: 100%;
  
  /* 스크롤바 스타일링 */
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.3);
    border-radius: 4px;
    transition: background 0.3s ease;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.5);
  }
  
  @media (max-width: 768px) {
    padding: 20px;
  }
`;

const ProductGridContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 15px;
  }
`;

const ProductCard = styled.div`
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  overflow: hidden;
  transition: all 0.2s ease;
  border: 1px solid #f0f0f0;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    border-color: #e5e7eb;
  }
  
  @media (max-width: 768px) {
    &:hover {
      transform: none;
    }
  }
`;

const ProductImage = styled.div`
  width: 100%;
  height: 200px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: #f8f9fa;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.1);
    z-index: 1;
  }
`;

const ProductImg = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
`;

const ImagePlaceholder = styled.div`
  color: #999;
  font-size: 14px;
  text-align: center;
  z-index: 1;
`;

const ProductInfo = styled.div`
  padding: 16px;
`;

const ProductName = styled.h3`
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #1a1a1a;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
`;

const ProductBrand = styled.div`
  color: #666666;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 8px;
`;

const ProductPrice = styled.div`
  font-size: 18px;
  font-weight: 700;
  color: #1a1a1a;
  margin-bottom: 12px;
`;

const ProductTags = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
`;

const Tag = styled.span`
  background: #f0f0f0;
  color: #666;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
`;

const ProductActions = styled.div`
  display: flex;
  gap: 8px;
`;

const ActionButton = styled.button`
  flex: 1;
  padding: 8px 12px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  transition: all 0.3s ease;
  text-decoration: none;
  
  /* a 태그로 사용될 때의 스타일 */
  &.primary {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: 600;
    
    &:hover {
      background: #1d4ed8;
      color: white;
      text-decoration: none;
    }
  }
  
  &.secondary {
    background: #ffffff;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    font-weight: 500;
    
    &:hover {
      background: #f9fafb;
      color: #374151;
      text-decoration: none;
      border-color: #9ca3af;
    }
  }
  
  &.liked {
    background: #2563eb;
    color: white;
    border: 1px solid #2563eb;
    border-radius: 8px;
    font-weight: 600;
    
    &:hover {
      background: #1d4ed8;
      color: white;
      text-decoration: none;
    }
  }
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #999;
  text-align: center;
`;

const LoadingState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #667eea;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 4px solid #f0f0f0;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

function ProductGrid({ products, isLoading, sessionId, onLikedProductsChange }) {
  const [likedProducts, setLikedProducts] = React.useState(new Set());
  const [likedProductsData, setLikedProductsData] = React.useState([]);

  // 컴포넌트 마운트 시 서버에서 좋아요 목록 로드
  React.useEffect(() => {
    if (sessionId) {
      console.log('ProductGrid - sessionId 변경됨:', sessionId);
      loadLikedProducts();
    }
  }, [sessionId]);

  // products가 변경될 때마다 좋아요 상태 다시 로드
  React.useEffect(() => {
    if (sessionId && products.length > 0) {
      console.log('ProductGrid - products 변경됨, 좋아요 상태 다시 로드');
      loadLikedProducts();
    }
  }, [products, sessionId]);

  // 좋아요 목록 로드
  const loadLikedProducts = async () => {
    try {
      console.log('좋아요 목록 로드 시작 - sessionId:', sessionId);
      const response = await fetch(`http://localhost:8001/likes/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        console.log('좋아요 목록 로드 성공:', data);
        const likedIds = new Set(data.liked_products.map(p => p.product_id.toString()));
        setLikedProducts(likedIds);
        setLikedProductsData(data.liked_products);
        if (onLikedProductsChange) {
          onLikedProductsChange(data.liked_products);
        }
        console.log('좋아요 상태 설정 완료:', likedIds);
      } else {
        console.error('좋아요 목록 로드 실패:', response.status);
      }
    } catch (error) {
      console.error('좋아요 목록 로드 오류:', error);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('ko-KR').format(price) + '원';
  };

  const handleProductClick = (product) => {
    // 상품 링크로 이동
    console.log('상품보기 클릭:', product);
    console.log('상품 URL 확인:', product.product_url);
    
    if (product.product_url) {
      console.log('URL 열기 시도:', product.product_url);
      
      // 가장 간단하고 확실한 방법
      try {
        // 직접 window.open 시도
        const newWindow = window.open(product.product_url, '_blank');
        
        // 팝업이 차단되었는지 확인
        if (!newWindow || newWindow.closed || typeof newWindow.closed === 'undefined') {
          console.log('팝업이 차단됨, 대체 방법 시도');
          
          // 대체 방법: 현재 탭에서 열기
          window.location.href = product.product_url;
        } else {
          console.log('새 탭에서 열기 성공');
        }
      } catch (error) {
        console.error('URL 열기 오류:', error);
        
        // 최후의 수단: 사용자에게 URL 복사 안내
        const url = product.product_url;
        if (navigator.clipboard) {
          navigator.clipboard.writeText(url).then(() => {
            alert('링크가 복사되었습니다. 브라우저 주소창에 붙여넣기 해주세요:\n\n' + url);
          });
        } else {
          alert('브라우저에서 링크를 열 수 없습니다. 수동으로 복사하여 주소창에 붙여넣기 해주세요:\n\n' + url);
        }
      }
    } else {
      console.log('상품 URL이 없습니다:', product);
      alert('상품 URL을 찾을 수 없습니다.');
    }
  };

  const handleLikeClick = async (product) => {
    const productId = product.product_id.toString();
    const isCurrentlyLiked = likedProducts.has(productId);
    
    console.log('좋아요 클릭:', product.product_name, '현재 상태:', isCurrentlyLiked);
    
    try {
      const response = await fetch('http://localhost:8001/like', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          product_id: product.product_id,
          action: isCurrentlyLiked ? 'unlike' : 'like'
        })
      });

      if (response.ok) {
        const data = await response.json();
        console.log('서버 응답:', data);
        
        // 서버에서 받은 좋아요 목록으로 상태 업데이트
        const newLikedIds = new Set(data.liked_products.map(p => p.product_id.toString()));
        setLikedProducts(newLikedIds);
        setLikedProductsData(data.liked_products);
        
        // 부모 컴포넌트에 좋아요 목록 변경 알림
        if (onLikedProductsChange) {
          onLikedProductsChange(data.liked_products);
        }
        
        console.log('좋아요 상태 업데이트 완료:', isCurrentlyLiked ? '취소' : '추가');
      } else {
        console.error('좋아요 처리 실패:', response.status);
      }
    } catch (error) {
      console.error('좋아요 처리 오류:', error);
    }
  };

  if (isLoading) {
    return (
      <ProductContainer>
        <LoadingState>
          <LoadingSpinner />
          <div>추천 상품을 찾고 있습니다...</div>
        </LoadingState>
      </ProductContainer>
    );
  }

  if (!products || products.length === 0) {
    return (
      <ProductContainer>
        <EmptyState>
          <ShoppingBag size={48} />
          <h3>추천 상품이 없습니다</h3>
          <p>AI 어시스턴트와 대화하여 맞춤형 상품을 추천받아보세요!</p>
          <p style={{ fontSize: '14px', color: '#999', marginTop: '10px' }}>
            예시: "버뮤다 팬츠 4만원 미만으로 추천해줘"
          </p>
        </EmptyState>
      </ProductContainer>
    );
  }

  console.log('ProductGrid 렌더링 - 상품 개수:', products.length);
  console.log('첫 번째 상품 데이터:', products[0]);
  
  return (
    <ProductContainer>
      <ProductGridContainer>
        {products.map((product) => (
          <ProductCard key={product.product_id}>
            <ProductImage>
              {product.image_url ? (
                <ProductImg src={product.image_url} alt={product.product_name} />
              ) : (
                <ImagePlaceholder>
                  이미지 없음
                </ImagePlaceholder>
              )}
            </ProductImage>
            
            <ProductInfo>
              <ProductName>{product.product_name}</ProductName>
              <ProductBrand>{product.brand_kr}</ProductBrand>
              <ProductPrice>{formatPrice(product.price)}</ProductPrice>
              
              {product.category && (
                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                  카테고리: {product.category}
                  {product.coordination_score && (
                    <span style={{ color: '#667eea', marginLeft: '8px' }}>
                      코디 점수: {(product.coordination_score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              )}
              
              {product.tags && product.tags.length > 0 && (
                <ProductTags>
                  {product.tags.slice(0, 3).map((tag, index) => (
                    <Tag key={index}>{tag}</Tag>
                  ))}
                  {product.tags.length > 3 && (
                    <Tag>+{product.tags.length - 3}</Tag>
                  )}
                </ProductTags>
              )}
              
              <ProductActions>
                <ActionButton 
                  as="a"
                  href={product.product_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="primary"
                  title="새 탭에서 상품 페이지 열기"
                  onClick={(e) => {
                    if (!product.product_url) {
                      e.preventDefault();
                      alert('상품 URL을 찾을 수 없습니다.');
                    }
                  }}
                >
                  <ExternalLink size={14} />
                  상품 보기
                </ActionButton>
                <ActionButton 
                  className={likedProducts.has(product.product_id.toString()) ? "liked" : "secondary"}
                  onClick={() => handleLikeClick(product)}
                >
                  <Heart size={14} fill={likedProducts.has(product.product_id.toString()) ? "white" : "none"} />
                  {likedProducts.has(product.product_id.toString()) ? "좋아요 취소" : "좋아요"}
                </ActionButton>
              </ProductActions>
            </ProductInfo>
          </ProductCard>
        ))}
      </ProductGridContainer>
    </ProductContainer>
  );
}

export default ProductGrid; 