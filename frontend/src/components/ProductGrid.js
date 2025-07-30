import React from 'react';
import styled from 'styled-components';
import { ShoppingBag, Heart, ExternalLink } from 'lucide-react';

const ProductContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  
  /* 스크롤바 스타일링 */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
  
  @media (max-width: 768px) {
    padding: 15px;
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
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  transition: all 0.3s ease;
  border: 1px solid #f0f0f0;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
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
  background: ${props => props.imageUrl ? `url(${props.imageUrl})` : '#f8f9fa'};
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.1);
  }
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
  color: #333;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
`;

const ProductBrand = styled.div`
  color: #667eea;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 8px;
`;

const ProductPrice = styled.div`
  font-size: 18px;
  font-weight: 700;
  color: #333;
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
  
  &.primary {
    background: #667eea;
    color: white;
    
    &:hover {
      background: #5a6fd8;
    }
  }
  
  &.secondary {
    background: #f8f9fa;
    color: #666;
    border: 1px solid #e0e0e0;
    
    &:hover {
      background: #e9ecef;
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

function ProductGrid({ products, isLoading }) {
  const formatPrice = (price) => {
    return new Intl.NumberFormat('ko-KR').format(price) + '원';
  };

  const handleProductClick = (product) => {
    // 상품 링크로 이동 (실제 구현 시)
    console.log('상품 클릭:', product);
  };

  const handleLikeClick = (product) => {
    // 좋아요 기능 (실제 구현 시)
    console.log('좋아요 클릭:', product);
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

  return (
    <ProductContainer>
      <ProductGridContainer>
        {products.map((product) => (
          <ProductCard key={product.product_id}>
            <ProductImage imageUrl={product.image_url}>
              {!product.image_url && (
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
                  className="primary"
                  onClick={() => handleProductClick(product)}
                >
                  <ExternalLink size={14} />
                  상품 보기
                </ActionButton>
                <ActionButton 
                  className="secondary"
                  onClick={() => handleLikeClick(product)}
                >
                  <Heart size={14} />
                  좋아요
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