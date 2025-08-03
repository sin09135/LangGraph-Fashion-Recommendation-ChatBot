import React from 'react';
import styled from 'styled-components';
import { Heart, ExternalLink } from 'lucide-react';

const LikedContainer = styled.div`
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

const LikedHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
  padding: 16px;
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  border-radius: 12px;
  color: white;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
`;

const LikedCount = styled.span`
  font-weight: 600;
  font-size: 18px;
`;

const LikedGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 15px;
  }
`;

const LikedCard = styled.div`
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  overflow: hidden;
  transition: all 0.2s ease;
  border: 2px solid #2563eb;
  position: relative;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.15);
  }
  
  @media (max-width: 768px) {
    &:hover {
      transform: none;
    }
  }
`;

const LikedBadge = styled.div`
  position: absolute;
  top: 10px;
  right: 10px;
  background: #2563eb;
  color: white;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 4px;
  z-index: 2;
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
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
  color: #333;
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
  
  &.primary {
    background: #1a1a1a;
    color: white;
    
    &:hover {
      background: #2d2d2d;
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
  padding: 40px 20px;
`;

function LikedProductsView({ likedProducts, onUnlike, sessionId }) {
  const formatPrice = (price) => {
    return new Intl.NumberFormat('ko-KR').format(price) + '원';
  };

  const handleUnlike = async (product) => {
    if (onUnlike) {
      onUnlike(product);
    }
  };

  if (!likedProducts || likedProducts.length === 0) {
    return (
      <LikedContainer>
        <LikedHeader>
          <Heart size={20} />
          <LikedCount>좋아요한 상품</LikedCount>
        </LikedHeader>
        <EmptyState>
          <Heart size={48} color="#ff6b6b" />
          <h3>좋아요한 상품이 없습니다</h3>
          <p>상품을 추천받고 마음에 드는 상품에 좋아요를 눌러보세요!</p>
        </EmptyState>
      </LikedContainer>
    );
  }

  return (
    <LikedContainer>
      <LikedHeader>
        <Heart size={20} />
        <LikedCount>좋아요한 상품 ({likedProducts.length}개)</LikedCount>
      </LikedHeader>
      
      <LikedGrid>
        {likedProducts.map((product) => (
          <LikedCard key={product.product_id}>
            <LikedBadge>
              <Heart size={12} fill="white" />
              좋아요
            </LikedBadge>
            
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
                </div>
              )}
              
              <ProductActions>
                <ActionButton 
                  as="a"
                  href={product.product_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="primary"
                  title="새 탭에서 상품 페이지 열기"
                >
                  <ExternalLink size={14} />
                  상품 보기
                </ActionButton>
                <ActionButton 
                  onClick={() => handleUnlike(product)}
                  style={{
                    background: '#1a1a1a',
                    color: 'white',
                    border: '1px solid #1a1a1a'
                  }}
                >
                  <Heart size={14} fill="white" />
                  좋아요 취소
                </ActionButton>
              </ProductActions>
            </ProductInfo>
          </LikedCard>
        ))}
      </LikedGrid>
    </LikedContainer>
  );
}

export default LikedProductsView; 