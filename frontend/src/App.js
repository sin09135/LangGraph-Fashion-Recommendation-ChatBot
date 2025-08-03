import React, { useState } from 'react';
import styled from 'styled-components';
import ChatInterface from './components/ChatInterface';
import ProductGrid from './components/ProductGrid';
import LikedProductsView from './components/LikedProductsView';
import Header from './components/Header';
import { MessageCircle, ShoppingBag, RefreshCw } from 'lucide-react';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  position: relative;
  
  &::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
      radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }
`;

const MainContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 32px 24px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  min-height: calc(100vh - 120px);
  position: relative;
  z-index: 1;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
    gap: 24px;
    padding: 24px 20px;
  }
  
  @media (max-width: 768px) {
    padding: 20px 16px;
    gap: 20px;
  }
`;

const ChatSection = styled.div`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 24px;
  box-shadow: 
    0 20px 40px rgba(0, 0, 0, 0.1),
    0 8px 16px rgba(0, 0, 0, 0.05);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  border: 1px solid rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
  height: calc(100vh - 200px);
  min-height: 500px;
  max-height: 700px;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 
      0 25px 50px rgba(0, 0, 0, 0.12),
      0 12px 24px rgba(0, 0, 0, 0.06);
  }
  
  @media (max-width: 1024px) {
    height: 500px;
    min-height: 400px;
    max-height: 600px;
  }
`;

const ProductSection = styled.div`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 24px;
  box-shadow: 
    0 20px 40px rgba(0, 0, 0, 0.1),
    0 8px 16px rgba(0, 0, 0, 0.05);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  border: 1px solid rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
  height: calc(100vh - 200px);
  min-height: 500px;
  max-height: 700px;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 
      0 25px 50px rgba(0, 0, 0, 0.12),
      0 12px 24px rgba(0, 0, 0, 0.06);
  }
  
  @media (max-width: 1024px) {
    height: auto;
    min-height: 400px;
    max-height: none;
  }
`;

const SectionHeader = styled.div`
  padding: 24px 28px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 700;
  color: #1a1a1a;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  font-size: 18px;
  
  @media (max-width: 768px) {
    padding: 20px 24px;
    font-size: 16px;
  }
`;

const ResetButton = styled.button`
  position: fixed;
  bottom: 32px;
  right: 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 16px 24px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  font-size: 14px;
  box-shadow: 
    0 8px 24px rgba(102, 126, 234, 0.3),
    0 4px 12px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  z-index: 1000;
  backdrop-filter: blur(10px);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 
      0 12px 32px rgba(102, 126, 234, 0.4),
      0 6px 16px rgba(0, 0, 0, 0.15);
  }
  
  &:active {
    transform: translateY(0);
  }
  
  @media (max-width: 768px) {
    bottom: 24px;
    right: 24px;
    padding: 12px 20px;
    font-size: 13px;
    border-radius: 14px;
  }
`;

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [products, setProducts] = useState([]);
  const [recommendedProducts, setRecommendedProducts] = useState([]); // 추천 상품 목록 별도 저장
  const [isLoading, setIsLoading] = useState(false);
  const [showLikedProducts, setShowLikedProducts] = useState(false);
  const [likedProducts, setLikedProducts] = useState([]);

  const resetSession = async () => {
    if (sessionId) {
      try {
        await fetch(`http://localhost:8001/session/${sessionId}`, {
          method: 'DELETE'
        });
      } catch (error) {
        console.error('세션 초기화 오류:', error);
      }
    }
    setSessionId(null);
    setProducts([]);
  };

  // 좋아요 목록 변경 핸들러
  const handleLikedProductsChange = (newLikedProducts) => {
    setLikedProducts(newLikedProducts);
  };

  // 좋아요 취소 핸들러
  const handleUnlike = async (product) => {
    try {
      const response = await fetch('http://localhost:8001/like', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          product_id: product.product_id,
          action: 'unlike'
        })
      });

      if (response.ok) {
        const data = await response.json();
        setLikedProducts(data.liked_products);
        console.log('좋아요 취소:', product.product_name);
      } else {
        console.error('좋아요 취소 실패');
      }
    } catch (error) {
      console.error('좋아요 취소 오류:', error);
    }
  };

  // 좋아요 목록 토글
  const toggleLikedProducts = () => {
    if (showLikedProducts) {
      setShowLikedProducts(false);
      setProducts(recommendedProducts); // 저장된 추천 상품 목록으로 복원
    } else {
      setShowLikedProducts(true);
    }
  };

  return (
    <AppContainer>
      <Header />
      <MainContent>
        <ChatSection>
          <SectionHeader>
            <MessageCircle size={20} />
            AI 패션 어시스턴트
          </SectionHeader>
          <ChatInterface 
            onProductsReceived={(newProducts) => {
              setProducts(newProducts);
              setRecommendedProducts(newProducts); // 추천 상품 목록도 저장
            }}
            sessionId={sessionId}
            setSessionId={setSessionId}
            setIsLoading={setIsLoading}
          />
        </ChatSection>
        
        <ProductSection>
          <SectionHeader>
            <ShoppingBag size={20} />
            {showLikedProducts ? '좋아요 목록' : '추천 상품'}
            {!showLikedProducts && (
              <button 
                onClick={toggleLikedProducts}
                style={{
                  marginLeft: 'auto',
                  background: '#1a1a1a',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  fontSize: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => e.target.style.background = '#2d2d2d'}
                onMouseOut={(e) => e.target.style.background = '#1a1a1a'}
              >
                ❤️ 좋아요 목록
              </button>
            )}
            {showLikedProducts && (
              <button 
                onClick={toggleLikedProducts}
                style={{
                  marginLeft: 'auto',
                  background: '#666666',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  fontSize: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => e.target.style.background = '#888888'}
                onMouseOut={(e) => e.target.style.background = '#666666'}
              >
                ← 추천 상품으로
              </button>
            )}
          </SectionHeader>
          {showLikedProducts ? (
            <LikedProductsView 
              likedProducts={likedProducts}
              onUnlike={handleUnlike}
              sessionId={sessionId}
            />
          ) : (
            <ProductGrid 
              products={products} 
              isLoading={isLoading}
              sessionId={sessionId}
              onLikedProductsChange={handleLikedProductsChange}
            />
          )}
        </ProductSection>
      </MainContent>
      
      <ResetButton onClick={resetSession}>
        <RefreshCw size={16} />
        대화 초기화
      </ResetButton>
    </AppContainer>
  );
}

export default App; 