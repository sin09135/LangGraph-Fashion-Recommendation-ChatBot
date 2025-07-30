import React, { useState } from 'react';
import styled from 'styled-components';
import ChatInterface from './components/ChatInterface';
import ProductGrid from './components/ProductGrid';
import Header from './components/Header';
import { MessageCircle, ShoppingBag, RefreshCw } from 'lucide-react';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: 'Inter', sans-serif;
`;

const MainContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
  height: calc(100vh - 100px);
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    height: auto;
    padding: 15px;
    gap: 20px;
  }
`;

const ChatSection = styled.div`
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const ProductSection = styled.div`
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const SectionHeader = styled.div`
  padding: 20px;
  border-bottom: 1px solid #f0f0f0;
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  color: #333;
`;

const ResetButton = styled.button`
  position: fixed;
  bottom: 30px;
  right: 30px;
  background: #ff6b6b;
  color: white;
  border: none;
  border-radius: 50px;
  padding: 15px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
  transition: all 0.3s ease;
  z-index: 1000;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(255, 107, 107, 0.4);
  }
  
  @media (max-width: 768px) {
    bottom: 20px;
    right: 20px;
    padding: 12px 16px;
    font-size: 14px;
    
    &:hover {
      transform: none;
    }
  }
`;

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [products, setProducts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const resetSession = async () => {
    if (sessionId) {
      try {
        await fetch(`http://localhost:8000/session/${sessionId}`, {
          method: 'DELETE'
        });
      } catch (error) {
        console.error('세션 초기화 오류:', error);
      }
    }
    setSessionId(null);
    setProducts([]);
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
            onProductsReceived={setProducts}
            sessionId={sessionId}
            setSessionId={setSessionId}
            setIsLoading={setIsLoading}
          />
        </ChatSection>
        
        <ProductSection>
          <SectionHeader>
            <ShoppingBag size={20} />
            추천 상품
          </SectionHeader>
          <ProductGrid 
            products={products} 
            isLoading={isLoading}
          />
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