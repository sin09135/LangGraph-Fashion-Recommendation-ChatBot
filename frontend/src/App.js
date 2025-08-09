import React, { useState } from 'react';
import styled from 'styled-components';
import Sidebar from './components/Sidebar';
import {
  MainPage,
  CoordinationRecommendation,
  SimilarProductSearch,
  ReviewAnalysis,
  ImageSearch,
  TrendAnalysis,
  LikedProducts,
  ProductBrowser,
  Settings
} from './components/FeatureComponents';
import ChatInterface from './components/ChatInterface';

const AppContainer = styled.div`
  display: flex;
  height: 100vh;
  background: #f8fafc;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
`;

const MainContent = styled.div`
  flex: 1;
  margin-left: 280px;
  background: #f8fafc;
  overflow: hidden;
`;

const ChatSection = styled.div`
  background: #ffffff;
  border-radius: 12px;
  margin: 24px;
  padding: 32px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border: 1px solid #f1f5f9;
  height: calc(100vh - 48px);
  display: flex;
  flex-direction: column;
`;

const SectionHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid #f1f5f9;
  
  h2 {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: #1e293b;
  }
  
  p {
    margin: 4px 0 0 0;
    color: #64748b;
    font-size: 16px;
  }
`;

const ResetButton = styled.button`
  padding: 12px 20px;
  background: #6366f1;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #5855eb;
    transform: translateY(-1px);
  }
`;

const ProductSection = styled.div`
  background: #ffffff;
  border-radius: 12px;
  margin: 24px;
  padding: 32px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border: 1px solid #f1f5f9;
  height: calc(100vh - 48px);
  overflow-y: auto;
`;

function App() {
  const [activeFeature, setActiveFeature] = useState('main');
  const [likedProducts, setLikedProducts] = useState([]);
  const [chatResetKey, setChatResetKey] = useState(0);

  const handleUnlike = (product) => {
    setLikedProducts(prev => prev.filter(p => p.product_id !== product.product_id));
  };

  const handleChatReset = () => {
    setChatResetKey(prev => prev + 1);
  };

  const renderActiveFeature = () => {
    switch (activeFeature) {
      case 'main':
        return <MainPage />;
      case 'chat':
        return (
          <ChatSection>
            <SectionHeader>
              <div>
                <h2>패션 추천 챗봇</h2>
                <p>자연어 대화를 통해 개인화된 패션 추천을 받아보세요</p>
              </div>
              <ResetButton onClick={handleChatReset}>대화 초기화</ResetButton>
            </SectionHeader>
            <ChatInterface 
              key={chatResetKey}
              likedProducts={likedProducts}
              setLikedProducts={setLikedProducts}
            />
          </ChatSection>
        );
      case 'coordination':
        return <CoordinationRecommendation />;
      case 'similar':
        return <SimilarProductSearch />;
      case 'reviews':
        return <ReviewAnalysis />;
      case 'image':
        return <ImageSearch />;
      case 'trends':
        return <TrendAnalysis />;
      case 'liked':
        return <LikedProducts likedProducts={likedProducts} onUnlike={handleUnlike} />;
      case 'products':
        return <ProductBrowser />;
      case 'settings':
        return <Settings />;
      default:
        return <MainPage />;
    }
  };

  return (
    <AppContainer>
      <Sidebar activeFeature={activeFeature} onFeatureChange={setActiveFeature} />
      <MainContent>
        {renderActiveFeature()}
      </MainContent>
    </AppContainer>
  );
}

export default App; 