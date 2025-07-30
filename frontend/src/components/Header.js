import React from 'react';
import styled from 'styled-components';
import { Sparkles } from 'lucide-react';

const HeaderContainer = styled.header`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  padding: 20px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
`;

const HeaderContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  
  @media (max-width: 768px) {
    flex-direction: column;
    gap: 8px;
    padding: 0 15px;
  }
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  color: white;
  font-size: 24px;
  font-weight: 700;
  
  @media (max-width: 768px) {
    font-size: 20px;
    gap: 8px;
  }
`;

const Subtitle = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 16px;
  font-weight: 400;
  
  @media (max-width: 768px) {
    font-size: 14px;
  }
`;

function Header() {
  return (
    <HeaderContainer>
      <HeaderContent>
        <Logo>
          <Sparkles size={32} />
          패션 추천 시스템
        </Logo>
        <Subtitle>
          AI 기반 개인화 패션 추천
        </Subtitle>
      </HeaderContent>
    </HeaderContainer>
  );
}

export default Header; 