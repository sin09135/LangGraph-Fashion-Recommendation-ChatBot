import React from 'react';
import styled from 'styled-components';
import { Sparkles, MessageCircle, ShoppingBag } from 'lucide-react';

const HeaderContainer = styled.header`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 24px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
    opacity: 0.3;
  }
`;

const HeaderContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: relative;
  z-index: 1;
  
  @media (max-width: 768px) {
    flex-direction: column;
    gap: 12px;
    padding: 0 16px;
  }
`;

const LogoSection = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  color: white;
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -0.5px;
  
  @media (max-width: 768px) {
    font-size: 24px;
    gap: 10px;
  }
`;

const LogoIcon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.3);
  
  @media (max-width: 768px) {
    width: 40px;
    height: 40px;
  }
`;

const Subtitle = styled.div`
  color: rgba(255, 255, 255, 0.9);
  font-size: 16px;
  font-weight: 500;
  margin-top: 4px;
  
  @media (max-width: 768px) {
    font-size: 14px;
    text-align: center;
  }
`;

const Features = styled.div`
  display: flex;
  align-items: center;
  gap: 24px;
  
  @media (max-width: 768px) {
    gap: 16px;
  }
`;

const Feature = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  font-weight: 500;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  
  @media (max-width: 768px) {
    font-size: 12px;
    padding: 6px 12px;
  }
`;

function Header() {
  return (
    <HeaderContainer>
      <HeaderContent>
        <LogoSection>
          <Logo>
            <LogoIcon>
              <Sparkles size={24} />
            </LogoIcon>
            패션 추천 챗봇
          </Logo>
          <Subtitle>
            AI 기반 개인화 패션 추천 시스템
          </Subtitle>
        </LogoSection>
        
        <Features>
          <Feature>
            <MessageCircle size={16} />
            실시간 채팅
          </Feature>
          <Feature>
            <ShoppingBag size={16} />
            스마트 추천
          </Feature>
        </Features>
      </HeaderContent>
    </HeaderContainer>
  );
}

export default Header; 