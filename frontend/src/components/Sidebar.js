import React from 'react';
import styled from 'styled-components';
import { 
  MessageCircle, 
  ShoppingBag, 
  Heart, 
  Search, 
  Image, 
  Star, 
  Palette,
  TrendingUp,
  Settings as SettingsIcon,
  Home
} from 'lucide-react';

const SidebarContainer = styled.div`
  width: 280px;
  height: 100vh;
  background: #ffffff;
  border-right: 1px solid #e2e8f0;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 100;
  overflow-y: auto;
  box-shadow: 1px 0 3px rgba(0, 0, 0, 0.05);
`;

const SidebarHeader = styled.div`
  padding: 28px 24px;
  border-bottom: 1px solid #f1f5f9;
  background: #ffffff;
  
  h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: #1e293b;
  }
  
  p {
    margin: 6px 0 0 0;
    font-size: 13px;
    color: #64748b;
  }
`;

const MenuSection = styled.div`
  padding: 20px 0;
  
  &:not(:last-child) {
    border-bottom: 1px solid #f8fafc;
  }
`;

const SectionTitle = styled.div`
  padding: 0 24px 16px;
  font-size: 11px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.8px;
`;

const MenuItem = styled.div`
  display: flex;
  align-items: center;
  padding: 14px 24px;
  cursor: pointer;
  transition: all 0.2s ease;
  color: ${props => props.active ? '#6366f1' : '#475569'};
  background: ${props => props.active ? '#f8fafc' : 'transparent'};
  border-left: 3px solid ${props => props.active ? '#6366f1' : 'transparent'};
  
  &:hover {
    background: ${props => props.active ? '#f1f5f9' : '#f8fafc'};
  }
  
  svg {
    width: 20px;
    height: 20px;
    margin-right: 14px;
    color: ${props => props.active ? '#6366f1' : '#64748b'};
  }
  
  span {
    font-size: 15px;
    font-weight: ${props => props.active ? '600' : '500'};
  }
`;

const MenuItemDescription = styled.div`
  font-size: 12px;
  color: #94a3b8;
  margin-top: 3px;
  margin-left: 34px;
  line-height: 1.4;
`;

const Sidebar = ({ activeFeature, onFeatureChange }) => {
  const features = [
    {
      id: 'main',
      title: '메인 페이지',
      icon: Home,
      description: 'AI 패션 어시스턴트 홈',
      section: 'main'
    },
    {
      id: 'chat',
      title: '패션 추천 챗봇',
      icon: MessageCircle,
      description: '자연어 대화 기반 상품 추천',
      section: 'chat'
    },
    {
      id: 'coordination',
      title: '코디네이션 추천',
      icon: Palette,
      description: '상품 조합 및 스타일링 추천',
      section: 'recommendation'
    },
    {
      id: 'similar',
      title: '유사 상품 검색',
      icon: Search,
      description: '유사한 스타일 상품 찾기',
      section: 'recommendation'
    },
    {
      id: 'reviews',
      title: '리뷰 분석',
      icon: Star,
      description: '상품 리뷰 및 평점 분석',
      section: 'analysis'
    },
    {
      id: 'image',
      title: '이미지 검색',
      icon: Image,
      description: '이미지 기반 상품 검색',
      section: 'analysis'
    },
    {
      id: 'trends',
      title: '트렌드 분석',
      icon: TrendingUp,
      description: '패션 트렌드 및 인기 상품',
      section: 'analysis'
    },
    {
      id: 'liked',
      title: '좋아요 목록',
      icon: Heart,
      description: '저장한 상품 관리',
      section: 'management'
    },
    {
      id: 'products',
      title: '상품 브라우저',
      icon: ShoppingBag,
      description: '전체 상품 카탈로그',
      section: 'management'
    },
    {
      id: 'settings',
      title: '설정',
      icon: SettingsIcon,
      description: '개인화 설정 및 옵션',
      section: 'settings'
    }
  ];

  const sections = {
    main: '메인',
    chat: 'AI 챗봇',
    recommendation: '추천 시스템',
    analysis: '분석 도구',
    management: '관리',
    settings: '설정'
  };

  const groupedFeatures = features.reduce((acc, feature) => {
    if (!acc[feature.section]) {
      acc[feature.section] = [];
    }
    acc[feature.section].push(feature);
    return acc;
  }, {});

  return (
    <SidebarContainer>
      <SidebarHeader>
        <h2>AI 패션 어시스턴트</h2>
        <p>Beta Version</p>
      </SidebarHeader>
      
      {Object.entries(groupedFeatures).map(([sectionKey, sectionFeatures]) => (
        <MenuSection key={sectionKey}>
          <SectionTitle>{sections[sectionKey]}</SectionTitle>
          {sectionFeatures.map((feature) => {
            const IconComponent = feature.icon;
            return (
              <MenuItem
                key={feature.id}
                active={activeFeature === feature.id}
                onClick={() => onFeatureChange(feature.id)}
              >
                <IconComponent />
                <div>
                  <span>{feature.title}</span>
                  <MenuItemDescription>{feature.description}</MenuItemDescription>
                </div>
              </MenuItem>
            );
          })}
        </MenuSection>
      ))}
    </SidebarContainer>
  );
};

export default Sidebar; 