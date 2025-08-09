import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { Send, User, Bot, Heart, ExternalLink } from 'lucide-react';

const ChatContainer = styled.div`
  display: flex;
  height: 100%;
  background: #ffffff;
  gap: 20px;
`;

const ChatSection = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
`;

const RecommendationsSection = styled.div`
  width: 300px;
  background: #f8fafc;
  border-left: 1px solid #e2e8f0;
  padding: 20px;
  overflow-y: auto;
  
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
`;

const RecommendationsHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
`;

const RecommendationsTitle = styled.h3`
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1e293b;
`;

const ViewLikedButton = styled.button`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: white;
  color: #6366f1;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #f8fafc;
    border-color: #6366f1;
  }
  
  svg {
    width: 14px;
    height: 14px;
  }
`;

const ProductCard = styled.div`
  background: white;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
  transition: all 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
`;

const ProductImage = styled.img`
  width: 100%;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 12px;
`;

const ProductName = styled.h4`
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 600;
  color: #1e293b;
  line-height: 1.3;
`;

const ProductPrice = styled.div`
  font-size: 16px;
  font-weight: 700;
  color: #6366f1;
  margin-bottom: 4px;
`;

const ProductDescription = styled.p`
  margin: 0;
  font-size: 12px;
  color: #64748b;
  line-height: 1.4;
`;

const NoRecommendations = styled.div`
  text-align: center;
  color: #94a3b8;
  font-size: 14px;
  padding: 40px 20px;
`;

const ProductActions = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 12px;
  gap: 8px;
`;

const ActionButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
  border: none;
  border-radius: 8px;
  background: ${props => props.isLiked ? '#ef4444' : '#f1f5f9'};
  color: ${props => props.isLiked ? 'white' : '#64748b'};
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 12px;
  gap: 4px;
  
  &:hover {
    background: ${props => props.isLiked ? '#dc2626' : '#e2e8f0'};
    transform: translateY(-1px);
  }
  
  svg {
    width: 14px;
    height: 14px;
  }
`;

const LikeButton = styled(ActionButton)`
  flex: 1;
`;

const LinkButton = styled(ActionButton)`
  background: #6366f1;
  color: white;
  
  &:hover {
    background: #5855eb;
  }
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 0;
  margin-bottom: 20px;
  
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
`;

const Message = styled.div`
  display: flex;
  margin-bottom: 20px;
  align-items: flex-start;
  gap: 12px;
  
  ${props => props.isUser ? `
    flex-direction: row-reverse;
    text-align: right;
  ` : ''}
`;

const MessageAvatar = styled.div`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: ${props => props.isUser ? '#6366f1' : '#f1f5f9'};
  color: ${props => props.isUser ? 'white' : '#64748b'};
  
  svg {
    width: 18px;
    height: 18px;
  }
`;

const MessageContent = styled.div`
  max-width: 70%;
  padding: 16px 20px;
  border-radius: 18px;
  background: ${props => props.isUser ? '#6366f1' : '#f8fafc'};
  color: ${props => props.isUser ? 'white' : '#1e293b'};
  font-size: 15px;
  line-height: 1.5;
  word-wrap: break-word;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border: 1px solid ${props => props.isUser ? 'transparent' : '#e2e8f0'};
  
  ${props => props.isUser ? `
    border-bottom-right-radius: 6px;
  ` : `
    border-bottom-left-radius: 6px;
  `}
`;

const InputContainer = styled.div`
  display: flex;
  gap: 12px;
  padding: 20px 0 0 0;
  border-top: 1px solid #f1f5f9;
  background: #ffffff;
`;

const Input = styled.input`
  flex: 1;
  padding: 16px 20px;
  border: 1px solid #e2e8f0;
  border-radius: 24px;
  font-size: 15px;
  outline: none;
  transition: all 0.2s ease;
  background: #ffffff;
  
  &:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }
  
  &::placeholder {
    color: #94a3b8;
  }
`;

const SendButton = styled.button`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #6366f1;
  color: white;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  
  &:hover {
    background: #5855eb;
    transform: translateY(-1px);
  }
  
  &:disabled {
    background: #cbd5e1;
    cursor: not-allowed;
    transform: none;
  }
  
  svg {
    width: 20px;
    height: 20px;
  }
`;

const LoadingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: #64748b;
  font-size: 14px;
  padding: 16px 20px;
  background: #f8fafc;
  border-radius: 18px;
  border-bottom-left-radius: 6px;
  border: 1px solid #e2e8f0;
  max-width: 120px;
  
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6366f1;
    animation: bounce 1.4s infinite ease-in-out;
    
    &:nth-child(1) { animation-delay: -0.32s; }
    &:nth-child(2) { animation-delay: -0.16s; }
  }
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
`;

const ChatInterface = ({ likedProducts: globalLikedProducts, setLikedProducts: setGlobalLikedProducts }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "안녕하세요! AI 패션 어시스턴트입니다. 어떤 스타일을 찾고 계신가요?",
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [localLikedProducts, setLocalLikedProducts] = useState(new Set());
  const [showLikedOnly, setShowLikedOnly] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          session_id: sessionId
        })
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = {
          id: Date.now() + 1,
          text: data.response,
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
        
        // 세션 ID 저장
        if (data.session_id) {
          setSessionId(data.session_id);
        }
        
        // 추천 상품이 있으면 업데이트
        if (data.recommendations && data.recommendations.length > 0) {
          setRecommendations(data.recommendations);
        }
      } else {
        throw new Error('API 요청 실패');
      }
    } catch (error) {
      console.error('채팅 오류:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleLikeProduct = (product) => {
    const productId = product.product_id || product.id;
    
    // 로컬 상태 업데이트 (UI용)
    setLocalLikedProducts(prev => {
      const newLiked = new Set(prev);
      if (newLiked.has(productId)) {
        newLiked.delete(productId);
      } else {
        newLiked.add(productId);
      }
      return newLiked;
    });
    
    // 전역 상태 업데이트 (좋아요 목록용)
    if (globalLikedProducts && setGlobalLikedProducts) {
      setGlobalLikedProducts(prev => {
        const isLiked = prev.some(p => (p.product_id || p.id) === productId);
        if (isLiked) {
          // 좋아요 취소
          return prev.filter(p => (p.product_id || p.id) !== productId);
        } else {
          // 좋아요 추가
          return [...prev, product];
        }
      });
    }
  };

  const handleProductLink = (productUrl) => {
    if (productUrl) {
      window.open(productUrl, '_blank');
    }
  };

  const toggleLikedView = () => {
    setShowLikedOnly(!showLikedOnly);
  };

  // 표시할 상품 목록 계산
  const displayProducts = showLikedOnly 
    ? recommendations.filter(product => localLikedProducts.has(product.product_id || product.id))
    : recommendations;

  return (
    <ChatContainer>
      <ChatSection>
        <MessagesContainer>
          {messages.map((message) => (
            <Message key={message.id} isUser={message.isUser}>
              <MessageAvatar isUser={message.isUser}>
                {message.isUser ? <User size={18} /> : <Bot size={18} />}
              </MessageAvatar>
              <MessageContent isUser={message.isUser}>
                {message.text}
              </MessageContent>
            </Message>
          ))}
          
          {isLoading && (
            <Message isUser={false}>
              <MessageAvatar isUser={false}>
                <Bot size={18} />
              </MessageAvatar>
              <LoadingIndicator>
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </LoadingIndicator>
            </Message>
          )}
          
          <div ref={messagesEndRef} />
        </MessagesContainer>
        
        <InputContainer>
          <Input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="메시지를 입력하세요..."
            disabled={isLoading}
          />
          <SendButton onClick={handleSendMessage} disabled={!inputValue.trim() || isLoading}>
            <Send size={20} />
          </SendButton>
        </InputContainer>
      </ChatSection>
      
      <RecommendationsSection>
        <RecommendationsHeader>
          <RecommendationsTitle>
            {showLikedOnly ? '좋아요 목록' : '추천 상품'}
          </RecommendationsTitle>
          {recommendations.length > 0 && (
            <ViewLikedButton onClick={toggleLikedView}>
              <Heart size={14} />
              {showLikedOnly ? '전체 보기' : '좋아요 목록'}
            </ViewLikedButton>
          )}
        </RecommendationsHeader>
        {displayProducts.length > 0 ? (
          displayProducts.map((product, index) => (
            <ProductCard key={index}>
              {product.image_url && (
                <ProductImage 
                  src={product.image_url} 
                  alt={product.name}
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              )}
              <ProductName>{product.product_name || product.name}</ProductName>
              <ProductPrice>{product.price ? `₩${product.price.toLocaleString()}` : '가격 정보 없음'}</ProductPrice>
              <ProductDescription>
                {product.brand_kr ? `${product.brand_kr} - ` : ''}
                {product.description || product.product_description || '상품 설명이 없습니다.'}
              </ProductDescription>
              
              <ProductActions>
                <LikeButton 
                  isLiked={localLikedProducts.has(product.product_id || product.id)}
                  onClick={() => handleLikeProduct(product)}
                >
                  <Heart size={14} />
                  {localLikedProducts.has(product.product_id || product.id) ? '좋아요 취소' : '좋아요'}
                </LikeButton>
                <LinkButton 
                  onClick={() => handleProductLink(product.product_url || product.url)}
                  disabled={!product.product_url && !product.url}
                >
                  <ExternalLink size={14} />
                  바로가기
                </LinkButton>
              </ProductActions>
            </ProductCard>
          ))
        ) : (
          <NoRecommendations>
            {showLikedOnly 
              ? '좋아요한 상품이 없습니다.'
              : '대화를 시작하면 추천 상품이 여기에 표시됩니다.'
            }
          </NoRecommendations>
        )}
      </RecommendationsSection>
    </ChatContainer>
  );
};

export default ChatInterface; 