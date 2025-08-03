import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { Send, Bot, User, Image, X } from 'lucide-react';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  
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
    gap: 14px;
  }
`;

const Message = styled.div`
  display: flex;
  gap: 10px;
  align-items: flex-start;
  animation: fadeIn 0.3s ease-in;
  
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
`;

const MessageAvatar = styled.div`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: ${props => props.isUser ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)'};
  color: ${props => props.isUser ? 'white' : '#64748b'};
  box-shadow: ${props => props.isUser ? '0 4px 12px rgba(102, 126, 234, 0.3)' : '0 2px 8px rgba(0, 0, 0, 0.1)'};
  border: 2px solid ${props => props.isUser ? 'rgba(255, 255, 255, 0.2)' : 'rgba(255, 255, 255, 0.8)'};
`;

const MessageContent = styled.div`
  flex: 1;
  padding: 14px 18px;
  border-radius: 20px;
  background: ${props => props.isUser ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'rgba(255, 255, 255, 0.9)'};
  color: ${props => props.isUser ? 'white' : '#1e293b'};
  max-width: 80%;
  word-wrap: break-word;
  line-height: 1.6;
  box-shadow: ${props => props.isUser ? '0 4px 12px rgba(102, 126, 234, 0.3)' : '0 2px 8px rgba(0, 0, 0, 0.08)'};
  border: 1px solid ${props => props.isUser ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.05)'};
  backdrop-filter: blur(10px);
`;

const InputContainer = styled.div`
  padding: 24px;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
  display: flex;
  gap: 12px;
  align-items: center;
  background: rgba(255, 255, 255, 0.5);
  backdrop-filter: blur(10px);
  
  @media (max-width: 768px) {
    padding: 20px;
    gap: 10px;
  }
`;

const Input = styled.input`
  flex: 1;
  padding: 14px 20px;
  border: 2px solid rgba(0, 0, 0, 0.1);
  border-radius: 28px;
  font-size: 14px;
  outline: none;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  
  &:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    background: rgba(255, 255, 255, 1);
  }
  
  &::placeholder {
    color: #94a3b8;
  }
  
  @media (max-width: 768px) {
    padding: 12px 18px;
    font-size: 16px; /* 모바일에서 자동 확대 방지 */
  }
`;

const SendButton = styled.button`
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
  }
  
  &:disabled {
    background: #cbd5e1;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
  
  @media (max-width: 768px) {
    width: 48px;
    height: 48px;
  }
`;

const ImageUploadButton = styled.button`
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 50%;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  color: #64748b;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.8);
  
  &:hover {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
  
  @media (max-width: 768px) {
    width: 48px;
    height: 48px;
  }
`;

const HiddenFileInput = styled.input`
  display: none;
`;

const ImagePreview = styled.div`
  position: relative;
  margin: 10px 0;
  display: inline-block;
`;

const PreviewImage = styled.img`
  max-width: 200px;
  max-height: 200px;
  border-radius: 8px;
  border: 2px solid #e0e0e0;
`;

const RemoveImageButton = styled.button`
  position: absolute;
  top: -8px;
  right: -8px;
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 50%;
  background: #ff4444;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 12px;
  
  &:hover {
    background: #cc0000;
  }
`;

const LoadingMessage = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: #666;
  font-style: italic;
`;

const LoadingDots = styled.div`
  display: flex;
  gap: 4px;
  
  span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #667eea;
    animation: bounce 1.4s infinite ease-in-out;
    
    &:nth-child(1) { animation-delay: -0.32s; }
    &:nth-child(2) { animation-delay: -0.16s; }
  }
  
  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
  }
`;

function ChatInterface({ onProductsReceived, sessionId, setSessionId, setIsLoading }) {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoadingLocal, setIsLoadingLocal] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 이미지 파일을 Base64로 변환하는 함수
  const convertImageToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // 이미지 파일 선택 처리
  const handleImageSelect = async (event) => {
    const file = event.target.files[0];
    if (file) {
      // 파일 크기 체크 (5MB 제한)
      if (file.size > 5 * 1024 * 1024) {
        alert('이미지 파일 크기는 5MB 이하여야 합니다.');
        return;
      }

      // 파일 타입 체크
      if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
      }

      try {
        const base64 = await convertImageToBase64(file);
        setSelectedImage(URL.createObjectURL(file));
        setImageBase64(base64);
      } catch (error) {
        console.error('이미지 변환 오류:', error);
        alert('이미지 처리 중 오류가 발생했습니다.');
      }
    }
  };

  // 이미지 제거
  const removeImage = () => {
    setSelectedImage(null);
    setImageBase64(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // 이미지 업로드 버튼 클릭
  const handleImageUploadClick = () => {
    fileInputRef.current?.click();
  };

  const sendMessage = async () => {
    if (!inputValue.trim() && !imageBase64) return;
    if (isLoadingLocal) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoadingLocal(true);
    setIsLoading(true);

    // 사용자 메시지 추가 (이미지가 있으면 이미지 포함)
    const messageContent = imageBase64 
      ? `${userMessage} [이미지 첨부됨]`
      : userMessage;
    
    const newUserMessage = {
      id: Date.now(),
      content: messageContent,
      isUser: true,
      timestamp: new Date(),
      image: selectedImage // 미리보기용
    };
    setMessages(prev => [...prev, newUserMessage]);

    try {
      const requestBody = {
        message: userMessage || "이 이미지와 유사한 상품 추천해줘",
        session_id: sessionId
      };

      // 이미지가 있으면 요청에 포함
      if (imageBase64) {
        requestBody.input_image = imageBase64;
      }

      const response = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      const data = await response.json();

      if (response.ok) {
        // AI 응답 추가
        const aiMessage = {
          id: Date.now() + 1,
          content: data.response,
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, aiMessage]);

        // 세션 ID 저장
        if (data.session_id && !sessionId) {
          setSessionId(data.session_id);
        }

        // 추천 상품 전달
        if (data.recommendations && data.recommendations.length > 0) {
          onProductsReceived(data.recommendations);
        }
      } else {
        // 에러 메시지 추가
        const errorMessage = {
          id: Date.now() + 1,
          content: `죄송합니다. 오류가 발생했습니다: ${data.detail || '알 수 없는 오류'}. 다시 시도해주세요.`,
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('채팅 오류:', error);
      const errorMessage = {
        id: Date.now() + 1,
        content: error.message.includes('Failed to fetch') 
          ? '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.'
          : '네트워크 오류가 발생했습니다. 연결을 확인해주세요.',
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoadingLocal(false);
      setIsLoading(false);
      // 이미지 초기화
      removeImage();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <ChatContainer>
      <MessagesContainer>
        {messages.length === 0 && (
          <Message>
            <MessageAvatar isUser={false}>
              <Bot size={16} />
            </MessageAvatar>
            <MessageContent isUser={false}>
              안녕하세요! 👋
              <br /><br />
              저는 AI 패션 추천 챗봇입니다. 당신만의 완벽한 스타일을 찾아드릴게요!
              <br /><br />
              🎯 <strong>주요 기능</strong>
              <br />
              • <strong>상품 추천</strong>: "버뮤다 팬츠 4만원 미만으로 추천해줘"
              <br />
              • <strong>코디 추천</strong>: "1번 상품과 코디하기 좋은 상품 추천해줘"
              <br />
              • <strong>유사 상품</strong>: "이 상품과 비슷한 스타일 추천해줘"
              <br />
              • <strong>리뷰 분석</strong>: "1번 상품 리뷰는 어때?"
              <br />
              • <strong>이미지 검색</strong>: 사진을 업로드하면 유사한 상품을 찾아드려요
              <br /><br />
              💡 <strong>사용 팁</strong>
              <br />
              - 구체적인 조건을 말씀해주시면 더 정확한 추천이 가능해요
              <br />
              - 가격, 브랜드, 스타일 등을 자유롭게 조합해서 요청해보세요
              <br />
              - 좋아하는 상품은 하트 버튼을 눌러서 저장할 수 있어요
              <br /><br />
              어떤 패션을 찾고 계신가요? 😊
            </MessageContent>
          </Message>
        )}
        
        {messages.map((message) => (
          <Message key={message.id}>
            <MessageAvatar isUser={message.isUser}>
              {message.isUser ? <User size={16} /> : <Bot size={16} />}
            </MessageAvatar>
            <MessageContent isUser={message.isUser}>
              {message.content}
              {message.image && (
                <div style={{ marginTop: '8px' }}>
                  <PreviewImage 
                    src={message.image} 
                    alt="첨부된 이미지" 
                    style={{ maxWidth: '150px', maxHeight: '150px' }}
                  />
                </div>
              )}
            </MessageContent>
          </Message>
        ))}
        
        {isLoadingLocal && (
          <Message>
            <MessageAvatar isUser={false}>
              <Bot size={16} />
            </MessageAvatar>
            <MessageContent isUser={false}>
              <LoadingMessage>
                <span>AI가 상품을 분석하고 추천 중입니다...</span>
                <LoadingDots>
                  <span></span>
                  <span></span>
                  <span></span>
                </LoadingDots>
              </LoadingMessage>
            </MessageContent>
          </Message>
        )}
        
        <div ref={messagesEndRef} />
      </MessagesContainer>
      
      <InputContainer>
        {/* 이미지 미리보기 */}
        {selectedImage && (
          <ImagePreview>
            <PreviewImage src={selectedImage} alt="선택된 이미지" />
            <RemoveImageButton onClick={removeImage}>
              <X size={12} />
            </RemoveImageButton>
          </ImagePreview>
        )}
        
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', width: '100%' }}>
          <ImageUploadButton onClick={handleImageUploadClick} disabled={isLoadingLocal}>
            <Image size={18} />
          </ImageUploadButton>
          
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="메시지를 입력하세요..."
            disabled={isLoadingLocal}
          />
          
          <SendButton onClick={sendMessage} disabled={isLoadingLocal || (!inputValue.trim() && !imageBase64)}>
            <Send size={18} />
          </SendButton>
        </div>
        
        <HiddenFileInput
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageSelect}
        />
      </InputContainer>
    </ChatContainer>
  );
}

export default ChatInterface; 