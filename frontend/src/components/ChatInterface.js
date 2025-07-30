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
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 15px;
  
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
    gap: 12px;
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
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: ${props => props.isUser ? '#667eea' : '#f0f0f0'};
  color: ${props => props.isUser ? 'white' : '#666'};
`;

const MessageContent = styled.div`
  flex: 1;
  padding: 12px 16px;
  border-radius: 18px;
  background: ${props => props.isUser ? '#667eea' : '#f8f9fa'};
  color: ${props => props.isUser ? 'white' : '#333'};
  max-width: 80%;
  word-wrap: break-word;
  line-height: 1.5;
`;

const InputContainer = styled.div`
  padding: 20px;
  border-top: 1px solid #f0f0f0;
  display: flex;
  gap: 10px;
  align-items: center;
  
  @media (max-width: 768px) {
    padding: 15px;
    gap: 8px;
  }
`;

const Input = styled.input`
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #e0e0e0;
  border-radius: 25px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.3s ease;
  
  &:focus {
    border-color: #667eea;
  }
  
  &::placeholder {
    color: #999;
  }
  
  @media (max-width: 768px) {
    padding: 10px 14px;
    font-size: 16px; /* 모바일에서 자동 확대 방지 */
  }
`;

const SendButton = styled.button`
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background: #667eea;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: #5a6fd8;
    transform: scale(1.05);
  }
  
  &:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
  }
  
  @media (max-width: 768px) {
    width: 44px;
    height: 44px;
  }
`;

const ImageUploadButton = styled.button`
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background: #f0f0f0;
  color: #666;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: #e0e0e0;
    transform: scale(1.05);
  }
  
  @media (max-width: 768px) {
    width: 44px;
    height: 44px;
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
              안녕하세요! AI 패션 어시스턴트입니다. 
              어떤 스타일의 옷을 찾고 계신가요? 
              예시: "버뮤다 팬츠 4만원 미만으로 추천해줘"
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