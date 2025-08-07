import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';

const MCPContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 30px;
  
  h1 {
    color: #333;
    margin-bottom: 10px;
  }
  
  p {
    color: #666;
    font-size: 16px;
  }
`;

const ToolGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const ToolCard = styled.div`
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid #e1e5e9;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
  }
`;

const ToolTitle = styled.h3`
  color: #2c3e50;
  margin-bottom: 10px;
  font-size: 18px;
`;

const ToolDescription = styled.p`
  color: #666;
  margin-bottom: 15px;
  line-height: 1.5;
`;

const FormGroup = styled.div`
  margin-bottom: 15px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 5px;
  color: #333;
  font-weight: 500;
`;

const Input = styled.input`
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  
  &:focus {
    outline: none;
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
  }
`;

const Select = styled.select`
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  background: white;
  
  &:focus {
    outline: none;
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
  }
`;

const Button = styled.button`
  background: #3498db;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s ease;
  
  &:hover {
    background: #2980b9;
  }
  
  &:disabled {
    background: #bdc3c7;
    cursor: not-allowed;
  }
`;

const ResultContainer = styled.div`
  background: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
  margin-top: 20px;
  border-left: 4px solid #3498db;
`;

const ResultTitle = styled.h4`
  color: #2c3e50;
  margin-bottom: 10px;
`;

const ResultText = styled.div`
  color: #333;
  line-height: 1.6;
  white-space: pre-wrap;
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 10px;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  background: #fee;
  color: #c33;
  padding: 10px;
  border-radius: 6px;
  margin-top: 10px;
  border-left: 4px solid #c33;
`;

const MCPInterface = () => {
  const [tools, setTools] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState({});
  const [formData, setFormData] = useState({});
  
  // MCP 서버 연결 상태
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    // MCP 서버에 연결
    connectToMCPServer();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectToMCPServer = () => {
    try {
      // WebSocket을 통한 MCP 서버 연결 (실제 구현에서는 적절한 엔드포인트 사용)
      const ws = new WebSocket('ws://localhost:8001/mcp');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('MCP 서버에 연결되었습니다.');
        setIsConnected(true);
        loadTools();
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMCPMessage(data);
      };

      ws.onerror = (error) => {
        console.error('MCP 서버 연결 오류:', error);
        setError('MCP 서버 연결에 실패했습니다.');
        setIsConnected(false);
      };

      ws.onclose = () => {
        console.log('MCP 서버 연결이 종료되었습니다.');
        setIsConnected(false);
      };

    } catch (error) {
      console.error('MCP 서버 연결 실패:', error);
      setError('MCP 서버 연결에 실패했습니다.');
    }
  };

  const loadTools = async () => {
    try {
      setLoading(true);
      
      // MCP 서버로부터 도구 목록 요청
      const response = await fetch('/api/mcp/tools');
      const toolsData = await response.json();
      
      setTools(toolsData);
      setError(null);
    } catch (error) {
      console.error('도구 목록 로드 실패:', error);
      setError('도구 목록을 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleMCPMessage = (data) => {
    if (data.type === 'tool_result') {
      setResults(prev => ({
        ...prev,
        [data.tool_name]: data.result
      }));
    }
  };

  const handleInputChange = (toolName, fieldName, value) => {
    setFormData(prev => ({
      ...prev,
      [toolName]: {
        ...prev[toolName],
        [fieldName]: value
      }
    }));
  };

  const callTool = async (toolName) => {
    try {
      setLoading(true);
      setError(null);

      const toolData = formData[toolName] || {};
      
      // MCP 서버로 도구 호출 요청
      const response = await fetch('/api/mcp/call-tool', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tool_name: toolName,
          arguments: toolData,
          session_id: 'default'
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(prev => ({
          ...prev,
          [toolName]: result.data
        }));
      } else {
        setError(result.error || '도구 호출에 실패했습니다.');
      }
    } catch (error) {
      console.error('도구 호출 오류:', error);
      setError('도구 호출 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const renderToolForm = (tool) => {
    const schema = tool.inputSchema;
    const properties = schema?.properties || {};
    const required = schema?.required || [];
    
    return (
      <ToolCard key={tool.name}>
        <ToolTitle>{tool.name}</ToolTitle>
        <ToolDescription>{tool.description}</ToolDescription>
        
        {Object.entries(properties).map(([fieldName, fieldSchema]) => (
          <FormGroup key={fieldName}>
            <Label>
              {fieldName}
              {required.includes(fieldName) && <span style={{color: 'red'}}> *</span>}
            </Label>
            
            {fieldSchema.type === 'string' && fieldSchema.enum ? (
              <Select
                value={formData[tool.name]?.[fieldName] || ''}
                onChange={(e) => handleInputChange(tool.name, fieldName, e.target.value)}
              >
                <option value="">선택하세요</option>
                {fieldSchema.enum.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </Select>
            ) : fieldSchema.type === 'number' ? (
              <Input
                type="number"
                placeholder={fieldSchema.description}
                value={formData[tool.name]?.[fieldName] || ''}
                onChange={(e) => handleInputChange(tool.name, fieldName, parseFloat(e.target.value) || '')}
              />
            ) : fieldSchema.type === 'array' ? (
              <Input
                type="text"
                placeholder={fieldSchema.description + " (쉼표로 구분)"}
                value={formData[tool.name]?.[fieldName] || ''}
                onChange={(e) => handleInputChange(tool.name, fieldName, e.target.value.split(',').map(s => s.trim()))}
              />
            ) : (
              <Input
                type="text"
                placeholder={fieldSchema.description}
                value={formData[tool.name]?.[fieldName] || ''}
                onChange={(e) => handleInputChange(tool.name, fieldName, e.target.value)}
              />
            )}
          </FormGroup>
        ))}
        
        <Button 
          onClick={() => callTool(tool.name)}
          disabled={loading || !isConnected}
        >
          {loading && <LoadingSpinner />}
          {tool.name.replace('fashion_', '').replace(/_/g, ' ').toUpperCase()} 실행
        </Button>
        
        {results[tool.name] && (
          <ResultContainer>
            <ResultTitle>결과:</ResultTitle>
            <ResultText>{results[tool.name]}</ResultText>
          </ResultContainer>
        )}
      </ToolCard>
    );
  };

  return (
    <MCPContainer>
      <Header>
        <h1>패션 추천 MCP 인터페이스</h1>
        <p>Model Context Protocol을 통한 패션 추천 시스템</p>
        <div style={{marginTop: '10px'}}>
          <span style={{
            color: isConnected ? '#27ae60' : '#e74c3c',
            fontWeight: 'bold'
          }}>
            {isConnected ? '🟢 연결됨' : '🔴 연결 안됨'}
          </span>
        </div>
      </Header>

      {error && (
        <ErrorMessage>
          {error}
        </ErrorMessage>
      )}

      {loading && tools.length === 0 ? (
        <div style={{textAlign: 'center', padding: '40px'}}>
          <LoadingSpinner />
          <p>도구 목록을 불러오는 중...</p>
        </div>
      ) : (
        <ToolGrid>
          {tools.map(renderToolForm)}
        </ToolGrid>
      )}
    </MCPContainer>
  );
};

export default MCPInterface; 