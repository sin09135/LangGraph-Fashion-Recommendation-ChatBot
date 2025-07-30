import React from 'react';
import ReactDOM from 'react-dom/client';
import { createGlobalStyle } from 'styled-components';
import App from './App';

const GlobalStyle = createGlobalStyle`
  * {
    box-sizing: border-box;
  }
  
  body {
    margin: 0;
    padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
      sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  
  /* 모바일에서 터치 하이라이트 제거 */
  * {
    -webkit-tap-highlight-color: transparent;
  }
  
  /* 포커스 스타일 개선 */
  *:focus {
    outline: 2px solid #667eea;
    outline-offset: 2px;
  }
  
  /* 버튼 포커스 스타일 */
  button:focus {
    outline: 2px solid #667eea;
    outline-offset: 2px;
  }
  
  /* 입력 필드 포커스 스타일 */
  input:focus {
    outline: none;
  }
`;

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <GlobalStyle />
    <App />
  </React.StrictMode>
); 