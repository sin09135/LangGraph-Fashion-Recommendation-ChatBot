from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from config import OPENAI_MODEL, OPENAI_TEMPERATURE

class LLMService:
    """LLM 서비스 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL, 
            temperature=OPENAI_TEMPERATURE
        )
        self.embeddings = OpenAIEmbeddings()
    
    def invoke(self, prompt: str) -> str:
        """LLM 호출"""
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return ""
    
    def create_embedding(self, text: str) -> list:
        """텍스트 임베딩 생성"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return []

# 전역 LLM 서비스 인스턴스
llm_service = LLMService() 