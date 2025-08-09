import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { 
  MessageCircle, 
  Palette, 
  Search, 
  Star, 
  Image, 
  TrendingUp, 
  Heart, 
  ShoppingBag, 
  Settings as SettingsIcon,
  Home,
  Upload,
  Send,
  RefreshCw
} from 'lucide-react';

// 공통 스타일 컴포넌트들
const FeatureContainer = styled.div`
  background: #ffffff;
  border-radius: 8px;
  padding: 32px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border: 1px solid #f1f5f9;
  height: calc(100vh - 120px);
  overflow-y: auto;
`;

const FeatureHeader = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 32px;
  padding-bottom: 20px;
  border-bottom: 1px solid #f1f5f9;
  
  svg {
    width: 28px;
    height: 28px;
    margin-right: 16px;
    color: #6366f1;
  }
  
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

const InputContainer = styled.div`
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
`;

const Input = styled.input`
  flex: 1;
  padding: 16px 20px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 16px;
  transition: all 0.2s ease;
  background: #ffffff;
  
  &:focus {
    outline: none;
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }
  
  &::placeholder {
    color: #94a3b8;
  }
`;

const Button = styled.button`
  padding: 16px 24px;
  background: #6366f1;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  
  &:hover {
    background: #5855eb;
    transform: translateY(-1px);
  }
  
  &:disabled {
    background: #cbd5e1;
    cursor: not-allowed;
    transform: none;
  }
`;

const ResultContainer = styled.div`
  background: #f8fafc;
  border-radius: 12px;
  padding: 24px;
  margin-top: 24px;
  border: 1px solid #e2e8f0;
`;

const ProductCard = styled.div`
  background: white;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  border: 1px solid #f1f5f9;
  transition: all 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const ProductName = styled.div`
  font-weight: 700;
  font-size: 18px;
  color: #1e293b;
  margin-bottom: 8px;
`;

const ProductPrice = styled.div`
  color: #6366f1;
  font-weight: 600;
  font-size: 16px;
`;

const ProductDescription = styled.div`
  color: #64748b;
  font-size: 14px;
  margin-top: 12px;
  line-height: 1.5;
`;

const CardGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 24px;
  margin-top: 24px;
`;

const InfoCard = styled.div`
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  border: 1px solid #f1f5f9;
  transition: all 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const CardTitle = styled.h3`
  font-size: 18px;
  font-weight: 700;
  color: #1e293b;
  margin: 16px 0 8px 0;
`;

const CardDescription = styled.p`
  font-size: 14px;
  color: #64748b;
  line-height: 1.5;
`;

// 1. 메인 페이지 컴포넌트
export const MainPage = () => {
  return (
    <FeatureContainer>
      <FeatureHeader>
        <Home />
        <div>
          <h2>AI 패션 어시스턴트</h2>
          <p>지능형 패션 추천 시스템에 오신 것을 환영합니다</p>
        </div>
      </FeatureHeader>
      
      <div style={{ textAlign: 'center', marginTop: '40px' }}>
        <h3 style={{ fontSize: '28px', color: '#1e293b', marginBottom: '20px', fontWeight: '700' }}>
          🎉 환영합니다!
        </h3>
        <p style={{ fontSize: '18px', color: '#64748b', lineHeight: '1.6', maxWidth: '600px', margin: '0 auto' }}>
          AI 패션 어시스턴트는 자연어 대화를 통해 개인화된 패션 추천을 제공합니다.
          <br />
          왼쪽 사이드바에서 원하는 기능을 선택하여 시작해보세요.
        </p>
        
        <CardGrid>
          <InfoCard>
            <MessageCircle size={40} color="#6366f1" />
            <CardTitle>챗봇 추천</CardTitle>
            <CardDescription>자연어 대화로 상품 추천</CardDescription>
          </InfoCard>
          <InfoCard>
            <Palette size={40} color="#6366f1" />
            <CardTitle>코디 추천</CardTitle>
            <CardDescription>상품 조합 및 스타일링</CardDescription>
          </InfoCard>
          <InfoCard>
            <Image size={40} color="#6366f1" />
            <CardTitle>이미지 검색</CardTitle>
            <CardDescription>이미지로 유사 상품 찾기</CardDescription>
          </InfoCard>
        </CardGrid>
      </div>
    </FeatureContainer>
  );
};

// 2. 코디네이션 추천 컴포넌트
export const CoordinationRecommendation = () => {
  const [baseProduct, setBaseProduct] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleCoordinationSearch = async () => {
    if (!baseProduct.trim()) return;
    
    setIsLoading(true);
    // API 호출 로직
    setTimeout(() => {
      setResults([
        {
          id: 1,
          name: '데님 팬츠',
          price: '45,000원',
          description: '스트라이프 티셔츠와 클래식한 조합'
        },
        {
          id: 2,
          name: '화이트 스니커즈',
          price: '89,000원',
          description: '깔끔한 화이트로 포인트'
        }
      ]);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <FeatureContainer>
      <FeatureHeader>
        <Palette />
        <div>
          <h2>코디네이션 추천</h2>
          <p>선택한 상품과 어울리는 조합을 추천해드립니다</p>
        </div>
      </FeatureHeader>
      
      <InputContainer>
        <Input
          placeholder="상품명 또는 상품 ID를 입력하세요 (예: 스트라이프 티셔츠)"
          value={baseProduct}
          onChange={(e) => setBaseProduct(e.target.value)}
        />
        <Button onClick={handleCoordinationSearch} disabled={isLoading}>
          {isLoading ? <RefreshCw size={20} /> : <Search size={20} />}
          {isLoading ? '검색 중...' : '코디 추천'}
        </Button>
      </InputContainer>
      
      {results.length > 0 && (
        <ResultContainer>
          <h3 style={{ marginBottom: '20px', color: '#1e293b', fontSize: '20px', fontWeight: '700' }}>추천 코디네이션</h3>
          {results.map((product) => (
            <ProductCard key={product.id}>
              <ProductName>{product.name}</ProductName>
              <ProductPrice>{product.price}</ProductPrice>
              <ProductDescription>{product.description}</ProductDescription>
            </ProductCard>
          ))}
        </ResultContainer>
      )}
    </FeatureContainer>
  );
};

// 3. 유사 상품 검색 컴포넌트
export const SimilarProductSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSimilarSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    // API 호출 로직
    setTimeout(() => {
      setResults([
        {
          id: 1,
          name: '유사한 스트라이프 티셔츠',
          price: '32,000원',
          description: '동일한 패턴이지만 다른 브랜드'
        },
        {
          id: 2,
          name: '스트라이프 니트',
          price: '55,000원',
          description: '비슷한 느낌의 니트 소재'
        }
      ]);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <FeatureContainer>
      <FeatureHeader>
        <Search />
        <div>
          <h2>유사 상품 검색</h2>
          <p>입력한 상품과 유사한 스타일의 상품을 찾아드립니다</p>
        </div>
      </FeatureHeader>
      
      <InputContainer>
        <Input
          placeholder="상품명을 입력하세요 (예: 스트라이프 티셔츠)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <Button onClick={handleSimilarSearch} disabled={isLoading}>
          {isLoading ? <RefreshCw size={20} /> : <Search size={20} />}
          {isLoading ? '검색 중...' : '유사 상품 찾기'}
        </Button>
      </InputContainer>
      
      {results.length > 0 && (
        <ResultContainer>
          <h3 style={{ marginBottom: '20px', color: '#1e293b', fontSize: '20px', fontWeight: '700' }}>유사 상품</h3>
          {results.map((product) => (
            <ProductCard key={product.id}>
              <ProductName>{product.name}</ProductName>
              <ProductPrice>{product.price}</ProductPrice>
              <ProductDescription>{product.description}</ProductDescription>
            </ProductCard>
          ))}
        </ResultContainer>
      )}
    </FeatureContainer>
  );
};

// 4. 리뷰 분석 컴포넌트
export const ReviewAnalysis = () => {
  const [productQuery, setProductQuery] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleReviewAnalysis = async () => {
    if (!productQuery.trim()) return;
    
    setIsLoading(true);
    
    try {
      // 새로운 리뷰 분석 API 호출
      const response = await fetch(`http://localhost:8001/review-analysis/${encodeURIComponent(productQuery)}`);
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.product_summaries && data.product_summaries.length > 0) {
          // 상품별 요약을 하나의 분석 결과로 통합
          const summaries = data.product_summaries.map(item => 
            `[${item.brand_kr}] ${item.product_name} (${item.price}원): ${item.summary}`
          ).join('\n\n');
          
          const avgRating = data.product_summaries.reduce((sum, item) => sum + item.average_rating, 0) / data.product_summaries.length;
          const totalReviews = data.product_summaries.reduce((sum, item) => sum + item.review_count, 0);
          
          setAnalysis({
            overallRating: avgRating.toFixed(1),
            pros: ['다양한 상품의 리뷰를 분석했습니다'],
            cons: ['개별 상품의 상세 리뷰는 확인이 필요합니다'],
            recommendation: summaries,
            reviewCount: totalReviews
          });
        } else {
          setAnalysis({
            overallRating: 0,
            pros: ['리뷰 데이터 없음'],
            cons: ['해당 키워드의 상품을 찾을 수 없습니다'],
            recommendation: `${productQuery} 관련 상품의 리뷰를 찾을 수 없습니다.`,
            reviewCount: 0
          });
        }
      } else {
        throw new Error('리뷰 분석 API 요청 실패');
      }
    } catch (error) {
      console.error('리뷰 분석 오류:', error);
      setAnalysis({
        overallRating: 0,
        pros: ['분석 실패'],
        cons: ['네트워크 오류'],
        recommendation: '리뷰 분석 중 오류가 발생했습니다.',
        reviewCount: 0
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <FeatureContainer>
      <FeatureHeader>
        <Star />
        <div>
          <h2>리뷰 분석</h2>
          <p>상품 리뷰를 분석하여 인사이트를 제공합니다</p>
        </div>
      </FeatureHeader>
      
      <InputContainer>
        <Input
          placeholder="상품명을 입력하세요 (예: 스트라이프 티셔츠)"
          value={productQuery}
          onChange={(e) => setProductQuery(e.target.value)}
        />
        <Button onClick={handleReviewAnalysis} disabled={isLoading}>
          {isLoading ? <RefreshCw size={20} /> : <Star size={20} />}
          {isLoading ? '분석 중...' : '리뷰 분석'}
        </Button>
      </InputContainer>
      
      {analysis && (
        <ResultContainer>
          <h3 style={{ marginBottom: '20px', color: '#1e293b', fontSize: '20px', fontWeight: '700' }}>
            "{productQuery}" 리뷰 분석 결과 ({analysis.product_summaries?.length || 0}개 상품)
          </h3>
          
          {analysis.product_summaries ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {analysis.product_summaries.slice(0, 10).map((product, index) => (
                <div key={index} style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '1fr 1fr', 
                  gap: '24px',
                  background: 'white',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px',
                  padding: '20px',
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
                }}>
                  {/* 왼쪽: 상품 정보 */}
                  <div>
                    <h4 style={{ 
                      margin: '0 0 16px 0', 
                      fontSize: '16px', 
                      fontWeight: '600', 
                      color: '#1e293b',
                      borderBottom: '2px solid #6366f1',
                      paddingBottom: '8px'
                    }}>
                      📦 상품 정보
                    </h4>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      {/* 상품 이미지 */}
                      <div style={{ 
                        width: '80px', 
                        height: '80px', 
                        borderRadius: '8px', 
                        overflow: 'hidden',
                        flexShrink: 0,
                        background: '#f8fafc',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        {product.image_url ? (
                          <img 
                            src={product.image_url} 
                            alt={product.product_name}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            onError={(e) => {
                              e.target.style.display = 'none';
                              e.target.nextSibling.style.display = 'flex';
                            }}
                          />
                        ) : null}
                        <div style={{ 
                          display: product.image_url ? 'none' : 'flex',
                          alignItems: 'center', 
                          justifyContent: 'center',
                          width: '100%',
                          height: '100%',
                          color: '#94a3b8',
                          fontSize: '12px'
                        }}>
                          이미지 없음
                        </div>
                      </div>
                      
                      {/* 상품 정보 */}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <h5 style={{ 
                          margin: '0 0 8px 0', 
                          fontSize: '16px', 
                          fontWeight: '600', 
                          color: '#1e293b',
                          lineHeight: '1.3'
                        }}>
                          {product.product_name}
                        </h5>
                        <p style={{ 
                          margin: '0 0 8px 0', 
                          fontSize: '14px', 
                          color: '#64748b' 
                        }}>
                          {product.brand_kr} • {product.price?.toLocaleString()}원
                        </p>
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '8px'
                        }}>
                          <span style={{ 
                            color: '#f59e0b', 
                            fontSize: '14px', 
                            fontWeight: '600' 
                          }}>
                            ⭐ {product.average_rating?.toFixed(1)}
                          </span>
                          <span style={{ 
                            color: '#64748b', 
                            fontSize: '14px' 
                          }}>
                            리뷰 {product.review_count}개
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* 오른쪽: 리뷰 요약 */}
                  <div>
                    <h4 style={{ 
                      margin: '0 0 16px 0', 
                      fontSize: '16px', 
                      fontWeight: '600', 
                      color: '#1e293b',
                      borderBottom: '2px solid #10b981',
                      paddingBottom: '8px'
                    }}>
                      💬 리뷰 요약
                    </h4>
                    <div style={{ 
                      background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                      borderRadius: '8px',
                      padding: '16px',
                      border: '1px solid #bae6fd'
                    }}>
                      <p style={{ 
                        margin: 0, 
                        fontSize: '14px', 
                        color: '#475569',
                        lineHeight: '1.6',
                        textAlign: 'justify'
                      }}>
                        {product.summary}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: '1fr 1fr', 
                gap: '24px',
                background: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                padding: '20px',
                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
              }}>
                {/* 왼쪽: 상품 정보 */}
                <div>
                  <h4 style={{ 
                    margin: '0 0 16px 0', 
                    fontSize: '16px', 
                    fontWeight: '600', 
                    color: '#1e293b',
                    borderBottom: '2px solid #6366f1',
                    paddingBottom: '8px'
                  }}>
                    📦 상품 정보
                  </h4>
                  <div style={{ 
                    textAlign: 'center', 
                    padding: '40px', 
                    color: '#64748b',
                    background: '#f8fafc',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0'
                  }}>
                    상품을 찾을 수 없습니다
                  </div>
                </div>
                
                {/* 오른쪽: 리뷰 요약 */}
                <div>
                  <h4 style={{ 
                    margin: '0 0 16px 0', 
                    fontSize: '16px', 
                    fontWeight: '600', 
                    color: '#1e293b',
                    borderBottom: '2px solid #10b981',
                    paddingBottom: '8px'
                  }}>
                    💬 리뷰 요약
                  </h4>
                  <div style={{ 
                    background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                    borderRadius: '8px',
                    padding: '16px',
                    border: '1px solid #bae6fd'
                  }}>
                    <div style={{ 
                      textAlign: 'center', 
                      padding: '40px', 
                      color: '#64748b'
                    }}>
                      {analysis.recommendation}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </ResultContainer>
      )}
    </FeatureContainer>
  );
};

// 5. 이미지 검색 컴포넌트
export const ImageSearch = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleImageSearch = async () => {
    if (!selectedFile) return;
    
    setIsLoading(true);
    // API 호출 로직
    setTimeout(() => {
      setResults([
        {
          id: 1,
          name: '유사한 스타일 티셔츠',
          price: '38,000원',
          description: '업로드한 이미지와 매우 유사한 패턴'
        },
        {
          id: 2,
          name: '스트라이프 니트',
          price: '52,000원',
          description: '비슷한 느낌의 니트 소재'
        }
      ]);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <FeatureContainer>
      <FeatureHeader>
        <Image />
        <div>
          <h2>이미지 검색</h2>
          <p>이미지를 업로드하여 유사한 상품을 찾아드립니다</p>
        </div>
      </FeatureHeader>
      
      <div style={{ marginBottom: '24px' }}>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ display: 'none' }}
          id="image-upload"
        />
        <label htmlFor="image-upload">
          <Button as="span" style={{ cursor: 'pointer' }}>
            <Upload size={20} />
            이미지 선택
          </Button>
        </label>
        {selectedFile && (
          <span style={{ marginLeft: '16px', color: '#64748b', fontSize: '16px' }}>
            선택된 파일: {selectedFile.name}
          </span>
        )}
      </div>
      
      {selectedFile && (
        <Button onClick={handleImageSearch} disabled={isLoading} style={{ marginBottom: '24px' }}>
          {isLoading ? <RefreshCw size={20} /> : <Search size={20} />}
          {isLoading ? '검색 중...' : '유사 상품 찾기'}
        </Button>
      )}
      
      {results.length > 0 && (
        <ResultContainer>
          <h3 style={{ marginBottom: '20px', color: '#1e293b', fontSize: '20px', fontWeight: '700' }}>유사 상품</h3>
          {results.map((product) => (
            <ProductCard key={product.id}>
              <ProductName>{product.name}</ProductName>
              <ProductPrice>{product.price}</ProductPrice>
              <ProductDescription>{product.description}</ProductDescription>
            </ProductCard>
          ))}
        </ResultContainer>
      )}
    </FeatureContainer>
  );
};

// 6. 트렌드 분석 컴포넌트
export const TrendAnalysis = () => {
  const [trends] = useState([
    { category: '상의', trend: '스트라이프 패턴', popularity: 85 },
    { category: '하의', trend: '와이드 팬츠', popularity: 78 },
    { category: '신발', trend: '화이트 스니커즈', popularity: 92 },
    { category: '가방', trend: '미니 백', popularity: 67 }
  ]);

  return (
    <FeatureContainer>
      <FeatureHeader>
        <TrendingUp />
        <div>
          <h2>트렌드 분석</h2>
          <p>현재 인기 있는 패션 트렌드를 분석해드립니다</p>
        </div>
      </FeatureHeader>
      
      <ResultContainer>
        <h3 style={{ marginBottom: '24px', color: '#1e293b', fontSize: '20px', fontWeight: '700' }}>현재 트렌드</h3>
        {trends.map((trend, index) => (
          <ProductCard key={index}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <ProductName>{trend.category}</ProductName>
                <ProductDescription>{trend.trend}</ProductDescription>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '28px', fontWeight: '700', color: '#6366f1' }}>
                  {trend.popularity}%
                </div>
                <div style={{ fontSize: '14px', color: '#64748b' }}>인기도</div>
              </div>
            </div>
          </ProductCard>
        ))}
      </ResultContainer>
    </FeatureContainer>
  );
};

// 7. 좋아요 목록 컴포넌트
export const LikedProducts = ({ likedProducts, onUnlike }) => {
  return (
    <FeatureContainer>
      <FeatureHeader>
        <Heart />
        <div>
          <h2>좋아요 목록</h2>
          <p>저장한 상품들을 관리합니다</p>
        </div>
      </FeatureHeader>
      
      {likedProducts.length === 0 ? (
        <div style={{ textAlign: 'center', marginTop: '80px', color: '#64748b' }}>
          <Heart size={64} color="#cbd5e1" />
          <p style={{ marginTop: '20px', fontSize: '18px' }}>아직 좋아요한 상품이 없습니다.</p>
        </div>
      ) : (
        <div>
          {likedProducts.map((product) => (
            <ProductCard key={product.product_id}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <ProductName>{product.product_name}</ProductName>
                  <ProductPrice>{product.price?.toLocaleString()}원</ProductPrice>
                </div>
                <Button
                  onClick={() => onUnlike(product)}
                  style={{ background: '#ef4444', fontSize: '14px', padding: '12px 16px' }}
                >
                  좋아요 취소
                </Button>
              </div>
            </ProductCard>
          ))}
        </div>
      )}
    </FeatureContainer>
  );
};

// 8. 상품 브라우저 컴포넌트
export const ProductBrowser = () => {
  const [products, setProducts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('http://localhost:8001/products');
        
        if (response.ok) {
          const data = await response.json();
          setProducts(data);
        } else {
          throw new Error('상품 데이터를 불러올 수 없습니다.');
        }
      } catch (error) {
        console.error('상품 브라우저 오류:', error);
        setError('상품 데이터를 불러오는 중 오류가 발생했습니다.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchProducts();
  }, []);

  return (
    <FeatureContainer>
      <FeatureHeader>
        <ShoppingBag />
        <div>
          <h2>상품 브라우저</h2>
          <p>전체 상품 카탈로그를 둘러보세요</p>
        </div>
      </FeatureHeader>
      
      {isLoading ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
          상품을 불러오는 중...
        </div>
      ) : error ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#ef4444' }}>
          {error}
        </div>
      ) : (
        <div>
          {products.map((product) => (
            <ProductCard key={product.id}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <ProductName>{product.name}</ProductName>
                  <ProductDescription>{product.brand} • {product.category}</ProductDescription>
                </div>
                <ProductPrice>{product.price.toLocaleString()}원</ProductPrice>
              </div>
            </ProductCard>
          ))}
        </div>
      )}
    </FeatureContainer>
  );
};

// 9. 설정 컴포넌트
export const Settings = () => {
  return (
    <FeatureContainer>
      <FeatureHeader>
        <SettingsIcon />
        <div>
          <h2>설정</h2>
          <p>개인화 설정 및 옵션을 관리합니다</p>
        </div>
      </FeatureHeader>
      
      <div style={{ display: 'grid', gap: '24px' }}>
        <div>
          <h4 style={{ marginBottom: '16px', color: '#1e293b', fontSize: '18px', fontWeight: '600' }}>개인화 설정</h4>
          <div style={{ background: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #f1f5f9' }}>
            <label style={{ display: 'flex', alignItems: 'center', marginBottom: '16px', fontSize: '16px' }}>
              <input type="checkbox" style={{ marginRight: '12px' }} />
              추천 개인화 활성화
            </label>
            <label style={{ display: 'flex', alignItems: 'center', marginBottom: '16px', fontSize: '16px' }}>
              <input type="checkbox" style={{ marginRight: '12px' }} />
              가격 알림 받기
            </label>
            <label style={{ display: 'flex', alignItems: 'center', fontSize: '16px' }}>
              <input type="checkbox" style={{ marginRight: '12px' }} />
              트렌드 알림 받기
            </label>
          </div>
        </div>
        
        <div>
          <h4 style={{ marginBottom: '16px', color: '#1e293b', fontSize: '18px', fontWeight: '600' }}>테마 설정</h4>
          <div style={{ background: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #f1f5f9' }}>
            <label style={{ display: 'flex', alignItems: 'center', marginBottom: '16px', fontSize: '16px' }}>
              <input type="radio" name="theme" style={{ marginRight: '12px' }} defaultChecked />
              라이트 모드
            </label>
            <label style={{ display: 'flex', alignItems: 'center', fontSize: '16px' }}>
              <input type="radio" name="theme" style={{ marginRight: '12px' }} />
              다크 모드
            </label>
          </div>
        </div>
      </div>
    </FeatureContainer>
  );
}; 