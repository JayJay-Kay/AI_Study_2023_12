<!DOCTYPE html>
<html>
<head>
  <title>챕터 8</title>
</head>

<body>
<h2>챕터 8: 감정 분석에 머신 러닝 적용</h2>

- 의견 분석(opinion mining)/감성 분석(sentiment analysis) <br>
  - 사람들의 의견, 감정, 태도와 같은 주관적 정보를 객관적인 형태로 추출하는 것 <br>
  - 분석 단계 <br>
      1. 극성 분석(polarity analysis) <br>
        - 텍스트가 긍정, 부정, 중립인지 판단 <br>
      2. 감정 분석(Emotion Analysis) <br>
        - 텍스트의 특정 감정 탐지 (예시: 행복, 슬픔, 분노) <br>
      3. 주제별 의견 분석 (Aspect-based Opinion Mining) <br>
        - 더 깊이 들어가서 사람들의 의견 분석 (맛집의 음식 맛, 서비스 퀄리티 분석) <br>

- BoW(Bag of Words) 모델<br>
  - 텍스트 문서와 같은 범주형 데이터를 수치화 시키는 것 (무슨 단어가 몇번 나왔는지 등등)<br>
  - 작동 방식<br>
    - 1. 어휘 사전 생성 (문서에 사용되는 단어 목록)<br>
    - 2. 특성 벡터 생성 (문서에 특정 단어가 몇번 나왔는지 계사하는 것)<br>
  - 활용 예시<br>
    - 1. 텍스트 분류 (스팸 감지, 뉴스 분류 등등)<br>
    - 2. 정보 검색 (특정 키워드가 포함된 문서 검색)<br>
    - 3. 문서 유사성 평가 (대학 논문 표절 감지)<br>
  - 장단점<br>
    - 장점: 구현이 간단하고 효율적<br>
    - 단점: 단어의 순서 정보를 잃어버림, 텍스트의 문맥을 제대로 이해 못할수도 있음<br>

  - 로지스틱 회귀 모델 훈련 (문서 분류)<br>
    - 결론부터<br>
      - 1. 교차 검증 정확도: 89.7%<br>
      - 2. 테스트 데이터셋 정확도: 89.9%<br>
      - 이 머신러닝 모델은 89.9% 정확도로 영화 리뷰가 긍정인지 부정인지 분류 및 예측 가능 <br>
    - 모델 훈련 방식 <br>
      - 1. 데이터셋 준비 (25,000개의 훈련 데이터셋, 25,000개의 테스트 데이터셋) <br>
      - 2. 모델 구성 및 훈련 <br>
        - TfidfVectorizer와 로지스틱 회귀를 결합한 파이프라인 구성 <br>
        - GridSearchCV를 사용하여 5-겹 곛ㅇ별 교차 검증으로 최적의 매개변수 조합 발견 <br>
    - 단점: 메모리 사용 많음 = 계산 비용 많음 (위의 예시에서도 5만개의 데이터 처리) <br>
    - 해결책: 외부 메모리 학습 <br>
      
  - 외부 메모리 학습 <br>
  - <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fgooopy.tistory.com%2F123&psig=AOvVaw05G7DXIpBZ2YVB89b_HS7z&ust=1704698646169000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCNiJirzfyoMDFQAAAAAdAAAAABAD" width="500"> <br></br> 
    - 데이터를 쪼개서 조금씩 학습시키기(?) <br>
    - 예시: 45개의 미니 배치 (각각 1000개 문서)를 사용하여 모델을 점진적으로 학습 후 마지막 5000개로 모델의 성능 평가 <br>
    - 정확도는 86.8%로 로지스틱 회귀 모델 보다 낮지만 메모리 효률적 = 가성비! <br>
    - 단점: 나쁜 데이터가 들어왔을때 한번에 줄어드는게 아니라 점진적으로 줄어듬 (변화 감지가 느리다?) <br>
      - 나쁜 데이터로 훈련 모델 수정 필요시 파악하는게 힘들수도 있음 <br>
    
  - 사용예시: 주식 거재시장 잡음 제거, 생물정보학 분야의 게놈 데이터나 유전자 발현 분석
  - 작동원리
    - 데이터 분산이 가장 큰 방향을 찾아내고 그 방향(=주성분)으로 데이터를 투영
    - 이렇게 생성된 새로운 특성 축은 서로 직각을 이루며 원본 특성 공간에서 가장 큰 분산을 가지는 방향을 나타냄
    - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/b3981cb8-e48e-417f-bbec-35ce60cb9460)
 - PCA 주요 단계
   1. 표준화 전처리: 원본 데이터셋을 표준화 (모든 특성의 중요도를 동일하게 취급하기 위해 필요)
    - <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/4bbf0df0-fe5b-429b-b435-317878c5b5f0" width="500"> <br></br>

   2. 데이터셋의 공분산 행렬 생성
    - 특성 xj와 xk사이의 공분산 계산법
    - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/71e7eec7-15ba-460b-8469-1b826c78cda0) <br></br>

   3. 고유 백터 및 고윳값 계산: 공분산 행렬을 고유 벡터와 고윳값으로 분해
    - 벡터와 스케일을 이용한 공식으로 고유 벡터와 고윳값을 직접 계산하는것도 가능하지만 넘파이의 linalg.eig함수를 사용하는게 더 간편
    - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/45c5424f-fcee-4b25-a9fc-8f067eb0d853) <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/76c332dc-003a-4a37-9abb-b8c0c9ab2fa3" width="400">
    - <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/8caec592-a347-4ba0-b96b-0701963242a9" width="500">
    - 예시에서는 첫 번째 주성분이 분산의 40%정도를 커버하고 있는 것을 볼 수 있음 (첫 2개는 60%정도 커버) <br></br>
   
   4. 고윳값 정렬 및 선택: 고윳값을 내림차순으로 정렬하고 가장 큰 고윳값을 가진 고육 벡터를 선택
    - 데이터셋 차원을 새로운 특성 부분 공간으로 압축하여 줄여야 하기에 가장 많은 정보(분산)를 가진 고유벡터를 선택
    - 설명된 분산 비율(explained variance ratio)을 통해 각 주성분이 어느정도의 분산을 커버하는지 보여줌
     - 공식이 있지만 넘파이 cumsum함수와 step함수를 사용하는게 더 쉽다~
     - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/8ef80feb-1ba0-4328-8c9f-797395cb7f93) <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/849717a1-ebde-4211-846b-7b4aeb058a99" width="400"><br></br>
</body>
 </html>