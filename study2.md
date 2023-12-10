<h2>챕터 4: 좋은 훈련 데이터 셋 만들기: 데이터 전처리</h2>
<h3>4.1.데이터 전처리의 중요성</h3>
 누락된 값 식별 <br>
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/9c93c658-10a5-43ba-ace9-cb6474eccaa9" width="300"> <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/4367273a-0714-4b63-80d9-572e187f8866" width="300"> <br>
- 판다스를 활용하여 DataFrame 작성 (NaN = Not a Number) <br>
- 작은 DataFrame은 육안으로 파악 가능하지만 큰 DataFrame은 .isnull 메서드로 쉽게 파악가능 <br></br>

누락된 값이 있는 훈련 샘플이나 특성 제외 <br>
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/12d50ea8-3f31-477c-a588-e773ac7ee5b6" width="330"> ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/2ecddd4d-f87a-439d-89e4-490a196ed174)
- axis=0: NaN이 하나라도 있는  삭제
- axis=1: NaN(Not a Number)이 하나라도 있는 열 삭제 <br></br>

<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/63fbd9dd-1841-453f-b257-0399d2bea711" width="450"> <br>
- 주의점: 너무 많은 데이터를 제거하면 제대로된 분석이 불가능할수도
  - 너무 많은 특성 영을 제거하면 클래스를 구분하는데 필요한 중요한 정보를 잃을 위험이 있다
  - 해결책: Interpolation(보간) 기법

누락된 값 대체 <br>
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/bf0df279-5713-447b-8c12-51ff6e9c9aeb" width="750"> <br>
- 보간 기법: Nan값을 mean, median, most_frequent 등등의 값으로 대체
- 사이킷런의 SimpleImputer 클래스를 사용하여 구현 가능
- Strategy에 어떤 방식을 택하는지에 따라 특성 열에서 계산한 특정 값(mean, median, most_frequent)으로 대체 <br></br>
![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/65c454e8-e096-4247-8b67-17b483a3b80d)
- 판다스 DataFrame객체에서 바로 평균값으로 대체 가능 <br></br>

사이킷런 추정기 APi익히기 <br>
- 머신 러닝 모델을 구축하고 훈련시키는 인터페이스
- fit() 메서드: 훈련 데이터에서 모델 파라미터 학습 (모델을 훈련 데이터에 맞추기)
- transform() 메서드: 학습한 파라미터로 데이터 변환 또는 차원 축소 작업 수행
- predict() 메서드: 훈련된 모델을 사용하여 새로운 데이터 예측 <br>
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/17560f0b-5c33-4661-b021-605dca6e1cc6" width="550">  <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/bb90f3c1-b85f-44bf-b51b-a04ca9fba1e4" width="300"> <br>

<h3>4.2. 범주형 데이터 다루기</h3>

- 수치형 데이터 외에도 범주형 데이터가 존재함 <br>
  - 범주형 데이터에는 수치가 있는 것과 없는 것으로 구분됨
  - 옷 사이즈 (XL > L > M > S) vs 색 (green, red, blue)

판다스를 사용한 범주형 데이터 인코딩 <br>
![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/963dc34c-ee97-447e-a7e6-6e5fb6c2d953)
- 순서가 없는 특성(color), 순서가 있는 특성(size), 수치형 특성(price), 클래스 레이블

순서가 있는 특성 매핑 <br>
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/a39ccf7c-dcf5-4ad1-ac55-ada44c6f34a4" width="600"> <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/8e77ade7-600d-4593-b14e-638cfb2c8a5c" width="600">
- 순석 특성 인식시키려면 정수로 바꾸어야 함
  - 자동으로 바꾸어주는 함수는 없다 (아직은?)

클래스 레이블 인코딩
- 클레스 레이블은 순서가 없지만 정수로 인코딩 하는것이 좋은 습관이다 (사소한 실수 방지)
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/6182de05-ac1c-43b6-aa7e-0d1156104820" width="600"> <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/46310444-d6fd-499f-b371-e385c9336c72" width="600">
- 정수로 정리하고 키-값(Key-Value)을 뒤집어서 원본 문자열로 표시 가능

순서가 없는 특성에 원-핫 인코딩 적용
- 순서가 없는 특성의 데이터(예시: color)에 정수값(머신러닝이 처리하기 위해서)을 매겼다가 학습 알고리즘이 데이터에 순위를 매기는 실수를 할 수 있다
  - 해결책: One-Hot Encoding 적용
  - 범주형 값에 0과 1로만 값을 매기는 이진특성을 사용
  - get_dummies()함수 사용하여 구현<br>
![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/a605c39a-13d8-4c10-81e5-79a3191dc2bb)

<h3>4.3 데이터셋을 훈련 데이터셋과 테스트 데이터셋으로 나누기</h3>

- 모델의 일반화 성능을 평가하기 위해 훈련 데이터셋과 테스트 데이터셋으로 나눔 <br>
- 훈련 데이터셋과 테스트 데이터셋 분할 비율 결정 (일반적으로 6:4, 7:3, 8:2로 결정)
- 대용량의 데이터인 경우, 9:1 또는 9.9:0.1로 비율 결정
- 데이터셋 랜덤 샘플링 (순서에 따른 편향 방지)

<h3>4.4. 특성 스케일 맞추기</h3>

- 특성 스케일 조정의 중요성 <br>
   - 대부분의 머신 러닝과 최적화 알고리즘은 특성 스케일이 같을때 더 나은 성능을 보임 (경사 하강법) <br>
  - 결정 트리와 랜덤 포레스트(나중에 추가 설명)는 특성 스케일 조정에 걱정할 필요가 없는 몇 안되는 머신러닝 알고리즘 <br>
- 스케일 조정에 사용되는 2가지 방법: 정규화(Normalization)와 표준화(Standardization) <br>
  - 정규화<br>
    - 특성의 스케일을 0,1 범위 맞추는 것 <br>
    - 최소-최대 스케일 변환(min-max scaling)을 사용하여 각 특성의 최소값과 최대값을 기준으로 조정 <br>
    - 최소-최대 스케일 변환 공식
  ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/3b47644d-694c-4370-8b20-2d3384d1e91c)<br>

  - 표준화 <br>
  - 특성의 평균을 0, 표준 편차를 1로 맞춰 정규 분호와 같은 특징을 가지도록 만듬
  - 이상치에 덜 밀감
  - 표준화 공식
  ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/acdb0883-481f-4ec0-9f96-aa34b00584b4)

- 사이킷런을 사용한 스케일 조정
  - MinMaxScaler는 정규화를 구현
  - StandardScaler는 표준화를 구현
  - RobustScaler는 이상치에 강한 스케일 조정
  - MaxAbsScaler는 각 특성별로 데이터를 최대 절댓값으로 나누어 스케일을 조 
<h3>4.5. 유용한 특성 선택</h3>

- 과대적합(Overfitting): 모델 복잡해서이 테스트 데이터셋보다 훈련 데이터셋에 성능이 더 높을때
  - 해결책
    - 더 많은 훈련 데이터 모집
    - 규제를 통해 복잡도 제한
    - 파라미터 개수가 적은 간단한 모델 선택
    - 데이터 차원 줄임

- L1 규제 & L2 규제
  - L2 규제: 개별 가중치 값 제한으로 모델 복잡도 제한
  ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/7bbef2b8-973d-4108-adff-f0e98cf0b406)
  - 비용 함수에 페널티 항(penalty term)을 추가하여 모델의 가중치 값을 작게 만드는데 목표를 둠
  - 규제가 없는 모델에 비해 더 작은 가중치 값을 가진 모델을 만ㄷ르어서 과대적합(OverFitting) 방지
    - <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/1038106b-50b2-4a03-b0d9-16351481bc15" width="300"> <br></br>

  - L1 규제: L2규제의 가중치 제곱을 그냥 가중치 절댓값을 바꾼것 (대부분의 가중치가 0이 되고 잡데이터가 많은 고차원 데이터셋에 효과적)
  ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/a72b3ecc-cbe0-4617-8759-fe23f145d53b)
  - 가중치의 절댓값 합에 패널티를 부과
  - 다이아몬드 모양의 제한 범위 내에서 특정 가중치를 0으로 만듦으로 모델의 희소성 증가 (불필요 특성 제거, 복잡성 감소)
    - <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/22df007a-d290-4bfc-b422-0fa576e8297f" width="300">

- 순차 특성 선택 알고리즘
  - 차원 축소 (dimensionality reduction): 규제가 없는 모델에서 유용
    - 특성 선택(feature selection)
      - 순차 특성 성택(sequential feature seleciton) 알고리즘은 탐욕적 탐색 알고리즘(greedy search algorithm) 사용
        - 초기 d차원의 특성 공간을 k차원의 특성 부분 공간으로 축소 (관련성이 높은 특성을 자동으로 선택하고 관련성이 낮은 특성이나 잡음은 제거)
      - 순차 후진 선택 알고리즘 (Sequential Backward Selection, SBS)는 전통적인 순차 특성 알고리즘
        - 모델 성능을 유지하면서 계산 효율성을 향상시키기 위해 초기 특성 집합에서 특성을 순차적으로 제
    - 특성 추출(feature extraction) <-5장에서 다룸
    
<h3>4.6. 랜덤 포레스트의 특성 중요도 사용</h3>

- 랜덤 포레스트(random forest) 알고리즘
 - 앙상블에 참여한 모든 결정 트리에서 계산한 평균적인 불순도 감소로 특성 중요도 측정 가능
 - <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/8a9faf89-26ae-4bde-a497-76057dfa09a1" width="500">
 - 사이킷런에서는 특성 중요도 값을 자동으로 계산해서 좀 더 수월하게 모델 훈련 가능
  - RandomForestClassifier 활용하여 모델 훈련후 feature_importances_를 통해 각 특성의 중요도 확인 가능
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/58159aae-20c6-4284-849b-753412c3ca2f" width="300"> ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/9eab6de9-e288-4c54-879a-0a0bd2582abf)
  - 주의점: 서로 상관관계가 높은 2개 이상의 특성이 있다면, 한 특성은 높은 중요도를 갖더라도 다른 특성은 높은 중요도를 못가질 수 있음 <br></br>

<h2>챕터 5: 차원 축소를 사용한 데이터 압축</h2>
<h3>5.1. 주성분 분석을 통한 비지도 차원 축소</h3>

- 특성 선택 vs 특성 추출
  - 특성 선택 알고리즘을 사용할때는 원본 특성을 유지하지만 특성 추출은 새로운 특성 공간으로 데이터를 변환하거나 투영함
  - 특성 추출은 저장 공간 절약, 알고리즘의 계산 효율성 향상, 찬원의 저주(차원 증가할수록 데이터 밀도 다운, 머신러닝 모델 성능 다운)문제 감소시킴으로서 예측 성능 향

- 주성분 분석 (Principal Component Analysis, PCA)
  - 비지도 선형 변환 기법으로 특성 추출과 차원 축소에 주로 사용
  - 데이터 패턴을 찾고 고차원 데이터를 더 낮은 차원의 새로운 부분 공간에 투영
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


   5. 선택된 고유벡터(들)로 투영 행렬 생성
    - 밑의 예시에서는 60%를 잡아낼 수 있는 2개의 고윳값에 해당하는 고유 벡터를 선택
    - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/0d655d12-ae11-4823-b0e3-376e2f75c29f)
    - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/d7bf40da-bda2-4b97-8fe0-bddc46598abb)
    - 변환된 데이터셋을 2차원 산점도로 시각화하여 데이터가 새로운 특성축을 따라 어떻게 퍼지는 확인 가능
    - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/76c1c1b9-196b-465b-8631-86ded13e3a8c)
     - 예시에서는 y축(두번째 주성분)보다 x축(첫번째 주성분)을 따라 더 넓게 퍼져있음
     - 추가로, plot_decision_regions함수를 사용하여 결정 경계를 볼 수 있음
       - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/96f69aaa-7e02-4b06-9bd9-6ab7659f9e87)
     - 모델이 제대로 작동할때 테스트 데이터셋에 적용했을시
       - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/9958eb67-58d4-462f-bbee-b7063a9e0244) <br></br>

   6. 차원 축소: 투영 행렬을 사용하여 원본 데이터셋을 새로운 저차원 특성 공간으로 변환


<h3>5.2. 선형 판별 분석을 통한 지도 방식의 데이터 압축</h3>

- 선형 판별 분석(Linear Discriminanat Analaysis, LDA)
 - 규제가 없는 모델에서 차원의 저주로 인한 과대적합을 줄이고 계산 효율성을 높이는 특성 추출 기법
 - 최적으로 클래스를 구분하는 특성 부분을 찾는것 (밑의 예시에서 x축으로 두 개의 정균 분포 클래스 구분 가능)
 - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/2ab3a007-96fe-4e1e-86bf-6cc5b1fbe533)

 - 주성분 분석(PCA)과 유사
   - 둘다 (PCA & LDA) 데이터셋의 찾원 개수를 줄이는 선형 변환 기

   - | 주성분 분석(PCA)              | 선형 판별 분석(LDA)                                 |
     | ---------------------------- | -------------------------------------------------- |
     | 비지도 학습 방식              | 지도 학습 방식                                      |
     | 분산이 최대인 직교 성분 찾는것 | 클래스를 최적으로 구분할 수 있는 특성 부분 공간 찾는 것 |
     | 데이터 시각화, 노이즈 제거, 특성 추출, 데이터 압축에 사용 | 분류 문제에서 차원 축소 도구로 사용 (PCA보다 분류 작업에서 더 효과적) |
     | 데이터셋의 분산을 최대로 보존해서 특징 포착 가능 | 클래스간 분산 최대화, 클래스 내 분산 최소화 |

- 선형 판별 분석 내부 동작 방식 (PCA와 유사)
   1. 표준화 전처리: 각 특성의 평균을 0, 분산을 1로 조정
   2. 평균 벡터 계산: 각 클래스에 대해 d차원의 평균 벡터를 계산 (=각 클래스별로 특성들의 평균값 계산 의미)
      - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/1339dcd2-c3e9-4499-94db-707977329014) <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/d6f3880c-dfbf-4fda-b8cf-70d44f3cf0b2" width="600">

   3. 산포 행렬 계산: 클래스 간 산포 행렬과 클래스 내 산포 행렬 구성
      - 클래스 간 산포 행렬 = 클래스 평균 간의 분산
       - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/31893e53-d4a2-41a5-8205-37b7418fa3c1)
       - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/76bce7f2-1928-4618-95b3-c8d742bd8fe0)![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/44548753-7b0b-4c62-8a2a-6035a5b8a1e1)
      - 클래스 내 산포 행렬 = 각 클래스 내의 분산
       - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/b5008223-dabc-4b9e-bd53-52add49be767)
       - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/53dd3849-9539-42cd-ae3e-9bbfb47206a0)

   4. 고유 벡터와 고윳값 계산: 산포 행렬로부터 고유 벡터와 고윳값 계산
      - 공분산 행렬에 대한 고윳값 분해(PCA)대신 행렬 Sw^-1 Sb의 고윳값 계산
      - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/a0d4c355-02b8-475f-a908-3520a4c27cce)
      - LDA에서 선형 판별 벡터는 최대 c - 1(클래스 레이블 갯수 - 1)
        - Sb(클래스 간의 산포행렬)가 랭크 1 또는 그 이하인 c개의 행렬을 합한 것이기 때문
   5. 고윳값 정렬: 고윳값을 내림차순으로 정렬하고 순서 매김 (윗 그림 참고)
   6. 변환 행렬 구성: 가장 큰 k개의 고윳값의 고유 벡터를 선택하여 d x k차원의 변환 행렬 구성 (행렬의 열 = 고유 벡터)
      - 일단 PCA와 비슷하게 각 고유 벡터/선형 판별 벡터가 커버하는 양이 얼만지 계산
      - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/ce9b88c1-fd81-42ef-991a-64ad56b9d45a)
      - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/0e756843-34c7-47b9-9bc5-d7fece6685e0)
        - 두개의 벡터가 100% 커버
      - 변환 행렬 W구현
       - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/24ca5109-921f-46c2-b018-82ea03201c40)

   7. 새로운 특성으로 공간 투영: 변환 행렬을 사용하여 샘플을 새로운 저차원의 특성 부분 공간으로 투영
      - 변환 행렬 W를 훈련 데이터셋에 곱해서 데이터 변환
      - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/0dd71663-9682-419f-acc5-86414a25cb88)
      - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/2a74d7dc-88df-45d9-bf63-81a7990f76a8)
      - PCA때와 같이 결정 경계를 볼 수 있음
        - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/abc23108-998e-4fc0-b6fe-19a272287496)
      - 모델이 제대로 작동할때 테스트 데이터셋에 적용했을시
        - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/bd16e437-2031-4334-a531-bbb2775a94e3)

<h3>5.3. 커널 PCA를 사용하여 비선형 매핑</h3>

- KCPA: PCA의 커널화 버전
 - 비선형 문제(PCA, LDA 비효율적)가 선형 문제보다 실전 어플리케이션에서 더 많기 때문에 KPCA필요
 - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/83260a9f-bdda-4d3c-ae2d-df1cc0ae3b20)
- 커널 함수들
 - 비선형 문제를 해결하기 위해 클래스가 선형으로 구분되는 고착원 특성 공간으로 투영함
 - 비선형 매핑 함수: ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/bdd2bfcc-6769-48ca-a9af-8d4273de42d8)
 - 커널트릭(Kernel trick)
   - 고차원 공간으로 변환후 표준 PCA를 사용하여 선형 분류기로 구분될 수 있는 저차원 공간으로 데이터 투영시, 계산 비용을 절약 시켜주는 방법
   - 원본 특성 공간에서 두 고차원 특성 벡터의 유사도 계산 가능
   - 자주 사용되는 커널들 (θ = 임계값, P = 사용자가 지정한 거듭제곱)
     - 다항 커널 ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/b7dd3057-fbef-4fb4-8cc7-5963f765a3d9)
     - 하이퍼볼릭 탄젠트 커널 ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/03291b8e-0566-4427-801d-9b863febfcde)
     - 방사 기저 함수(Radial Basis Function, RBF)/가우시안 커널 ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/f37b022f-dda4-4c9a-b8e0-d2fbfc669fb4)


- 파이썬으로 RBF커널 구현
 - 사이파이와 넘파이 헬퍼 함수로 쉽게 구현 가능 (참 쉽...죠?)
 - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/ffc410ba-7bc3-4970-89de-2e5aa0ab4106) ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/1d94c925-fd78-49bd-9f54-b17a352460b2)
   - 단점: 사전에 ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/135d3bef-65ca-40ff-9df1-716848da08eb) 매개변수를 지정해야 함 (적절한 값을 찾을려면 실험을 해야 함)
 - 다양한 예시들로 PCA vs KPCA 비교
   - 예시1: 반달 모양 (rbf_kernel_pca)함수를 비선형 데이터셋에 적용해서 구현
   - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/a3122efb-1660-4386-8c34-9ab1509762e2)
   - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/32090cbf-a971-44ca-87c7-c33b890262ab)
     - 첫 그래프는 수직 축을 기준으로 반전됨 (선형 분류기에 도움 안됨)
     - 두번째 그래프(첫번째 주성분만 그렸을때)는 선형적으로 구분 불가
   - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/e1111ae6-fc32-44cf-a078-e2fed8ff1a8e) <br>

   - 예시2: 동심원 분리하기
   - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/842a7513-37fb-4a19-9f36-c912881ec9c4)
   - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/63f65453-929d-44cf-92b0-cbd9508a2be1)
   - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/57f09890-f69c-4145-82dc-5dbc4c745ccb)

- 데이터셋 변환의 필요성
  - 실제 애플리케이션에서는 훈련 데이터셋 이외에도 테스트 데이터셋이나 새로 수집된 샘플을 변환해야 함
- PCA vs 커널 PCA
  - 기본 PCA는 변환 행렬과 입력과 샘플 사이의 점곱을 계산하여 데이터를 투영 (공분산에서 얻은 최상위 고유 벡터 포함)
  - 커널 PCA는 중심을 맞춘 커널 행렬의 고유 벡터를 구함 (기존 샘플은 이미 주성분 축에 투영되어있으며 새로운 샘플을 주성분 축에 투영하기 위해 별도의 계산이 필요함)
- 커널 트릭의 활용
  - 커널 PCA는 커널 트릭을 사용하여 새로운 샢믈의 명시적인 투영을 계산할 필요 없음
  - 커널 PCA는 메모리 기반 방법으로 새로운 샘플을 투영하기 위해 원본 훈련 데이터셋을 재사용
  - 훈련 데이터셋의 각 샘플과 새로운 샘플 사이의 RBF커널(유사도)을 계산하여 투영

- 사이킷런의 활용 (커널 PCA)
  - sklearn.decomposition모듈에 커널 PCA 클래스가 구현되어 있음
  - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/4eaae4eb-99ad-41fe-a1a2-6037bf6b0466) ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/cc39632f-e34a-4b18-a5f4-7b7a37e3ca7d)
  - 시각화
  - ![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/935f9155-19fb-4d7d-90d4-badfcb16b856)
  - 그외에 지역 선형 임베딩을 적용한 데이터셋, SNE를 적용한 데이터셋, 커널 PCA를 적용한 동심원 데이터셋 구현 가능







