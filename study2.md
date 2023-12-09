<h2>챕터 4 좋은 훈련 데이터 셋 만들기: 데이터 전처리</h2>
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

사이킷런 추정기 APi익히 <br>
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
  - 결정 트리와 랜덤 포레스트는 특성 스케일 조정에 걱정할 필요가 없는 몇 안되는 머신러닝 알고리즘 <br>
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
    - <img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/1038106b-50b2-4a03-b0d9-16351481bc15" width="300">

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
