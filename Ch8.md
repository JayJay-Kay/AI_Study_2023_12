<!DOCTYPE html>
<html>
<head>
  <title>챕터 8</title>
</head>

<body>
<h1>챕터 8: 감정 분석에 머신 러닝 적용</h1>
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
</body>
 </html>