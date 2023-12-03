# AI_Study_2023_12

발표팁
- 마인드 노드
- 수학 공식은 굳이 발표에 포함 시킬 필요 없음
    - 단편적으로? 파이썬언어와 같이 주요 공식과 def 및 return만? 
- 용어 정리
- 테이블도 넣어서 정리하기

- AI는 머핀과 치와와를 구분할 수 있는가
- 회귀 데이터(?)
    - 분류형 & 수치형 데이터?
- 매니폴드 러닝 (차원 축소에 관해서)

챕터 1. 컴퓨터는 데이터에서 배운다

AI 추천 언어: 파이썬
- 단순한 신텍스, 높은 가독성, 풍부한 라이브러리

핫한 인공지능 관련 용어들의 차이

<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/21239c3a-8b8e-43fb-8e84-6b254048b710" width="700">

머신러닝
- 컴퓨터를 인간처럼 학습싴켜 스스로 규칙을 형성할 수 있게 만드는 알고리즘의 과학

지도 학습 (Supervised learning)
- 래이블된 데이터를 학습시켜 규칙 형성
- 예시: 스팸 메일 필터링 (스팸인지 아닌지)

비지도 학습 (Unsupervised learning)
- 사전에 정답을 모르는 비레이블 데이터 또는 보상이 뭔지 모르는 알 수 없는 구조의 데이터를 다룸
- 클러스터링(clustering) 또는 차원축소 (dimension reduction) 활용
    - 차원 축소: 데이터를 잘 나타낼 수 있는 차원으로 떨어트린다? (쓸데 없는 데이터 제거)
    - 매니폴드 러닝
- 예시: 소비자 구매 패턴 (빵을 사면 우유도 많이 사는 패턴?) (우유옆에 빵을 두면 매출++)

강화 학습 (Reinforced learning)
- 환경과 상호작용 보상 또는 피드백으로 머신러닝 진행
- 예시: 체스게임 (환경: 체스판 상황, 보상: 게임 승리)

머신 러닝 로드맵
<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/dc11e68c-9ea4-4475-8e75-243142ebe731">

1. 전처리: 필요한 데이터만 추려서 정리
2. 학습: 알맞는 머신러닝 알고리즘 찾아서 테스스
3. 평가 & 예측: 이전에 본적 없는 데이터를 테스트하여 성능 확인 (정확도 확인)

머신러닝 적용을 위한 파이썬 기반 라이브러리
1. Numpy: 행렬이나 대규모 다차원 데이터 분석을 위한 라이브러리
2. Scipy: 기술 통계를 위한 라이브러리
3. Scikit-leran: 파이썬의 대표적인 머신러닝 라이브러리
4. Matplotlib: 데이터 분석 시각화 라이브러리
5. Panads: 행과 열을 가지는 2차원 데이터 분석을 위한 라이브러리
6. Tensorflow: 추상적으로 전체적 논리에 집중할 수 있도록 세부 구현, 함수 출력&입력값 변환 해주는 라이브러리

<br/><br/>

챕터 2. 간단한 분류 알고리즘 훈련

![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/956a4c71-6823-479c-9d4e-7ca2d91812d2)
퍼셉트론(perceptron)
- 뇌의 신경망을 흉내내는 알고리즘 (다수의 신호(input)을 받아서 하나의 신호(output)을 출력)
- 학습 데이터를 설명할 수 있는 최적의 값을 찾음
- 최종입력(Net Input)이 임계 값(θ)보다 크면 1, 아니면 0(or -1)으로 예측하는 선형 분류(linear classifier) 모형

![image](https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/c351033d-536b-4fb4-aede-dcad0c9b3fa3)

퍼셉트론 학습 방법
- 0또는 임의로 설정된 가중치(weight)로 시작
- 샘플 X에서 출력값 y계산
- 퍼셉트론 모형 분류가 잘못됐으면 weight 수정
    - 정답 맞을때까지 무식하게 계속 수정?

<img src="https://github.com/JayJay-Kay/AI_Study_2023_12/assets/110762505/d81f1028-83b4-490d-89b8-bfee88db9bac" width="420">  

<br/><br/>

적응형 선형 뉴런/아달린 (ADAptive LInear NEurone = ADALINE) 
- 퍼셉트론 업그레이드 버전
- 가중치를 업데이트하는데 단위 계단 함수 대신 선형 활성화 함수 사용
- 지속적인 에러 수정(error correction), 잡 데이터(noisy data) 관리를 해줌

경사 하강법 (Gradient descent)
- 1차 근삿값 발견용 최적화 알고리즘
- 함수의 기울기/경사/gradient를 구하고 경사의 반대 방향으로 계속 이동시켜 극값에 올때까지 반복시키는 것

<br/><br/>

챕터 3. 사이킷런을 타고 떠나는 머신 러닝 분류 모델 투어

머신러닝 알고리즘 훈련을 위한 5단계
1. 특성 선택 & 훈련 샘플 모집
2. 성능 지표 선택
3. 분류 모델 & 최적화 알고리즘 선택
4. 모델 성능 평가
5. 알고리즘 튜닝/최적화

사이킷런 라이브러리 (Scikit-learn)
- 파이썬의 대표적인 머신러닝 라이브러리
- 머신러닝을 위한 다양한 알고리즘, 프레임 워크, API 제공

로지스틱 회귀 알고리즘
- 회귀가 아닌 분류 알고리즘
- 예측 및 분류를 위해 사용 (에시: 사기/fraud 예측)
- 산업계에서 널리 사용

과대적합 & 과소적합
- 피라미터가 너무 많은 데이터에서 복잡한 모델을 만들 경우, 다른 데이터에 적용 안됨 (과대적합)
    - 규제를 통해 공산성 또는 잡음 데이터(noisy data)를 제거해서 과대적합 방지
- 데이터에서 패턴을 감지하지 못할 정도로 모델이 단순하면 편향이 커짐

커널 SVM
- 입력 데이터에서 단순한 초평명은 정의되지 않는 복잡한 모델을 만들 수 있도록 확장한 것
