<!DOCTYPE html>
<html>
<head>
  <title>챕터 9</title>
</head>

<body>
<h2>챕터 9: 웹 애플리케이션에 머신 러닝 모델 내장</h2>

- 직렬화의 필요성 <br>
  - 훈련된 머신러닝 모델 재사용 <br>
  - pickle모듈을 사용하여 훈련된 모델을 직렬화하여 저장 <br>
    - 저장된 파일은 pkl_objects 서브 디렉토리에 저장됨 <br>
    - 주의사항: 악성코드에 안전하지 않음으로 출처를 알수없는 데이터를 복원하는 것은 위험함 <br>
  - pickle vs joblib
      - joblib은 로지스틱 모델의 가중치 벡터와 같은 대규모(?) 넘파이 배열을 효율적으로 관리
      
- SQLite 데이터베이스 <br>
  - 오픈 소스 SQL 데이터베이스 엔진
  - 목적: 웹 애플리케이션 사용자의 피드백을 저장 및 활용하여 분류 모델 업데이트 <br>
  - 특징: 별도의 서버가 필요 없어 작은 프로젝트나 간단한 웹 애플리케이션에 적합
  - sqlite3모듈을 사용하여 SQLite 데이터베이스에 연결

- 플라스크 마이크로프레임워크 <br>
  - 다른 라이브러리들과 연결하여 쉽게 확장할 수 있음
    - 예시: HTML폼 요소 추가로 HTML사용 가능
  - conda나 pip을 사용하여 설치가능

</body>
 </html>
