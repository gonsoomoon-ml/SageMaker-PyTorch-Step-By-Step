# SageMaker-PyTorch-Step-By-Step


### 에러 사항
- 세이지 메이커 스크립트 모드의 로컬 모드 사용시에 framework 버전이 달리 여러번 실행한 경우에 발생 
```
framework_version = 1.6.0 시에 아래 에러 발생
framework_version = 1.7.1 시에 아래 에러 발생
 An error occurred (403) when calling the HeadObject operation: Forbidden
```
    - Workaround
        - 해당 노트북의 커널을 아직 Kill 하고 다시 실행