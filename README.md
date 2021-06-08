# SageMaker-PyTorch-Step-By-Step
이 워크샵은 파이토치를 SageMaker 에서 단계적으로 실행하는 방법을 배웁니다.
Cifar10 이미지를 이용하여 아래와 같은 부분을 배울 수 있습니다. 이 워크샵은 [PyTorch CIFAR-10 local training](https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb) 의 기본코드를 사용을 하였고, 아래 정의된 추가적인 부분을 구현하였습니다.
]

- 세이지메이커와 전혀 상관 없이 스크래치 버전의 훈련 코드를 작성 합니다.
- 스크래치 버전의 훈련 코드를 변환하여 SageMaker Script Mode를 로컬에서 훈련 합니다.
- 스크래치 버전의 훈련 코드를 변환하여 SageMaker Script Mode를 세이지 메이커에서 훈련 합니다.
- 사용자 정의의 inference code를 작성 합니다.
    - input_fn, model_fn, predict_fn 의 사용자 정의를 작성합니다.
    - 위 사용자 정의 함수를 로컬 노트북에서 동작 테스트 합니다.
- 세이지 메이커에서 훈련된 모델 아티펙트를 로컬 엔드포인트를 생성하고 추론 합니다.
    - 로컬 엔드포인트 생성을 위한 디버깅 팁을 기술 하였습니다.
- 세이지 메이커에서 훈련된 모델 아티펙트를 세이지 메이커 엔드포인트를 생성하고 추론 합니다.    

# 코드 구조

- ./cifar10/code:
    - 0.0.Setup-Environment.ipynb
        - 필요한 파이토치 등의 패키지를 설치 합니다.
    - 1.1.Train-Scratch.ipynb
        - 세이지 메이커가 필요 없는 스크래치 버전의 훈련 코드로 훈련
    - 1.5.Train_LocalMode.ipynb
        - 세이지 메이커의 로컬 및  스크립트 모드로 훈련
    - 1.6.Train_Horovod.ipynb        
        - 호로 보드를 통한 분산 훈련
    - 2.1.Inference-Scratch.ipynb
        - 세이지 메이커의 로컬 엔드 포인트 생성
    - 2.2.Inference-SageMaker.ipynb
        - 세이지 메이커의 엔드포인트 생성


- ./cifar10/code/source:
    - model_def.py        
        - 모델 정의 파일
    - train_lib.py  
        - 훈련 코드 함수 정의
    - train.py
        - 세이지 메이커 스크립트 모드에 사용할 훈련 코드
    - train_horovod.py        
        - 호로보드 훈련 스크립트
    - inference.py
        - 사용자 정의 인퍼런스 코드 정의        
    - utils_cifar.py
        - 유틸리티 함수
    - requirements.txt 
        - 의존성 파이썬 패키지

## 참고 자료

- PyTorch CIFAR-10 local training
    - https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb


- MNIST training with PyTorch
    - https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_train.html


- PYTORCH ON AWS
    - https://torchserve-on-aws.workshop.aws/en/150.html


- Amazon SageMaker now supports PyTorch and TensorFlow 1.8
    - https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-now-supports-pytorch-and-tensorflow-1-8/


- Extending our PyTorch containers
    - https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb


- Adapting Your Own Inference Container
    - https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html


- Use PyTorch with the SageMaker Python SDK
    - https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html


- Amazon SageMaker Local Mode Examples
    - TF, Pytorch, SKLean, SKLearn Processing JOb에 대한 로컬 모드 샘플
        - https://github.com/aws-samples/amazon-sagemaker-local-mode
    - Pytorch 로컬 모드
        - https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/pytorch_script_mode_local_training_and_serving/pytorch_script_mode_local_training_and_serving.py    
