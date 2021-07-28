# SageMaker-PyTorch-Step-By-Step

## 0. 개요
이 워크샵은 파이토치를 SageMaker 에서 단계적으로 실행하는 방법을 배웁니다.
Cifar10 이미지를 이용하여 아래와 같은 부분을 배울 수 있습니다. 이 워크샵은 아래 두가지의 예시를 참조 하였습니다.
- [PyTorch CIFAR-10 local training](https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb) 
- [pytorch_managed_spot_training_checkpointing](https://github.com/aws-samples/amazon-sagemaker-managed-spot-training/tree/main/pytorch_managed_spot_training_checkpointing)

---

## 1. 워크샵 기술 사용 부분

- 스크래치 코드 개발
    - **[세이지메이커와 전혀 상관 없이]** 스크래치 버전의 훈련 코드를 작성 합니다.
    
    
- 로컬 및 호스트 모드로 훈련
    - 스크래치 버전의 훈련 코드를 변환하여 SageMaker Script Mode를 로컬 및 세이지 메이커 호스트 모드 에서 훈련 합니다.
    
    
- 스팟 인스턴스로 훈련
    - 훈련 아티펙트의 체크 포인트를 남기어서 스팟 인스턴스로 훈련 합니다.
    
    
- Horobod로 분산 훈련


- 세이지 메이커 Distributed Data Parallel 로 훈련


- 사용자 정의의 inference code 작성
    - input_fn, model_fn, predict_fn 의 사용자 정의를 작성합니다.
    - 위 사용자 정의 함수를 로컬 노트북에서 동작 테스트 합니다.
    
    
- 로컬에서 및 세이지 메이커 호스트 모드로 앤드 포인트 생성 및 추론


--- 

## 2. 실행 가이드 및 코드 구조

### (1) 실행 가이드
- 아래 노트북의 [필수] 노트북만을 진행하셔도 되고, [필수][옵션] 을 순서대로 진행하셔도 됩니다.
- 실습시 GPU가 있는 ml.p2.xlarge 정도 혹은 이상에서 노트북 인스턴스에서 테스트를 권장 드립니다.

### (2) 코드 구조
- ./cifar10/code:
    - [필수] 0.0.Setup-Environment.ipynb
        - 필요한 파이토치 등의 패키지를 설치 합니다.
    - [필수] 1.1.Train-Scratch.ipynb
        - 세이지 메이커가 필요 없는 스크래치 버전의 훈련 코드로 훈련
    - [필수] 1.2.Train_Local_Script_Mode.ipynb
        - 세이지 메이커의 로컬 및  스크립트 모드로 훈련
    - [옵션] 1.3.Train_Spot_Checkpoint.ipynb
        - 체크 포인트를 이용한 스팟 인스턴스로 훈련
    - [옵션] 1.6.Train_Horovod.ipynb        
        - 호로 보드를 통한 분산 훈련
    - [옵션] 1.7.Train_DDP.ipynb        
        - 세이지 메이커 Distributed Data Parallel 로 훈련
    - [옵션] 2.1.Inference-Scratch.ipynb
        - 세이지 메이커의 로컬 엔드 포인트 생성
    - [필수] 2.2.Inference-SageMaker.ipynb
        - 세이지 메이커의 엔드포인트 생성


- ./cifar10/code/source:
    - model_def.py        
        - 모델 정의 파일
    - train_lib.py  
        - 훈련 코드 함수 정의
    - train.py
        - 세이지 메이커 스크립트 모드에 사용할 훈련 코드
    - train_spot.py
        - 세이지 메이커 스팟 인스턴스 훈련 코드        
    - train_horovod.py        
        - 호로보드 훈련 스크립트
    - inference.py
        - 사용자 정의 인퍼런스 코드 정의        
    - utils_cifar.py
        - 유틸리티 함수
    - requirements.txt 
        - 의존성 파이썬 패키지

---

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

- Pytorch weight 저장에 대해 우리가 알아야하는 모든 것
    - https://comlini8-8.tistory.com/50