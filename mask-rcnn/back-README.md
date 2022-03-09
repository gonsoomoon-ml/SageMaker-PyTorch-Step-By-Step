# Amazon SageMaker를 사용한 분산 TensorFlow 교육

## 전제 조건
1. [AWS 계정 생성 및 활성화](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [SageMaker 서비스 제한 관리](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/)
- 노트북 인스턴스
    - SageMaker 분산 학습 알고리즘의 로컬모드 실행을 위해서 ```ml.p3.16xlarge``` 노트북 인스턴스 생성이 필요 합니다. 로컬 모드 실행이 필요하지 않을시에는 ```ml.p3.2xlarge``` 를 사용하시면 됩니다.
- 훈련용 인스턴스    
    - 훈련을 위한 2개의 ```ml.p3.16xlarge``` 및 2개의 ```ml.p3dn.24xlarge``` 인스턴스 유형의 최소 제한이 필요하지만 각 인스턴스 유형에 대해 4개의 서비스 제한이 권장됩니다. 서비스 한도는 각 AWS 리전에 따라 다릅니다. 이 튜토리얼에서는 ```us-west-2``` 리전을 사용하는 것이 좋습니다.

3. 이 자습서를 실행할 AWS 리전에 [Amazon S3 버킷]((https://docs.aws.amazon.com/en_pv/AmazonS3/latest/gsg/CreatingABucket.html))을 생성합니다. S3 버킷 이름을 저장합니다. 나중에 필요할 것입니다.


## VPN 설치 및 Fsx Luster, EFS 설치


우리는 [COCO 2017 dataset](http://cocodataset.org/#home) 을 사용할 것이고 사이즈가 약 20GB 됩니다. 

This tutorial has two key steps:

1. 우리는  [Amazon Virtual Private Network (VPC)](https://aws.amazon.com/vpc/) 안에   [Sagemaker notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/nbi.html) 를 생성 할겁니다.그러기 위해서 [Amazon CloudFormation](https://aws.amazon.com/cloudformation/) 을 사용할 겁니다.

2. VPC 안에 있는 SageMaker 노트북 인스턴스를 사용하여 분산 훈련 작업을 생성 합니다. 이때 데이터 소스는 [Amazon FSx Lustre](https://aws.amazon.com/fsx/)를 사용할 겁니다.

만약 이미 SageMaker 노트북 인스턴스를 사용하고 계신 분은 기존의 사용중인 것을 사용하고 싶을 겁니다.
새로운 SageMaker 노트북 인스턴스가 필요한 이유는 현재 SageMaker 노트북 인스턴스가 VPC에서 실행되지 않을 수 있고 또한 관련된 [IAM 역할](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/id_roles.html)이 필수 AWS 리소스에 대한 액세스를 제공하거나 필요한 [EFS 탑재 대상](https://docs.aws.amazon.com/en_pv/efs/latest/ug/accessing-fs.html)에 대한 액세스 권한이 없을 수 있기 때문 입니다.


### VPC에서 SageMaker 노트북 인스턴스 생성
이 단계의 목표는 VPC에서 SageMaker 노트북 인스턴스를 생성하는 것입니다. 두 가지 옵션이 있습니다. 새 VPC에서 SageMaker 노트북 인스턴스를 생성하거나 기존 VPC에서 노트북 인스턴스를 생성할 수 있습니다. 아래에서 두 가지 옵션을 모두 다룹니다.

#### Create SageMaker notebook instance in a new VPC

이 단계를 실행하려는 [AWS IAM User](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/id_users.html) or [AWS IAM Role](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/id_roles.html) 은 [Network Administrator](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/access_policies_job-functions.html) 유사한 권한이 있어야 합니다.


CloudFormation 템플릿 [cfn-sm.yaml](cfn-sm.yaml)을 사용하여 [CloudFormation 스택](https://docs.aws.amazon.com/en_pv/AWSCloudFormation/latest/UserGuide/stacks.html)을 생성할 수 있습니다.새 VPC에 SageMaker 노트북 인스턴스를 생성합니다.

우리는 클라우드 포메이션 콘솔에서 직접 [cfn-sm.yaml](cfn-sm.yaml) 를 이용하여 [create the CloudFormation stack](https://docs.aws.amazon.com/en_pv/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html) 을 생성 할 수 있습니다. 단 [AWS Command Line Interface (CLI)](https://docs.aws.amazon.com/en_pv/cli/latest/userguide/cli-chap-welcome.html) 가 설치된 환경에서 사용 가능합니다.
- 단 사용할 ``AWS_REGION``` 리전 and 버킷```S3_BUCKET``` 은 지정해야 합니다.


이 CloudFormation 스택을 생성하는 데 예상되는 시간은 9분입니다. 스택은 다음 AWS 리소스를 생성합니다.
   
   1. [SageMaker execution role](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/sagemaker-roles.html)
   2. [Virtual Private Network (VPC)](https://aws.amazon.com/vpc/) 안에 Internet Gateway (IGW), 1 public subnet, 3 private subnets, a NAT gateway, [Security Group](https://docs.aws.amazon.com/en_pv/vpc/latest/userguide/VPC_SecurityGroups.html), [VPC Gateway Endpoint to S3](https://docs.aws.amazon.com/en_pv/vpc/latest/userguide/vpc-endpoints-s3.html)
   3. 옵션인 [Amazon EFS](https://aws.amazon.com/efs/) 파일 시스템. 니느 VPC 안의 각 private subnet에 마운트 되어 있습니다.
   4. VPC 안에 [SageMaker Notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/nbi.html) 생성 됨
      * EFS 파일 시스템이 세이지 메이커 노트북 인스턴스에 마운트 되어 있습니다.
      * 세이지 메이커 노트북 인스턴스에 부착된 SageMaker execution role 에 필요한 권한이 팔당 되어 있습니다.
      
*스크립트의 요약 출력을 저장합니다. 나중에 필요할 것입니다. AWS Management Console의 CloudFormation 스택 출력 탭에서 출력을 볼 수도 있습니다.*
