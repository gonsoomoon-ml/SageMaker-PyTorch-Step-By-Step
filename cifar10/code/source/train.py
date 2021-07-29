import argparse
import os
import json

from train_lib import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #### 사용자 정의 커맨드 인자
    ##################################
    
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, metavar="BS", help="batch size (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--dist_backend", type=str, default="gloo", help="distributed backend (default: gloo)"
    )
    
    ##################################
    #### 세이지 메이커 프레임워크의 도커 컨테이너 환경 변수 인자
    ##################################

    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])    
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])    
       
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    
    # parse arguments
    args = parser.parse_args() 
    
    ##################################
    #### 훈련 함수 콜
    ##################################
    
    train(args)
    
