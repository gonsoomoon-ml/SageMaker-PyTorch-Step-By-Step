import logging, sys, os
import numpy as np

from PIL import Image
import io

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
from model_def import Net

import subprocess

subprocess.call(['pip', 'install', 'sagemaker_inference'])
from sagemaker_inference import content_types, decoder



def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    # https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  

logger = _get_logger()


# 파이토치 서브의 디폴트 model_fn, input_fn 코드
# https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/default_pytorch_inference_handler.py

import logging
def model_fn(model_dir):
    logger.info("--> model_dir : {}".format(model_dir))
    
    # 디바이스 할당
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    
    # 모델 네트워크 로딩
    model = Net()
    
    logger.info("--> model network is loaded")    

    # 모델을 파이토치 DataParallel 의 모델로 변경
    model = nn.DataParallel(model)        
        
    # 모델 아티펙트 경로       
    model_file_path = os.path.join(model_dir, "model.pth")
    print("model_file_path: ", model_file_path)                      
        

    # 모델 가중치 로딩    
    try:    
        with open(model_file_path, "rb") as f:
            model.load_state_dict(torch.load(f))
        print("####### Model is loaded #########")        

    except BaseException:
        logging.exception("An exception was thrown!")        
        logger.info("---> ########## Failure loading a Model #######")        
        # 디버깅에만 sleep을 사용하세요.
        # import time
        # time.sleep(600) # 디버깅을 위해서 10분 정지            
        
    return model.to(device)

def input_fn(input_data, content_type):
    '''
    content_type == 'application/x-npy' 일시에 토치 텐서의 변환 작업 수행
    '''
    logger.info("#### input_fn starting ######")
    logger.info(f"content_type: {content_type}")    
    
    # 디바이스 할당
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    # 입력 데이타를 numpy array 로 변경
    if isinstance(input_data, (np.ndarray)):
        np_array = input_data # 로컬에서 테스트시에 numpy.ndarray 로 제공 되는 경우
    else: # 토치 서브를 통해서 numpy.ndarray가 제공되면, IOByte( 확인 필요) 로 제공되기에 디코딩이 필요함.
        np_array = decoder.decode(input_data, content_type)
        
    
    logger.info(f"np_array shape: {np_array.shape}  ")
    
    # np.array 를 토치 텐서로 변환
    tensor = torch.FloatTensor(
            np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    

    return tensor.to(device)    
    


def predict_fn(data, model):
    '''
    모델의 추론 함수
    '''
    logger.info("#### predict_fn starting ######")    
    # 디바이스 할당
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = data.to(device)
    
    # 모델 추론
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output


