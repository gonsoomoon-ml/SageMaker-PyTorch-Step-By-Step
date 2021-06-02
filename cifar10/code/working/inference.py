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

import subprocess

subprocess.call(['pip', 'install', 'sagemaker_inference'])
from sagemaker_inference import content_types, decoder

# src 폴더 안에 훈련 코드(예: 클래스 정의)가 있어서 경로를 Path에 추가 함

def get_model_network():
    '''
    참조: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
    '''    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    return Net



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

# def model_fn(model_dir):
#     logger.info("model_fn")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     Net = get_model_network()
#     model = Net()

#     if torch.cuda.device_count() > 1:
#         logger.info("Gpu count: {}".format(torch.cuda.device_count()))
#         model = nn.DataParallel(model)

#     with open(os.path.join(model_dir, "model.pth"), "rb") as f:
#         model.load_state_dict(torch.load(f))
#     return model.to(device)

# 파이토치 서브의 디폴트 model_fn, input_fn 코드
# https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/default_pytorch_inference_handler.py

def model_fn(model_dir):
    logger.info("--> model_dir : {}".format(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    Net = get_model_network()
    model = Net()
    
    logger.info("--> model network is loaded")    

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        
    model_file_path = os.path.join(model_dir, "model.pth")
    print("model_file_path: ", model_file_path)                      
        

        
    try:    
        with open(model_file_path, "rb") as f:
            model.load_state_dict(torch.load(f))
        logger.info("---> ####### Model is loaded #########")
    except:
        # 디버깅에만 sleep을 사용하세요.
        import time
        logger.info("---> ########## Failure loading a Model #######")
        # time.sleep(600) # 디버깅을 위해서 10분 정지            
           
        
    return model.to(device)

# Deserialize the request body
def input_fn2(request_body, request_content_type='application/x-image'):
    '''
    import io

    file_path = 'test_0.jpg'

    with open(file_path, mode='rb') as file:
        img_byte = bytearray(file.read())
        #img_byte = file.read()
    print(len(img_byte))
    # img_byte
    img_arr = np.array(Image.open(io.BytesIO(img_byte)))
    data = input_fn2(img_arr, request_content_type='application/x-npy')
    '''
    print('An input_fn that loads a image tensor')
    print(request_content_type)
    if request_content_type == 'application/x-image':             
        img = np.array(Image.open(io.BytesIO(request_body)))
        logger.info(f"img shape: {img.shape}")        
    elif request_content_type == 'application/x-npy':    
        img = request_body
        # img = np.frombuffer(request_body, dtype='uint64')
        # img = np.frombuffer(request_body)
        logger.info(f"img shape: {img.shape}")
    else:
        raise ValueError(
            'Requested unsupported ContentType in content_type : ' + request_content_type)

    img = 255 - img
    img = img[:,:,np.newaxis]
    img = np.repeat(img, 3, axis=2)    

    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tensor = test_transforms(img)

    return img_tensor         



def input_fn(input_data, content_type):
    '''
    content_type == 'application/x-npy' 일시에 토치 텐서의 변환 작업 수행
    '''
    logger.info("#### input_fn starting ######")
    logger.info(f"content_type: {content_type}")    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if isinstance(input_data, (np.ndarray)):
        np_array = input_data # 로컬에서 테스트시에 numpy.ndarray 로 제공 되는 경우
    else: # 토치 서브를 통해서 numpy.ndarray가 제공되면, IOByte( 확인 필요) 로 제공되기에 디코딩이 필요함.
        np_array = decoder.decode(input_data, content_type)
        
    
    logger.info(f"np_array shape: {np_array.shape}  ")
    
    
    tensor = torch.FloatTensor(
            np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    return tensor.to(device)    
    
def input_fn3(input_data, content_type):
    '''
    content_type == 'application/x-npy' 일시에 토치 텐서의 변환 작업 수행
    '''
    logger.info("#### input_fn starting ######")
    logger.info(f"content_type: {content_type}")    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    try:
        if content_type == 'application/x-npy':    
            # img = np.frombuffer(input_data, dtype='uint8') # 1차원 데이터로 바꿈
            logger.info(f"###### #################")
            logger.info(f"type: {type(input_data)}")
            #logger.info(f"data: {input_data}")
            
            if isinstance(input_data, (bytearray)):
                print("byte darray")
                img = np.array(Image.open(io.BytesIO(input_data)))
            else:
                img = input_data ## 차원을 보존하고 바로 리턴
            logger.info(f"######        #################")                
    

            logger.info(f"img shape: {img.shape}")
        else:

            raise ValueError(
                'Requested unsupported ContentType in content_type : ' + content_type)
    except:
            logger.info(f"Error occured when converting input_data to img")               
            logger.info(f"content_type: {content_type}")            
            raise ValueError("Error when input_data is conversioned to img data")
            


    # tensor = torch.FloatTensor(np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    try:
        tensor = torch.from_numpy(img)    
        tensor = tensor.to(device)
    except:
#             logger.info("Error when img is conversioned to torch tensor")
#             logger.info(f"content_type: {content_type}")           
            raise ValueError("Error when img is conversioned to torch tensor")            
    
    return tensor


def predict_fn(data, model):
    logger.info("#### predict_fn starting ######")    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output



# # Predicts on the deserialized object with the model from model_fn()
# def predict_fn(input_data, model):
#     logger.info('Entering the predict_fn function')
#     start_time = time.time()
#     input_data = input_data.unsqueeze(0)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()
#     input_data = input_data.to(device)
                          
#     result = {}
                                                 
#     with torch.no_grad():
#         logits = model(input_data)
#         pred_probs = F.softmax(logits, dim=1).data.squeeze()   
#         outputs = topk(pred_probs, 5)                  
#         result['score'] = outputs[0].detach().cpu().numpy()
#         result['class'] = outputs[1].detach().cpu().numpy()
    
#     print("--- Elapsed time: %s secs ---" % (time.time() - start_time))    
#     return result        




