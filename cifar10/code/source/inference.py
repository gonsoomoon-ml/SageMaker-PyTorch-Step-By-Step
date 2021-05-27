import logging, sys, os
import numpy as np

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



def model_fn(model_dir):
    logger.info("--> model_dir : {}".format(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    Net = get_model_network()
    model = Net()

    
    model = Net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        
    model_file_path = os.path.join(model_dir, "model.pth")
    print("model_file_path: ", model_file_path)                      
        
    with open(model_file_path, "rb") as f:
        model.load_state_dict(torch.load(f))
    logger.info("---> Model is loaded")
        
    return model.to(device)


def input_fn(input_data, content_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    np_array = encoders.decode(input_data, content_type)
#     np_array = encoder.decode(input_data, content_type)    
    if content_type == 'application/x-image':             
        img = np.array(Image.open(io.BytesIO(input_data)))
    elif content_type == 'application/x-npy':    
        img = np.frombuffer(input_data, dtype='uint8')
    else:
        raise ValueError(
            'Requested unsupported ContentType in content_type : ' + content_type)


    # tensor = torch.FloatTensor(np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    tensor =  torch.from_numpy(img)    
    return tensor.to(device)


def predict_fn(data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output


# 대근님 소스

# # Deserialize the request body
# def input_fn(request_body, request_content_type='application/x-image'):
#     print('An input_fn that loads a image tensor')
#     print(request_content_type)
#     if request_content_type == 'application/x-image':             
#         img = np.array(Image.open(io.BytesIO(request_body)))
#     elif request_content_type == 'application/x-npy':    
#         img = np.frombuffer(request_body, dtype='uint8').reshape(137, 236)   
#     else:
#         raise ValueError(
#             'Requested unsupported ContentType in content_type : ' + request_content_type)

#     img = 255 - img
#     img = img[:,:,np.newaxis]
#     img = np.repeat(img, 3, axis=2)    

#     test_transforms = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     img_tensor = test_transforms(img)

#     return img_tensor         



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




