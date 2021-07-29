import argparse
import logging
import os

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

import sys
import json


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


classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def train(args):

    # devie  결정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    logger.info("###### Loading Cifar10 dataset")
    
    ####################################
    # 데이터 준비
    ####################################    
    
    # 노멀라이즈 변형기 생성
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 훈련 데이터 세트 로딩
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )
    # 훈련 데이터 로더 생성
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    
    ####################################
    # 모델 네트워크 준비
    ####################################        
    
    # 사용자 정의 모델 네트워크 로딩
    model = Net() 
    logger.info("Model network loaded from get_model_network()")    
    
    ## 코드를 DataParallel 로 실행
    model = nn.DataParallel(model)            

    # 모델을 디바이스에 할당
    model = model.to(device)

    ####################################
    # 모델 훈련 준비
    ####################################            
    
    # 로스 함수 및 옵티마이저 생성
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # 주어진 epochs  만큼 훈련 실행
    logger.info(f"Training starts until epochs of {args.epochs}")    
    for epoch in range(0, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                #break # 2000 미니 배치만 실행
    print("Training is finished")
    
    ####################################
    # 모델 아티펙트 저장 및 완료
    ####################################                
    
    return _save_model(model, args.model_dir)


def train_checkpoint(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    # 체크포인트 폴더 생성
    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    logger.info("Loading Cifar10 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    logger.info("Model loaded from get_model_network()")
    
    
    model = Net() # 사용자 모델
    

#     if torch.cuda.device_count() > 1:
#         logger.info("Gpu count: {}".format(torch.cuda.device_count()))
#         model = nn.DataParallel(model)    

    ## 코드를 DataParallel 로 실행
    model = nn.DataParallel(model)            


    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Check if checkpoints exists
    if not os.path.isfile(args.checkpoint_path + '/checkpoint.pth'):
        epoch_number = 0
    else:    
        model, optimizer, epoch_number = _load_checkpoint(model, optimizer, args)           

    for epoch in range(epoch_number, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                #break # 2000 미니 배치만 실행
                
        _save_checkpoint(model, optimizer, epoch, loss, args)                
                
    print("Finished Training")
    return _save_model(model, args.model_dir)

def _save_checkpoint(model, optimizer, epoch, loss, args):
    print("epoch: {} - loss: {}".format(epoch+1, loss))
    checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
    print("Saving the Checkpoint: {}".format(checkpointing_path))
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, checkpointing_path)

    
def _load_checkpoint(model, optimizer, args):
    print("--------------------------------------------")
    print("Checkpoint file found!")
    print("Loading Checkpoint From: {}".format(args.checkpoint_path + '/checkpoint.pth'))
    checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_number = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Checkpoint File Loaded - epoch_number: {} - loss: {}".format(epoch_number, loss))
    print('Resuming training from epoch: {}'.format(epoch_number+1))
    print("--------------------------------------------")
    return model, optimizer, epoch_number



def _save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    logger.info(f"the model is saved at {path}")    
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)








    

