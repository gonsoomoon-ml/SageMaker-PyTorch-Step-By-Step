import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

def _metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

    # Horovod: use test_sampler to determine the number of examples in this worker's partition.
    test_loss /= len(test_loader.sampler)
    test_accuracy /= len(test_loader.sampler)

    # Horovod: average metric values across workers.
    test_loss = _metric_average(test_loss, "avg_loss")
    test_accuracy = _metric_average(test_accuracy, "avg_accuracy")

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(test_loss, 100 * test_accuracy)
    )


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def _get_train_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get train data sampler and data loader")
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),        
    )


    ######################    
    # DDP 코드
    # SageMaker data parallel: Set num_replicas and rank in DistributedSampler
    ######################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank= dist.get_rank()
    )
    


    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )

    return train_loader

    

def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data sampler and data loader")
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,        
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),        
    )
    
    ######################    
    # DDP 코드    
    # SageMaker data parallel: Set num_replicas and rank in DistributedSampler    
    ######################        
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank= dist.get_rank()
    )


    
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=test_batch_size, sampler=test_sampler, **kwargs
    )
    return test_loader


######################    
# DDP 코드: 1. 라이브러리 임포트
######################    

# SageMaker data parallel: Import the library PyTorch API
import smdistributed.dataparallel.torch.distributed as dist

# SageMaker data parallel: Import the library PyTorch DDP
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP

# SageMaker data parallel: Initialize the library
dist.init_process_group()


def main(args):
    
    ##  Horovod: initialize library ##
    # hvd.init()
    
    batch_size = args.batch_size    
    
    ######################    
    # DDP 코드 : 2. 배치 사이즈 결정       
    # SageMaker data parallel: Scale batch size by world size
    ######################        
    batch_size //= dist.get_world_size()
    batch_size = max(batch_size, 1)

    ######################    
    # DDP 코드 : 3. 각 GPU 를 DDP LIb 프로세스에 할당      
    # SageMaker data parallel: Pin each GPU to a single library process.
    ######################        

    local_rank = dist.get_local_rank()    
    torch.cuda.set_device(local_rank)    
#    model.cuda(local_rank)    
    
    ######################    
    # DDP 코드 : 4. 데이타 샘플러에 num_replicas, rank 정보 설정
    # 관련 코드는 아래 _get_train_data_loader() 정의에 있음
    # SageMaker data parallel: Set num_replicas and rank in DistributedSampler
    ######################        

    kwargs = {"num_workers": 1, "pin_memory": True}

    train_loader = _get_train_data_loader(batch_size, args.data_dir, **kwargs)
    test_loader = _get_test_data_loader(batch_size, args.data_dir, **kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    
    ######################    
    # DDP 코드 : 5. 모델을 DDP 로 감싸기 
    # SageMaker data parallel: Wrap the PyTorch model with the library's DDP
    ######################        
    
    model = DDP(Net().to(device))
    logger.info("Model loaded")    
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        if rank == 0:
            test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")

    
    logger.info("Training is finished")
    return _save_model(model, args.model_dir)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data) * args.world_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        if args.verbose:
            print("Batch", batch_idx, "from rank", args.rank)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )


    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )


    #########################
    # Container Environment
    #########################    
    
    #parser.add_argument("--data_dir", type=str, default="Data")    
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    
#    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])    
    
#    parser.add_argument("--model_dir", type=str, default="model")    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])    
    
#    parser.add_argument("--current-host", type=str, default=os.environ["HOST"])    
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    
#    parser.add_argument("--hosts", type=list, default=os.environ["HOST"])    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    
    # parse arguments
    args = parser.parse_args() 
    
    main(args)
    
##########################    



class Net(nn.Module):
    ...
    # Define model

def train(...):
    ...
    # Model training

def test(...):
    ...
    # Model evaluation

def main():
    

if __name__ == '__main__':
    main()    