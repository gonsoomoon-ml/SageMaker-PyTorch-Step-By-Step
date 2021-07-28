import argparse
import logging
import time
import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from model_def import Net

from train_lib import train_checkpoint



# def _save_model(model, model_dir):
#     print("Saving the model.")
#     path = os.path.join(model_dir, 'model.pth')
#     # recommended way from http://pytorch.org/docs/master/notes/serialization.html
#     torch.save(model.cpu().state_dict(), path)


# def _save_checkpoint(model, optimizer, epoch, loss, args):
#     print("epoch: {} - loss: {}".format(epoch+1, loss))
#     checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
#     print("Saving the Checkpoint: {}".format(checkpointing_path))
#     torch.save({
#         'epoch': epoch+1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         }, checkpointing_path)

    
# def _load_checkpoint(model, optimizer, args):
#     print("--------------------------------------------")
#     print("Checkpoint file found!")
#     print("Loading Checkpoint From: {}".format(args.checkpoint_path + '/checkpoint.pth'))
#     checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch_number = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print("Checkpoint File Loaded - epoch_number: {} - loss: {}".format(epoch_number, loss))
#     print('Resuming training from epoch: {}'.format(epoch_number+1))
#     print("--------------------------------------------")
#     return model, optimizer, epoch_number

    
# def model_fn(model_dir):
#     print('model_fn')
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = Net()
#     if torch.cuda.device_count() > 1:
#         print("Gpu count: {}".format(torch.cuda.device_count()))
#         model = nn.DataParallel(model)

#     with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
#         model.load_state_dict(torch.load(f))
#     return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints")
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train_checkpoint(parser.parse_args())
