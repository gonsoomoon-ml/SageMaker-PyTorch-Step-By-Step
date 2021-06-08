import argparse
import os
import json

from train_lib import train


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
    
    # print("os: \n", os.environ)

    #env = environment.Environment()
    #parser.add_argument("--hosts", type=list, default=env.hosts)
#     parser.add_argument("--current-host", type=str, default=env.current_host)
#    parser.add_argument("--data-dir", type=str, default=env.channel_input_dirs.get("training"))

####### 스크립트 모드
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

#     parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')


#     parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    
    # parse arguments
    args = parser.parse_args() 
    
    train(args)
    
