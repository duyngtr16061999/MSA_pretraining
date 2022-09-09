# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch
import transformers


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif optim == 'AdamW':
        optimizer = transformers.AdamW
    elif optim == 'Adafactor':
        optimizer = transformers.Adafactor
    elif 'bert' in optim:
        optimizer = 'bert'
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer

def get_scheduler(shced):
    # Bind the optimizer
    if shced == 'constant':
        print("Scheduler: constant")
        scheduler = transformers.get_constant_schedule
    elif shced == 'constant_warmup':
        print("Scheduler: constant with warmup")
        scheduler = transformers.get_constant_schedule_with_warmup
    elif shced == 'cosine_warmup':
        print("Scheduler: cosine with warmup")
        scheduler = transformers.get_cosine_schedule_with_warmup
    elif shced == 'adamax':
        print("Scheduler: cosine hard restarts with warmup")
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup
    elif shced == 'sgd':
        print("Scheduler: linear with warmup")
        scheduler = transformers.get_linear_schedule_with_warmup
    elif shced == 'sgd':
        print("Scheduler: constant with warmup")
        scheduler = transformers.get_linear_schedule_with_warmup
    else:
        assert False, "Please add your scheduler %s in the list." % shced

    return scheduler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default='iemocap,mosei,mosi')
    parser.add_argument("--valid", type=str, default='iemocap,mosei,mosi')
    parser.add_argument("--test", type=str, default=None)


    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--sched', default='constant')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1606, help='random seed')
    parser.add_argument("--accum_iter", type=int, default=-1) 
    
    parser.add_argument("--vlayers", default=3, type=int, help='Number of visual layers')
    parser.add_argument("--alayers", default=3, type=int, help='Number of audio layers.')
    parser.add_argument("--clayers", default=3, type=int, help='Number of cross layers.')
    
    parser.add_argument("--visual_feat_dim", default=35, type=int)
    parser.add_argument("--audio_feat_dim", default=74, type=int)
    
    parser.add_argument("--visual_hidden_dim", default=288, type=int)
    parser.add_argument("--visual_num_attention_heads", default=6, type=int)
    parser.add_argument("--visual_intermediate_size", default=576, type=int)
    
    parser.add_argument("--audio_hidden_dim", default=288, type=int)
    parser.add_argument("--audio_num_attention_heads", default=6, type=int)
    parser.add_argument("--audio_intermediate_size", default=576, type=int)
    
    parser.add_argument("--cross_hidden_dim", default=288, type=int)
    parser.add_argument("--cross_num_attention_heads", default=6, type=int)
    parser.add_argument("--cross_intermediate_size", default=576, type=int)
    
    parser.add_argument("--taskMatched", dest='matching_task', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskV", dest='task_mask_v', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskA", dest='task_mask_a', action='store_const', default=False, const=True)
    parser.add_argument("--taskMatchedV", dest='matching_visual', action='store_const', default=False, const=True)
    parser.add_argument("--taskMatchedA", dest='matching_audio', action='store_const', default=False, const=True)
    
    parser.add_argument('--save', type=str, default="./snap",
                        help='Save the model.')
    parser.add_argument('--save_path', type=str, default='./snap')
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--load_name', type=str, default=None,
                        help='Load the model name.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True)
    
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=8)
    parser.add_argument("--wandb_name", type=str, default='test')
    
    # Parse the arguments.
    args = parser.parse_args()

    # # Bind optimizer class.
    # args.optimizer = get_optimizer(args.optim)
    # args.scheduler = get_scheduler(args.sched)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


Args = parse_args()
