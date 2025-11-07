import os
import argparse
import torch.utils.data as data
from solver import Solver
from torch.backends import cudnn
from facial_paralysis_dataset import FacialParalysisDataset
import random
import torch
import numpy as np

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    # Dataset and Dataloader for facial paralysis classification
    if config.task_type == 'classification':
        if config.phase == 'test':
            trainset = None
            train_loader = None
        else:
            trainset = FacialParalysisDataset(
                data_dir=config.train_data_dir,
                phase='train',
                task_type=config.task_type,
                num_classes=config.num_classes,
                res=config.res,
                augment=config.augment
            )
            train_loader = data.DataLoader(
                trainset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True
            )
        
        testset = FacialParalysisDataset(
            data_dir=config.test_data_dir,
            phase='test',
            task_type=config.task_type,
            num_classes=config.num_classes,
            res=config.res,
            augment=False
        )
        test_loader = data.DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    else:
        # Original landmark detection mode
        from dataset import Dataset
        imgdirs_train = ['./data/afw/', './data/helen/trainset/', './data/trainset/']
        imgdirs_test_commomset = ['./data/ibug/']
        if config.phase == 'test':
            trainset = None
            train_loader = None
        else:
            trainset = Dataset(imgdirs_train, config.phase, 'train', config.rotFactor, config.res, config.gamma)
            train_loader = data.DataLoader(trainset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=config.num_workers,
                                          pin_memory=True)
        testset = Dataset(imgdirs_test_commomset, 'test', config.attr, config.rotFactor, config.res, config.gamma)
        test_loader = data.DataLoader(testset,
                                      batch_size=4,
                                      num_workers=config.num_workers,
                                      pin_memory=True)
    
    # Solver for training and testing.
    if trainset:
        print(f'Training set size: {len(trainset)}')
    print(f'Test set size: {len(testset)}')
    
    solver = Solver(train_loader, test_loader, config)
    if config.phase == 'train':
        solver.train()
    else:
        solver.load_state_dict(config.best_model)
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task configuration.
    parser.add_argument('--task_type', type=str, default='classification', 
                       choices=['classification', 'regression'],
                       help='Task type: classification for facial paralysis, regression for landmark detection')
    parser.add_argument('--num_classes', type=int, default=2, 
                       help='Number of classes (2 for binary, >2 for grading)')
    
    # Data configuration.
    parser.add_argument('--train_data_dir', type=str, default='./data/facial_paralysis/train',
                       help='Directory containing training data')
    parser.add_argument('--test_data_dir', type=str, default='./data/facial_paralysis/test',
                       help='Directory containing test data')
    
    # Training configuration.
    parser.add_argument('--nPoints', type=int, default=68, help='keypoint nums (for regression only)')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=10000, help='number of total iterations for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weightDecay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--attr', type=str, default='test', help='test, pose, blur, occlusion etc (for regression only)')
    parser.add_argument('--gamma', type=int, default=3, help='gaussian kernel (for regression only)')
   
    # Augmentation options 
    parser.add_argument('--rotFactor', type=int, default=30, help='rotation factor (in degrees, for regression only)') 
    parser.add_argument('--res', type=int, default=128, help='input resolution') 
    parser.add_argument('--augment', type=str2bool, default=True, help='use data augmentation')
    
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--best_model', default='checkpoint/models/best_checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to save best checkpoint (default: checkpoint)')

    # Directories.
    parser.add_argument('--log_dir', type=str, default='checkpoint/logs')
    parser.add_argument('--model_save_dir', type=str, default='checkpoint/models')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=2000)
    
    config = parser.parse_args()
    main(config)