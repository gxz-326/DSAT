import torch
import os
import time
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# from resnet import resnet
from utils import rmse_batch, flip_channels, shuffle_channels_for_horizontal_flipping
from model import FAN
from Config import get_CTranS_config


class Solver(object):
    """Solver for training facial paralysis classification and grading."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Training configurations.
        self.nPoints = config.nPoints
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.lr = config.lr
        self.weightDecay = config.weightDecay
        self.resume_iters = config.resume_iters
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.phase = config.phase
        self.task_type = getattr(config, 'task_type', 'classification')  # 'classification' or 'regression'
        self.num_classes = getattr(config, 'num_classes', 2)
        self.config = get_CTranS_config()

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Set appropriate metric based on task type
        if self.task_type == 'classification':
            self.best_metric = 0.0  # Best accuracy for classification
        else:
            self.min_error = 0.08  # Best RMSE for regression
            
        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.test_step = 0
        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        self.best = 0

    def build_model(self):
        """Create network."""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # Initialize model based on task type
        if self.task_type == 'classification':
            self.model = FAN(self.config, 3, 81, num_classes=self.num_classes, 
                           task_type='classification').cuda()
            # Use CrossEntropyLoss for classification
            self.critertion = torch.nn.CrossEntropyLoss()
        else:
            # Original landmark detection
            self.model = FAN(self.config, 3, 81, task_type='regression').cuda()
            self.critertion = torch.nn.MSELoss()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2],
                                          weight_decay=self.weightDecay)
        self.print_network(self.model, 'model')
        self.model.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        # print(name)
        # print(model)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    def load_state_dict(self, path_best_model):
        self.model.load_state_dict(torch.load(path_best_model))

    def save_checkpoint(self, model_path, resume_iters):
        state = {
            'resume_iters': resume_iters,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr,
            'task_type': self.task_type,
            'num_classes': self.num_classes
        }
        
        if self.task_type == 'classification':
            state['best_metric'] = self.best_metric
        else:
            state['min_error'] = self.min_error

        torch.save(state, model_path)

    def restore_model(self):
        """Restore the trained model."""
        print('Loading the pretrained models.')
        model_path = os.path.join(self.model_save_dir, 'Checkpoint.pth.tar')
        state = torch.load(model_path)
        self.resume_iters = state['resume_iters']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        
        # Restore task-specific metrics
        if 'task_type' in state:
            self.task_type = state['task_type']
        if 'num_classes' in state:
            self.num_classes = state['num_classes']
            
        if self.task_type == 'classification':
            self.best_metric = state.get('best_metric', 0.0)
        else:
            self.min_error = state.get('min_error', 0.08)
            
        lr = state['lr']
        self.update_lr(lr)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        # from logger import Logger
        # self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rate."""
        for param_group in self.optimizer.param_groups:  # The learnable parameters of a model are returned by net.parameters()
            param_group['lr'] = lr
        self.lr = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def train(self):
        """Train network."""

        data_iter = iter(self.train_loader)
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            self.restore_model()
            start_iters = self.resume_iters
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            self.model.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            # Fetch real images and labels.
            try:
                if self.task_type == 'classification':
                    images, labels = next(data_iter)
                else:
                    images, targets, kps, tforms = next(data_iter)
            except:
                data_iter = iter(self.train_loader)
                if self.task_type == 'classification':
                    images, labels = next(data_iter)
                else:
                    images, targets, kps, tforms = next(data_iter)
                    
            images = images.to(self.device)
            
            if self.task_type == 'classification':
                labels = labels.to(self.device)
            else:
                targets = targets.to(self.device)

            # =================================================================================== #
            #                             2. Train network                                        #
            # =================================================================================== #

            out, mask = self.model(images)
            
            if self.task_type == 'classification':
                loss = self.critertion(out, labels)
            else:
                loss = self.critertion(out, targets)

            self.reset_grad()
            loss.backward()
            self.optimizer.step()
            # Logging.
            losses = {}
            losses['train_loss'] = loss.item()
            
            if self.task_type == 'classification':
                # Calculate accuracy for classification
                _, predicted = torch.max(out.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                losses['train_accuracy'] = correct / total
            else:
                # Original RMSE calculation for landmark detection
                rmse = rmse_batch(out.cpu(), kps, tforms)
                losses['train_rmse'] = np.mean(rmse)

            # =================================================================================== #
            #                             3. Miscellaneous                                        #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in losses.items():
                    log += ", {}: {:.5f}".format(tag, value)
                print(log)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                self.test()
                model_save_path = os.path.join(self.model_save_dir, 'Checkpoint.pth.tar')
                self.save_checkpoint(model_save_path, i + 1)
                print('Save model checkpoint into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0:
                lr = self.lr * 0.5
                self.update_lr(lr)
                print('Decayed learning rates, lr: {}.'.format(lr))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            # Start testing.
            print('Start testing...')
            start_time = time.time()
            
            if self.task_type == 'classification':
                # Classification evaluation
                all_predictions = []
                all_labels = []
                
                for i, (images, labels) in enumerate(self.test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    out, mask = self.model(images)
                    
                    # Get predictions
                    _, predicted = torch.max(out.data, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                
                accuracy = accuracy_score(all_labels, all_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_predictions, average='weighted')
                
                losses = {
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1
                }
                
                print(f'Test Accuracy: {accuracy:.5f}')
                print(f'Test Precision: {precision:.5f}')
                print(f'Test Recall: {recall:.5f}')
                print(f'Test F1-Score: {f1:.5f}')
                
                # Save best checkpoint
                if accuracy > self.best_metric:
                    self.best_metric = accuracy
                    print(f'Test Accuracy Best: {accuracy:.5f}')
                    str_tmp = f"{accuracy:.5f}_best.pth.tar"
                    model_save_path = os.path.join(self.model_save_dir, str_tmp)
                    print('Save best checkpoint into {}...'.format(self.model_save_dir))
                    torch.save(self.model.state_dict(), model_save_path)
                
            else:
                # Original landmark detection evaluation
                idx = 0
                Rmse = np.zeros([self.test_loader.dataset.__len__()])
                for i, (images, targets, kps, tforms) in enumerate(self.test_loader):

                    # =================================================================================== #
                    #                             1. Preprocess input data                                #
                    # =================================================================================== #
                    images = images.to(self.device)
                    #                targets = targets.to(self.device)
                    bs = images.size(0)
                    # =================================================================================== #
                    #                             2. test network                                         #
                    # =================================================================================== #
                    out1, mask = self.model(images)
                    # flip
                    images_flip = torch.from_numpy(images.cpu().numpy()[:, :, :, ::-1].copy())  # 左右翻转
                    images_flip = images_flip.to(self.device)
                    out2, mask = self.model(images_flip)
                    out2 = flip_channels(out2.cpu())
                    out2 = shuffle_channels_for_horizontal_flipping(out2)
                    out = (out1.cpu() + out2) / 2
                    # loss = self.critertion(out, targets)

                    # Logging
                    losses = {}
                    # losses['test_loss'] = loss.item()
                    rmse = rmse_batch(out, kps, tforms)
                    Rmse[idx:idx + bs] = rmse
                    idx += bs
                    losses['test_rmse'] = np.mean(rmse)
                    # =================================================================================== #
                    #                             3. Miscellaneous                                        #
                    # =================================================================================== #

                    # Print out testing information.
                    if (i + 1) % self.log_step == 0:
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, len(self.test_loader))
                        for tag, value in losses.items():
                            log += ", {}: {:.5f}".format(tag, value)
                        print(log)
                        # if self.use_tensorboard and self.phase=='train':
                        #     for tag, value in losses.items():
                        #         self.logger.scalar_summary(tag, value, self.test_step+i+1)

                mean_rmse = np.mean(Rmse)

                print('Test Inter-pupil Normalisation: {}'.format(mean_rmse))
                # save best checkpoint
                if mean_rmse < self.min_error:
                    self.min_error = mean_rmse
                    print('Test Inter-pupil Normalisation Best: {}'.format(mean_rmse))
                    str_tmp = str(mean_rmse) + ".pth.tar"
                    model_save_path = os.path.join(self.model_save_dir, str_tmp)
                    print('Save best checkpoint into {}...'.format(self.model_save_dir))
                    torch.save(self.model.state_dict(), model_save_path)

            self.test_step += len(self.test_loader)
