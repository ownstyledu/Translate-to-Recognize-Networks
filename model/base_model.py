import os
import shutil
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class BaseModel(nn.Module):

    def name(self):
        pass

    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.gpu_ids = cfg.GPU_IDS
        num_gpu = len(cfg.GPU_IDS.split(','))
        self.use_gpu = num_gpu > 0
        self.multi_gpu = num_gpu > 1
        self.model = None
        self.device = torch.device('cuda' if self.gpu_ids else 'cpu')
        self.save_dir = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.MODEL, str(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            os.mkdir(self.save_dir)
        self.imgs_all = []

    # schedule for modifying learning rate
    def set_schedulers(self, cfg):

        if cfg.PHASE == 'train':
            self.schedulers = [self._get_scheduler(optimizer, cfg, cfg.LR_POLICY) for optimizer in self.optimizers]

    def _get_scheduler(self, optimizer, cfg, lr_policy, decay_start=None, decay_epochs=None):
        if lr_policy == 'lambda':
            print('use lambda lr')
            if decay_start is None:
                decay_start = cfg.NITER
                decay_epochs = cfg.NITER_DECAY

            def lambda_rule(epoch):
                lr_l = 1 - max(0, epoch - decay_start - 1) / float(decay_epochs)
                # if lr_l < 1:
                #     lr_l = 0.5 * lr_l
                # if epoch < decay_epochs + decay_start:
                #     lr_l = 1 - max(0, epoch - decay_start) / float(decay_epochs)
                # else:
                #     lr_l = 0.01
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            print('use step lr')
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.LR_DECAY_ITERS, gamma=0.1)
        elif lr_policy == 'plateau':
            print('use plateau lr')
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                                       threshold=0.001, factor=0.5, patience=5, eps=1e-7)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
        return scheduler

    def set_input(self, input):

        pass

    # choose labelled or unlabelled dataset
    def get_dataloader(self, cfg, epoch):
        if cfg.UNLABELED:
            print('Training with no labeled data..., training image: {0}'.format(
                len(self.unlabeled_loader.dataset.imgs)))
            dataset = self.unlabeled_loader
            self.unlabeled_flag = True
        else:
            dataset = self.train_loader
            self.unlabeled_flag = False
            print('# Training images num = {0}'.format(self.train_image_num))
            # classes = zip(self.train_loader.dataset.classes, (self.cfg.CLASS_WEIGHTS_TRAIN.cpu().numpy() / self.train_image_num) * 100)
            # print('Class weight:')
            # print(['{0}:{1}'.format(cla[0], round(cla[1])) for cla in classes])

        return dataset

    def set_data_loader(self, train_loader=None, val_loader=None, unlabeled_loader=None):

        if train_loader is not None:
            self.train_loader = train_loader
            self.train_image_num = len(train_loader.dataset.imgs)
        if val_loader is not None:
            self.val_loader = val_loader
            self.val_image_num = len(val_loader.dataset.imgs)
        if unlabeled_loader is not None:
            self.unlabeled_loader = unlabeled_loader
            self.unlabled_train_image_num = len(unlabeled_loader.dataset.imgs)

    def get_current_errors(self, current=True):

        loss_dict = OrderedDict()
        for key, value in sorted(self.loss_meters.items(), reverse=True):

            if 'TEST' in key or 'VAL' in key or 'ACC' in key or value.val == 0 or 'LAYER' in key:
                continue
            if current:
                loss_dict[key] = value.val
            else:
                loss_dict[key] = value.avg
        return loss_dict

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        pass

    def update_learning_rate(self, val=None, epoch=None):
        for scheduler in self.schedulers:
            if val is not None:
                scheduler.step(val)
            else:
                scheduler.step(epoch)

        for optimizer in self.optimizers:
            print('default lr', optimizer.defaults['lr'])
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                print('/////////learning rate = %.7f' % lr)

    def set_log_data(self, cfg):
        pass


    def print_current_errors(self, errors, epoch, i=None, t=None):
        if i is None:
            message = '(Training Loss_avg [Epoch:{0}]) '.format(epoch)
        else:
            message = '(epoch: {epoch}, iters: {iter}, time: {time:.3f}) '.format(epoch=epoch, iter=i, time=t)

        for k, v in errors.items():
            if 'CLS' in k and i is None:
                message += '{key}: [{value:.3f}] '.format(key=k, value=v)
            else:
                message += '{key}: {value:.3f} '.format(key=k, value=v)
        print(message)
