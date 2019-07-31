import os
import shutil
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import util.utils as util


# some common actions are abstracted here
# customized ones could be implemented in the corresponding model
class BaseModel(nn.Module):

    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.gpu_ids = cfg.GPU_IDS
        num_gpu = len(cfg.GPU_IDS.split(','))
        self.use_gpu = num_gpu > 0
        self.multi_gpu = num_gpu > 1
        self.model = None
        self.device = torch.device('cuda' if self.gpu_ids else 'cpu')
        self.save_dir = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.MODEL,
                                     str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))))
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

    # choose labelled or unlabelled dataset for training
    def get_train_loader(self, cfg):
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

    def set_optimizer(self, cfg):

        self.optimizers = []
        # self.optimizer = torch.optim.Adam([{'params': self.net.fc.parameters(), 'lr': cfg.LR}], lr=cfg.LR / 10, betas=(0.5, 0.999))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        print('optimizer: ', self.optimizer)
        self.optimizers.append(self.optimizer)

    def _optimize(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _cal_loss(self, epoch):

        pass

    def build_output_keys(self, gen_img=True, cls=True):

        out_keys = []

        if gen_img:
            out_keys.append('gen_img')

        if cls:
            out_keys.append('cls')

        return out_keys

    def load_checkpoint(self, net=None, checkpoint_path=None, keep_kw_module=True, keep_fc=None):

        keep_fc = keep_fc if keep_fc is not None else not self.cfg.NO_FC

        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            state_model = net.state_dict()
            state_checkpoint = checkpoint['state_dict']

            # the weights of ckpt are stored when data-paralleled, remove 'module' if you
            # update the raw model with such ckpt
            if not keep_kw_module:
                new_state_dict = OrderedDict()
                for k, v in state_checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_checkpoint = new_state_dict

            if keep_fc:
                states_ckp = {k: v for k, v in state_checkpoint.items() if k in state_model}
            else:
                states_ckp = {k: v for k, v in state_checkpoint.items() if k in state_model and 'fc' not in k}

            # if successfully load weights
            assert (len(states_ckp) > 0)

            state_model.update(states_ckp)
            net.load_state_dict(state_model)
            print('load ckpt {0}'.format(checkpoint_path))
            return checkpoint

        else:
            print("=> !!! No checkpoint found at '{}'".format(checkpoint_path))
            return

    def evaluate(self, cfg):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()

        self.imgs_all = []
        self.pred_index_all = []
        self.target_index_all = []
        self.fake_image_num = 0

        with torch.no_grad():

            print('# Cls val images num = {0}'.format(self.val_image_num))
            # batch_index = int(self.val_image_num / cfg.BATCH_SIZE)
            # random_id = random.randint(0, batch_index)

            for i, data in enumerate(self.val_loader):
                self.set_input(data, self.cfg.DATA_TYPE)

                self._forward()
                self._process_fc()

                # accuracy
                prec1 = util.accuracy(self.cls.data, self.label, topk=(1,))
                self.loss_meters['VAL_CLS_ACC'].update(prec1[0].item(), self.batch_size)

        # Mean ACC
        mean_acc = self._cal_mean_acc(cfg=cfg, data_loader=self.val_loader)
        print('mean_acc: [{0}]'.format(mean_acc))
        return mean_acc

    def _process_fc(self):

        pred, self.pred_index = util.process_output(self.cls.data)

        self.pred_index_all.extend(list(self.pred_index))
        self.target_index_all.extend(list(self._label.numpy()))

    def _cal_mean_acc(self, cfg, data_loader):

        mean_acc = util.mean_acc(np.array(self.target_index_all), np.array(self.pred_index_all),
                                 cfg.NUM_CLASSES,
                                 data_loader.dataset.classes)
        return mean_acc

    def _forward(self):

        pass

