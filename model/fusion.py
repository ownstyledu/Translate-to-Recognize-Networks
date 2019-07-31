import copy
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn

import util.utils as util
from util.average_meter import AverageMeter
from . import networks
from .base_model import BaseModel


class Fusion(BaseModel):

    def __init__(self, cfg, writer):

        super(Fusion, self).__init__(cfg)
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model = 'trecg'
        self.net_rgb = networks.define_TrecgNet(cfg_tmp, upsample=False, device=self.device)
        self.net_depth = networks.define_TrecgNet(cfg_tmp, upsample=False, device=self.device)

        # load parameters
        checkpoint_path_A = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.RESUME_PATH_A)
        checkpoint_path_B = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.RESUME_PATH_B)
        super().load_checkpoint(net=self.net_rgb, checkpoint_path=checkpoint_path_A, keep_kw_module=False)
        super().load_checkpoint(net=self.net_depth, checkpoint_path=checkpoint_path_B, keep_kw_module=False)

        # fusion model
        self.net = networks.Fusion(cfg, self.net_rgb, self.net_depth)
        networks.print_network(self.net)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.net = nn.DataParallel(self.net).to(self.device)

        self.set_optimizer(cfg)
        self.set_schedulers(cfg)
        self.set_log_data(cfg)
        self.writer = writer

        if cfg.USE_FAKE_DATA:
            print('Use fake data: sample model is {0}'.format(cfg.SAMPLE_MODEL_PATH))
            print('fake ratio:', cfg.FAKE_DATA_RATE)
            cfg_tmp.USE_FAKE_DATA = False
            self.sample_model_AtoB = networks.define_TrecgNet(cfg_tmp, upsample=True, device=self.device)
            self.sample_model_BtoA = networks.define_TrecgNet(cfg_tmp, upsample=True, device=self.device)
            self.sample_model_AtoB.eval()
            self.sample_model_BtoA.eval()
            self.sample_model_AtoB = nn.DataParallel(self.sample_model_AtoB).to(self.device)
            self.sample_model_BtoA = nn.DataParallel(self.sample_model_BtoA).to(self.device)
            super().load_checkpoint(net=self.sample_model_AtoB, checkpoint_path=checkpoint_path_A)
            super().load_checkpoint(net=self.sample_model_BtoA, checkpoint_path=checkpoint_path_B)

    def set_input(self, data, d_type='pair'):

        input_A = data['A']
        input_B = data['B']
        self.img_names = data['img_name']
        self.imgs_all.extend(data['img_name'])
        self.input_rgb = input_A.to(self.device)
        self.input_depth = input_B.to(self.device)

        self.batch_size = input_A.size(0)

        if 'label' in data.keys():
            self._label = data['label']
            self.label = torch.LongTensor(self._label).to(self.device)

    def _forward(self):

        # # use fake data to train
        if self.cfg.USE_FAKE_DATA:
            with torch.no_grad():
                out_keys = self.build_output_keys(gen_img=True, cls=False)
                [self.fake_depth] = self.sample_model_AtoB(source=self.input_rgb, out_keys=out_keys)
                [self.fake_rgb] = self.sample_model_BtoA(source=self.input_depth, out_keys=out_keys)
            input_num = len(self.fake_depth)
            indexes = [i for i in range(input_num)]
            rgb_random_index = random.sample(indexes, int(len(self.fake_rgb) * self.cfg.FAKE_DATA_RATE))
            depth_random_index = random.sample(indexes, int(len(self.fake_depth) * self.cfg.FAKE_DATA_RATE))

            for i in rgb_random_index:
                self.input_rgb[i, :] = self.fake_rgb.data[i, :]
            for j in depth_random_index:
                self.input_depth[j, :] = self.fake_depth.data[j, :]

        out_keys = self.build_output_keys(gen_img=False, cls=True)
        [self.cls] = self.net(self.input_rgb, self.input_depth, label=self.label, out_keys=out_keys)

    def train_parameters(self, cfg):

        train_total_steps = 0
        train_total_iter = 0
        best_prec = 0

        for epoch in range(cfg.START_EPOCH, cfg.NITER_TOTAL + 1):

            self.phase = 'train'
            self.net.train()

            start_time = time.time()

            if cfg.LR_POLICY != 'plateau':
                self.update_learning_rate(epoch=train_total_iter)
            else:
                self.update_learning_rate(val=self.loss_meters['VAL_CLS_MEAN_ACC'].avg)

            for key in self.loss_meters:
                self.loss_meters[key].reset()

            iters = 0
            for i, data in enumerate(self.train_loader):

                self.set_input(data, self.cfg.DATA_TYPE)
                iter_start_time = time.time()
                train_total_steps += self.batch_size
                train_total_iter += 1
                iters += 1

                self._forward()
                loss = self._cal_loss(epoch)
                self._optimize(loss)

                if train_total_steps % cfg.PRINT_FREQ == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time)
                    self.print_current_errors(errors, epoch, i, t)

            print('iters in one epoch:', iters)
            print('gpu_ids:', cfg.GPU_IDS)

            self._write_loss(phase=self.phase, global_step=train_total_iter)

            train_errors = self.get_current_errors(current=False)
            print('#' * 10)
            self.print_current_errors(train_errors, epoch)

            print('Training Time: {0} sec'.format(time.time() - start_time))

            # if self.cfg.USE_FAKE_DATA:
            #     print('Fake data usage: {0} / {1}'.format(self.fake_image_num, self.train_image_num))

            # Validate cls
            if cfg.EVALUATE:

                self.imgs_all = []
                self.pred_index_all = []
                self.target_index_all = []
                self.fake_image_num = 0

                mean_acc = self.evaluate(cfg=self.cfg)

                print('Mean Acc Epoch <{epoch}> * Prec@1 <{mean_acc:.3f}> '
                      .format(epoch=epoch, mean_acc=mean_acc))

                if not cfg.INFERENCE:
                    self.loss_meters['VAL_CLS_MEAN_ACC'].update(mean_acc)
                    self._write_loss(phase=self.phase, global_step=train_total_iter)

                assert (len(self.pred_index_all) == len(self.val_loader))

                if cfg.SAVE_BEST and epoch >= total_epoch - 10:
                    # save model
                    is_best = mean_acc > best_prec
                    best_prec = max(mean_acc, best_prec)

                    if is_best:
                        # confusion matrix
                        # save_dir = os.path.join(self.save_dir, 'confusion_matrix' + '.png')
                        # plot_confusion_matrix(self.target_index_all, self.pred_index_all, save_dir,
                        #                       self.val_loader.dataset.classes)

                        model_filename = '{0}_{1}_best.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION)
                        self.save_checkpoint(epoch, model_filename)
                        print('best mean acc is {0}, epoch is {1}'.format(best_prec, epoch))

            print('End of iter {0} / {1} \t '
                  'Time Taken: {2} sec'.format(train_total_iter, cfg.NITER_TOTAL, time.time() - start_time))
            print('-' * 80)

    def _cal_loss(self, epoch=None):

        loss_total = torch.zeros(1)
        if self.use_gpu:
            loss_total = loss_total.cuda()

        cls_loss = self.criterion_cls(self.cls, self.label) * self.cfg.ALPHA_CLS
        loss_total = loss_total + cls_loss

        cls_loss = round(cls_loss.item(), 4)
        self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss, self.batch_size)

        prec1 = util.accuracy(self.cls.data, self.label, topk=(1,))
        self.loss_meters['TRAIN_CLS_ACC'].update(prec1[0].item(), self.batch_size)

        # total loss
        return loss_total

    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'VAL_CLS_LOSS',
            'TRAIN_CLS_MEAN_ACC',
            'VAL_CLS_MEAN_ACC'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def _write_loss(self, phase, global_step):

        if phase == 'train':

            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            self.writer.add_scalar('TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                   global_step=global_step)
            self.writer.add_scalar('TRAIN_CLS_MEAN_ACC', self.loss_meters['TRAIN_CLS_MEAN_ACC'].avg,
                                   global_step=global_step)

        if phase == 'test':

            if self.cfg.EVALUATE:
                self.writer.add_scalar('VAL_CLS_LOSS', self.loss_meters['VAL_CLS_LOSS'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].avg,
                                       global_step=global_step)

                self.writer.add_scalar('VAL_CLS_MEAN_ACC_FUSION', self.loss_meters['VAL_CLS_MEAN_ACC'].avg,
                                       global_step=global_step)
