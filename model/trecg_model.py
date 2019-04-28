import os
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision

import util.utils as util
from util.average_meter import AverageMeter
from util.confusion_matrix import plot_confusion_matrix
from . import networks
from .base_model import BaseModel


class TRecgNet(BaseModel):

    def __init__(self, cfg, writer=None):
        super(TRecgNet, self).__init__(cfg)

        util.mkdir(self.save_dir)
        assert (self.cfg.WHICH_DIRECTION is not None)
        self.AtoB = self.cfg.WHICH_DIRECTION == 'AtoB'
        self.modality = 'rgb' if self.AtoB else 'depth'
        self.sample_model = None
        self.phase = cfg.PHASE
        self.upsample = not cfg.NO_UPSAMPLE
        self.content_model = None
        self.content_layers = []

        self.writer = writer

        # networks
        self.use_noise = cfg.WHICH_DIRECTION == 'BtoA'
        self.net = networks.define_TrecgNet(cfg, self.use_noise, device=self.device)
        networks.print_network(self.net)


    def set_input(self, data, type='pair'):

        if type == 'pair':

            input_A = data['A']
            input_B = data['B']
            self.img_names = data['img_name']
            self.real_A = input_A.to(self.device)
            self.real_B = input_B.to(self.device)

            AtoB = self.AtoB
            self.source_modal = self.real_A if AtoB else self.real_B
            self.target_modal = self.real_B if AtoB else self.real_A
            self.source_modal_original = self.source_modal

            self.batch_size = input_A.size(0)

            if 'label' in data.keys():
                self._label = data['label']
                self.label = torch.LongTensor(self._label).to(self.device)

    def build_output_keys(self, gen_img=True, cls=True):

        out_keys = []

        if gen_img:
            out_keys.append('gen_img')

        if cls:
            out_keys.append('cls')

        return out_keys

    def _optimize(self, cfg, epoch):

        self._forward(epoch)

        self.optimizer.zero_grad()
        total_loss = self._construct_TRAIN_G_LOSS(epoch)
        total_loss.backward()
        self.optimizer.step()

    def train_parameters(self, cfg):

        assert (self.cfg.LOSS_TYPES)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.EVALUATE:
            self.criterion_cls = torch.nn.CrossEntropyLoss(self.cfg.CLASS_WEIGHTS_TRAIN.to(self.device))

        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            self.criterion_content = torch.nn.L1Loss()
            self.content_model = networks.Content_Model(cfg, self.criterion_content).to(self.device)
            assert (self.cfg.CONTENT_LAYERS)
            self.content_layers = self.cfg.CONTENT_LAYERS.split(',')

        self.set_optimizer(cfg)
        self.set_log_data(cfg)
        self.set_schedulers(cfg)
        self.net = nn.DataParallel(self.net).to(self.device)

        if cfg.USE_FAKE_DATA:
            print('Use fake data: sample model is {0}'.format(cfg.SAMPLE_MODEL_PATH))
            sample_model_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.SAMPLE_MODEL_PATH)
            checkpoint = torch.load(sample_model_path)
            model = networks.TRecgNet_Upsample_Resiual(cfg, not self.use_noise, upsample=True, device=self.device)
            self.load_checkpoint(model, sample_model_path, checkpoint, data_para=True)
            self.sample_model = nn.DataParallel(model).to(self.device)
            self.sample_model.eval()

        train_total_steps = 0
        train_total_iter = 0
        best_prec = 0

        for epoch in range(cfg.START_EPOCH, cfg.NITER_TOTAL + 1):

            self.imgs_all = []
            self.pred_index_all = []
            self.target_index_all = []

            start_time = time.time()
            data_loader = self.get_dataloader(cfg, epoch)

            if cfg.LR_POLICY != 'plateau':
                self.update_learning_rate(epoch=epoch)
            else:
                self.update_learning_rate(val=self.loss_meters['VAL_CLS_MEAN_ACC'].avg)

            self.phase = 'train'
            self.net.train()

            for key in self.loss_meters:
                self.loss_meters[key].reset()

            if self.sample_model is not None:
                self.fake_image_num = 0

            iters = 0
            for i, data in enumerate(data_loader):

                self.set_input(data, self.cfg.DATA_TYPE)
                iter_start_time = time.time()
                train_total_steps += self.batch_size
                train_total_iter += 1
                iters += 1

                self._optimize(cfg, epoch)

                if train_total_steps % cfg.PRINT_FREQ == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time)
                    self.print_current_errors(errors, epoch, i, t)

            print('iters in one epoch:', iters)

            self._write_loss(phase=self.phase, global_step=epoch)

            train_errors = self.get_current_errors(current=False)
            print('#' * 10)
            self.print_current_errors(train_errors, epoch)

            if self.sample_model is not None:
                print('Fake data usage: {0} / {1}'.format(self.fake_image_num, self.train_image_num))
            print('Training Time: {0} sec'.format(time.time() - start_time))

            # Validate cls
            if cfg.EVALUATE:

                mean_acc = self.evaluate(cfg=self.cfg, epoch=epoch)

                print('Mean Acc Epoch <{epoch}> * Prec@1 <{mean_acc:.3f}> '
                      .format(epoch=epoch, mean_acc=mean_acc))

                if not cfg.INFERENCE:
                    self.loss_meters['VAL_CLS_MEAN_ACC'].update(mean_acc)
                    self._write_loss(phase=self.phase, global_step=epoch)

                assert (len(self.pred_index_all) == len(self.val_loader))

                if cfg.SAVE_BEST and epoch >= self.cfg.NITER_TOTAL - 10:
                    # save model
                    is_best = mean_acc > best_prec
                    best_prec = max(mean_acc, best_prec)

                    if is_best:
                        # confusion matrix
                        save_dir = os.path.join(self.save_dir, 'confusion_matrix' + '.png')
                        plot_confusion_matrix(self.target_index_all, self.pred_index_all, save_dir,
                                              self.val_loader.dataset.classes)

                        model_filename = '{0}_{1}_best.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION)
                        self.save_checkpoint(epoch, model_filename)
                        print('best mean acc is {0}, epoch is {1}'.format(best_prec, epoch))

            print('End of Epoch {0} / {1} \t '
                  'Time Taken: {2} sec'.format(epoch, cfg.NITER_TOTAL, time.time() - start_time))
            print('-' * 80)

    # encoder-decoder branch
    def _forward(self, epoch=None):

        self.gen = None
        self.source_modal_show = None
        self.target_modal_show = None
        self.cls_loss = None

        if self.phase == 'train':

            # # use fake data to train
            if self.sample_model is not None:
                with torch.no_grad():
                    out_keys = self.build_output_keys(gen_img=True, cls=False)
                    [fake_source], self.loss = self.sample_model(source=self.target_modal,
                                                      out_keys=out_keys, return_losses=False)
                input_num = len(fake_source)
                index = [i for i in range(0, input_num) if np.random.uniform() > 1 - self.cfg.FAKE_DATA_RATE]
                for j in index:
                    self.source_modal[j, :] = fake_source.data[j, :]
                self.fake_image_num += len(index)

            if 'CLS' not in self.cfg.LOSS_TYPES or self.cfg.UNLABELED:

                out_keys = self.build_output_keys(gen_img=True, cls=False)
                [self.gen] = self.net(source=self.source_modal, out_keys=out_keys,
                                                 content_layers=self.content_layers)

            elif self.upsample:
                out_keys = self.build_output_keys(gen_img=True, cls=True)
                [self.gen, self.cls] = self.net(source=self.source_modal, out_keys=out_keys,
                                                           content_layers=self.content_layers)
            else:
                out_keys = self.build_output_keys(gen_img=False, cls=True)
                [self.cls] = self.net(source=self.source_modal, out_keys=out_keys)

            self.source_modal_show = self.source_modal
            self.target_modal_show = self.target_modal

        else:

            if self.upsample:

                out_keys = self.build_output_keys(gen_img=True, cls=True)
                [self.gen, self.cls]= self.net(self.source_modal, out_keys=out_keys)
                self.source_modal_show = self.source_modal
                self.target_modal_show = self.target_modal

            else:
                out_keys = self.build_output_keys(gen_img=False, cls=True)
                [self.cls] = self.net(self.source_modal, label=self.label, out_keys=out_keys)

    def _construct_TRAIN_G_LOSS(self, epoch=None):

        loss_total = torch.zeros(1)
        if self.use_gpu:
            loss_total = loss_total.cuda()

        if self.gen is not None:
            assert (self.gen.size(-1) == self.cfg.FINE_SIZE)

        if 'CLS' in self.cfg.LOSS_TYPES:
            cls_loss = self.criterion_cls(self.cls, self.label) * self.cfg.ALPHA_CLS
            loss_total = loss_total + cls_loss

            cls_loss = round(cls_loss.item(), 4)
            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss, self.batch_size)

            prec1 = util.accuracy(self.cls.data, self.label, topk=(1,))
            self.loss_meters['TRAIN_CLS_ACC'].update(prec1[0].item(), self.batch_size)

        # ) content supervised
        if self.cfg.NITER_START_CONTENT <= epoch <= self.cfg.NITER_END_CONTENT:

            if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                source_features = self.content_model((self.gen + 1) / 2, layers=self.content_layers)
                target_features = self.content_model((self.target_modal + 1) / 2, layers=self.content_layers)
                len_layers = len(self.content_layers)
                loss_fns = [self.criterion_content] * len_layers
                alpha = [1] * len_layers

                layer_wise_losses = [alpha[i] * loss_fns[i](source_feature, target_features[i])
                                     for i, source_feature in enumerate(source_features)] * self.cfg.ALPHA_CONTENT

                content_loss = sum(layer_wise_losses)
                loss_total = loss_total + content_loss

                self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss.item(), self.batch_size)

        # total loss
        return loss_total


    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'VAL_CLS_LOSS',
            'TRAIN_CLS_MEAN_ACC',
            'VAL_CLS_MEAN_ACC'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def save_checkpoint(self, epoch, filename=None):

        if filename is None:
            filename = 'TRecg2Net_{0}_{1}.pth'.format(self.cfg.WHICH_DIRECTION, epoch)

        net_state_dict = self.net.state_dict()
        save_state_dict = {}
        for k, v in net_state_dict.items():
            if 'content_model' in k:
                continue
            save_state_dict[k] = v

        state = {
            'epoch': epoch,
            'state_dict': save_state_dict,
            'optimizer': self.optimizer.state_dict(),
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, net, checkpoint_path, checkpoint, optimizer=None, data_para=True):

        keep_fc = not self.cfg.NO_FC

        if os.path.isfile(checkpoint_path):

            state_dict = net.state_dict()
            state_checkpoint = checkpoint['state_dict']
            if data_para:
                new_state_dict = OrderedDict()
                for k, v in state_checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_checkpoint = new_state_dict

            if keep_fc:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict}
            else:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict and 'fc' not in k}

            state_dict.update(pretrained_G)
            net.load_state_dict(state_dict)

            if self.phase == 'train' and not self.cfg.INIT_EPOCH:
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> !!! No checkpoint found at '{}'".format(self.cfg.RESUME))
            return

    def set_optimizer(self, cfg):

        self.optimizers = []
        # self.optimizer = torch.optim.Adam([{'params': self.net.fc.parameters(), 'lr': cfg.LR}], lr=cfg.LR / 10, betas=(0.5, 0.999))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        print('optimizer: ', self.optimizer)
        self.optimizers.append(self.optimizer)

    def evaluate(self, cfg, epoch=None):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()

        self.imgs_all = []
        self.pred_index_all = []
        self.target_index_all = []

        with torch.no_grad():

            print('# Cls val images num = {0}'.format(self.val_image_num))

            for i, data in enumerate(self.val_loader):
                self.set_input(data, self.cfg.DATA_TYPE)

                self._forward()
                self._process_fc()

                if not cfg.INFERENCE:
                    # loss
                    cls_loss = self.criterion_cls(self.cls, self.label) * self.cfg.ALPHA_CLS
                    self.loss_meters['VAL_CLS_LOSS'].update(round(cls_loss.item(), 4), self.batch_size)

                # accuracy
                prec1 = util.accuracy(self.cls.data, self.label, topk=(1,))
                self.loss_meters['VAL_CLS_ACC'].update(prec1[0].item(), self.batch_size)

        # Mean ACC
        mean_acc = self._cal_mean_acc(cfg=cfg, data_loader=self.val_loader)
        print('mean_acc:', mean_acc)
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

    def _write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES

        if phase == 'train':

            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('TRAIN_CLS_MEAN_ACC', self.loss_meters['TRAIN_CLS_MEAN_ACC'].avg,
                                       global_step=global_step)

            if 'SEMANTIC' in loss_types:
                self.writer.add_scalar('TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                                       global_step=global_step)

            if self.upsample and self.gen is not None and not self.cfg.NO_VIS:
                self.writer.add_image('Train_Source',
                                      torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image('Train_Gen', torchvision.utils.make_grid(self.gen[:6].clone().cpu().data, 3,
                                                                                 normalize=True),
                                      global_step=global_step)
                self.writer.add_image('Train_Target',
                                      torchvision.utils.make_grid(self.target_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

        if phase == 'test':

            if self.cfg.EVALUATE and self.cfg.CAL_LOSS:
                self.writer.add_scalar('VAL_CLS_LOSS', self.loss_meters['VAL_CLS_LOSS'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('VAL_CLS_MEAN_ACC', self.loss_meters['VAL_CLS_MEAN_ACC'].avg,
                                       global_step=global_step)

            if self.upsample and self.gen is not None and not self.cfg.NO_VIS:
                self.writer.add_image('Val_Source',
                                      torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image('Val_Gen', torchvision.utils.make_grid(self.gen[:6].clone().cpu().data, 3,
                                                                               normalize=True), global_step=global_step)
                self.writer.add_image('Val_Target',
                                          torchvision.utils.make_grid(self.target_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
