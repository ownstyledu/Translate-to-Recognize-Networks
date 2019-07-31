import os
import random
from collections import Counter
from functools import reduce

import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import data.aligned_conc_dataset as conc_dataset
import util.utils as util
from config.default_config import DefaultConfig
from config.resnet_sunrgbd_config import RESNET_SUNRGBD_CONFIG
from data import DataProvider
from model.models import create_model

cfg = DefaultConfig()
args = {
    # model should be defined as 'fusion' and set paths for RESUME_PATH_A and RESUME_PATH_B
    'resnet_sunrgbd': RESNET_SUNRGBD_CONFIG().args(),
}

# Setting random seed
if cfg.MANUAL_SEED is None:
    cfg.MANUAL_SEED = random.randint(1, 10000)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

# args for different backbones
cfg.parse(args['resnet_sunrgbd'])

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS
device_ids = torch.cuda.device_count()
print('device_ids:', device_ids)
project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
util.mkdir('logs')

train_dataset = None
val_dataset = None
unlabeled_dataset = None
train_loader = None
val_loader = None
unlabeled_loader = None

train_transforms = list()
train_transforms.append(conc_dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
train_transforms.append(conc_dataset.RandomCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)))
train_transforms.append(conc_dataset.RandomHorizontalFlip())
train_transforms.append(conc_dataset.ToTensor())
train_transforms.append(conc_dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

val_transforms = list()
val_transforms.append(conc_dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
val_transforms.append(conc_dataset.CenterCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)))
val_transforms.append(conc_dataset.ToTensor())
val_transforms.append(conc_dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

if cfg.DATA_TYPE == 'pair':

    train_dataset = conc_dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_TRAIN,
                                                    transform=transforms.Compose(train_transforms))
    val_dataset = conc_dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_VAL,
                                                  transform=transforms.Compose(val_transforms))

train_loader = DataProvider(cfg, dataset=train_dataset)
val_loader = DataProvider(cfg, dataset=val_dataset, shuffle=False)
# class weights
num_classes_train = list(Counter([i[1] for i in train_loader.dataset.imgs]).values())
cfg.CLASS_WEIGHTS_TRAIN = torch.FloatTensor(num_classes_train)

writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard
model = create_model(cfg, writer)
model.set_data_loader(train_loader, val_loader, unlabeled_loader)


def train():

    print('>>> task path is {0}'.format(project_name))

    # train
    model.train_parameters(cfg)

    print('save model ...')
    model_filename = '{0}_{1}_finish.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION)
    model.save_checkpoint(cfg.NITER_TOTAL, model_filename)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    train()
