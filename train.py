import os
import random
from collections import Counter
from functools import reduce

import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import data.aligned_conc_dataset as dataset
import util.utils as util
from config.default_config import DefaultConfig
from config.resnet18_sunrgbd_config import RESNET18_SUNRGBD_CONFIG
from data import DataProvider
from model.trecg_model import TRecgNet

cfg = DefaultConfig()
args = {
    'resnet18': RESNET18_SUNRGBD_CONFIG().args(),
}

# Setting random seed
if cfg.MANUAL_SEED is None:
    cfg.MANUAL_SEED = random.randint(1, 10000)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

# args for different backbones
cfg.parse(args['resnet18'])

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS
device_ids = torch.cuda.device_count()
print('device_ids:', device_ids)
project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
util.mkdir('logs')

# data
train_dataset = dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_TRAIN, transform=transforms.Compose([
    dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
    dataset.RandomCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
    dataset.RandomHorizontalFlip(),
    dataset.ToTensor(),
    dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

]))

val_dataset = dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_VAL, transform=transforms.Compose([
    dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
    dataset.CenterCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
    dataset.ToTensor(),
    dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

]))
batch_size_val = cfg.BATCH_SIZE

unlabeled_loader = None
if cfg.UNLABELED:
    unlabeled_dataset = dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_UNLABELED, transform=transforms.Compose([
        dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
        dataset.RandomCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
        dataset.RandomHorizontalFlip(),
        dataset.ToTensor(),
        dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

                          )]), labeled=False)

    unlabeled_loader = DataProvider(cfg, dataset=unlabeled_dataset)

train_loader = DataProvider(cfg, dataset=train_dataset)
val_loader = DataProvider(cfg, dataset=val_dataset, batch_size=batch_size_val, shuffle=False)

# class weights
num_classes_train = list(Counter([i[1] for i in train_loader.dataset.imgs]).values())
cfg.CLASS_WEIGHTS_TRAIN = torch.FloatTensor(num_classes_train)

writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard
model = TRecgNet(cfg, writer)
model.set_data_loader(train_loader, val_loader, unlabeled_loader)

def train():

    if cfg.RESUME:
        checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
        checkpoint = torch.load(checkpoint_path)
        load_epoch = checkpoint['epoch']
        model.load_checkpoint(model.net, checkpoint_path, checkpoint, data_para=True)
        cfg.START_EPOCH = load_epoch

        if cfg.INIT_EPOCH:
            # just load pretrained parameters
            print('load checkpoint from another source')
            cfg.START_EPOCH = 1

    print('>>> task path is {0}'.format(project_name))

    # train
    model.train_parameters(cfg)

    print('save model ...')
    model_filename = '{0}_{1}_{2}.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION, cfg.NITER_TOTAL)
    model.save_checkpoint(cfg.NITER_TOTAL, model_filename)

    if writer is not None:
        writer.close()

if __name__ == '__main__':
    train()
