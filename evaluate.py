import os
import time
from functools import reduce

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import data.aligned_conc_dataset as dataset
import util.utils as util
from config.default_config import DefaultConfig
from config.evalute_resnet18_config import EVALUATE_RESNET18_CONFIG
from data import DataProvider
from model.networks import TRecgNet_Upsample_Resiual

cfg = DefaultConfig()
args = {
    'resnet18': EVALUATE_RESNET18_CONFIG().args(),
}

# args for different backbones
cfg.parse(args['resnet18'])

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS
device_ids = torch.cuda.device_count()
print('device_ids:', device_ids)
project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
util.mkdir('logs')

val_dataset = dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_VAL, transform=transforms.Compose([
    dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
    dataset.CenterCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
    dataset.ToTensor(),
    dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

]))
batch_size_val = cfg.BATCH_SIZE

val_loader = DataProvider(cfg, dataset=val_dataset, batch_size=batch_size_val, shuffle=False)
writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard
model = TRecgNet_Upsample_Resiual(cfg, writer)
model.set_data_loader(None, val_loader, None)
model.net = nn.DataParallel(model.net).to(model.device)
model.set_log_data(cfg)

def evaluate():

    checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
    checkpoint = torch.load(checkpoint_path)
    load_epoch = checkpoint['epoch']
    model.load_checkpoint(model.net, checkpoint_path, checkpoint, data_para=False)
    cfg.START_EPOCH = load_epoch

    print('>>> task path is {0}'.format(project_name))

    model.evaluate(cfg)

    if writer is not None:
        writer.close()

if __name__ == '__main__':
    start_time = time.time()
    evaluate()
    print('time consumption: {0} secs', time.time() - start_time)
