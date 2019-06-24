import os
import socket
from datetime import datetime

class RESNET18_SUNRGBD_CONFIG:

    def args(self):
        args = {'ROOT_DIR': '/home/dudapeng/workspace/trecgnet/summary/resnet18'}
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')

        ########### Quick Setup ############
        model = 'trecg'
        modality = 'rgb'
        task_name = 'test'
        lr_schedule = 'lambda' # lambda|step|plateau
        pretrained = 'place'
        content_pretrained = 'place'
        gpus = '4,5,6,7'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size = 160
        direction = 'AtoB'  # AtoB: RGB->Depth
        # direction = 'BtoA'
        loss = ['SEMANTIC','CLS']  # remove 'CLS' if trained with unlabeled data
                                   # more loss types could be flexibly added, e.g.,'GAN','PIX2PIX'
        no_upsample = False  # True for removing Decoder network
        unlabeld = False     # True for training with unlabeled data
        evaluate = True      # report mean acc after each epoch
        content_layers = '0,1,2,3,4' # layer-wise semantic layers, you can change it to better adapt your task

        len_gpu = str(len(gpus.split(',')))

        use_fake = False
        fake_rate = 0.3
        sample_path = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                                 'PSG_BtoA.pth')
        resume = False
        resume_path = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                     '10k_place_AtoB.pth')
        resume_path_AtoB = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                     'PS_AtoB.pth')
        resume_path_BtoA = os.path.join('/home/dudapeng/workspace/trecgnet/resnet18/sample_model/', content_pretrained,
                     'PS_BtoA.pth')


        log_path = os.path.join(args['ROOT_DIR'], 'sunrgbd', modality, content_pretrained,
                                ''.join([task_name, '_', lr_schedule, '_', 'gpus-', len_gpu
                                ]), current_time)

        return {

            'MODEL': model,
            'GPU_IDS': gpus,
            'WHICH_DIRECTION': direction,
            'BATCH_SIZE': batch_size,
            'LOSS_TYPES': loss,
            'PRETRAINED': pretrained,

            'LOG_PATH': log_path,

            # MODEL
            'ARCH': 'resnet18',
            'SAVE_BEST': True,
            'NO_UPSAMPLE': no_upsample,

            #### DATA
            'NUM_CLASSES': 19,
            'UNLABELED': unlabeld,
            'USE_FAKE_DATA': use_fake,
            'SAMPLE_MODEL_PATH': sample_path,
            'FAKE_DATA_RATE': fake_rate,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'RESUME_PATH_AtoB': resume_path_AtoB,
            'RESUME_PATH_BtoA': resume_path_BtoA,
            'LR_POLICY': lr_schedule,

            'NITER': 1600,
            'NITER_DECAY': 6400,
            'NITER_TOTAL': 8000,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,

            # translation task
            'WHICH_CONTENT_NET': 'resnet18',
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,

            'NITER_START_GAN': 20   # warm up for GAN
        }
