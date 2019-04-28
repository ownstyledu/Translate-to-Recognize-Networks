import os
import socket
from datetime import datetime

class EVALUATE_RESNET18_CONFIG:

    def args(self):
        args = {'ROOT_DIR': '/home/dudapeng/workspace/trecgnet/summary/resnet18'}
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        modality = 'rgb'
        task_name = 'inference_' + \
                    'trecgnet_' + modality
        log_path = os.path.join(args['ROOT_DIR'], 'sunrgbd', modality, ''.join([task_name, ]), current_time)

        return {
            'LOG_DIR': log_path,
            # MODEL
            'ARCH': 'resnet18',
            'PRETRAINED': 'imagenet',
            'NO_UPSAMPLE': False,

            'NO_FC': False,
            'INFERENCE': True,

            # 'WHICH_DIRECTION': 'AtoB',
            'WHICH_DIRECTION': 'BtoA',  # AtoB: RGB->depth, BtoA: depth->RGB
            'GPU_IDS': '6, 7',
            'NUM_CLASSES': 19,
            'BATCH_SIZE': 180,
            'RESUME_PATH': '/home/dudapeng/workspace/trecgnet/resnet18/sample_model/' +
                           'trecg_BtoA_best.pth',

            'EVALUATE': True,
        }
