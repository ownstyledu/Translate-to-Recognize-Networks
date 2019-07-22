import os
from datetime import datetime

class RESNET18_SUNRGBD_CONFIG:

    def args(self):

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        ########### Quick Setup ############
        log_path = 'summary'            # path for tensorboardX log file
        lr_schedule = 'lambda'          # lambda|step|plateau
        pretrained = 'place'
        content_pretrained = 'place'
        gpus = '0,1'                    # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size = 80
        # direction = 'AtoB'              # AtoB: RGB->Depth
        direction = 'BtoA'
        loss = ['CLS', 'SEMANTIC']      # remove 'CLS' if trained with unlabeled data
        no_upsample = False             # True for removing Decoder network
        unlabeled = False               # True for training with unlabeled data
        content_model_path = 'resnet18_places365.pth'      # places model downloaded from http://places2.csail.mit.edu/
        content_layers = '0,1,2,3,4'    # layer-wise semantic layers, you can change it to better adapt your task

        # use generated data while training
        use_fake = True
        sample_path = os.path.join('trecg_AtoB_100.pth')    # path of saved TrecgNet model for generating fake images
        resume = False
        resume_path = os.path.join('/your_saved_model_path')    # path of loading TrecgNet model


        # summary_dir_root = '/home/dudapeng/workspace/trecgnet/summary/resnet18'  # dir for saving files from tensorboardX
        # modality = 'rgb'
        # task_name = 'your_task_name'  # name for tensorboardX log file
        # len_gpu = str(len(gpus.split(',')))
        # log_path = os.path.join(summary_dir_root, 'sunrgbd', modality, content_pretrained,
        #                         ''.join([task_name, '_', lr_schedule, '_', 'gpu('+len_gpu+')'
        #                         ]), current_time)
        return {

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
            'UNLABELED': unlabeled,
            'USE_FAKE_DATA': use_fake,
            'SAMPLE_MODEL_PATH': sample_path,
            'CONTENT_MODEL_PATH': content_model_path,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,

            'NITER': 20,
            'NITER_DECAY': 80,
            'NITER_TOTAL': 100,
            'EVALUATE': True, # True if you want to check the test result after each epoch

            # translation task
            'WHICH_CONTENT_NET': 'resnet18',
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': 10
        }
