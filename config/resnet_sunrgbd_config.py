import os

class RESNET_SUNRGBD_CONFIG:

    def args(self):

        ########### Quick Setup ############
        model = 'fusion'    # | fusion
        arch = 'resnet18'  # | resnet50
        content_model = 'resnet18'  # | resnet50
        pretrained = 'imagenet'        # | imagenet
        content_pretrained = 'imagenet'   # | imagenet

        gpus = '2,3'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size = 80

        log_path = 'summary'            # path for tensorboardX log file
        lr_schedule = 'lambda'          # lambda|step|plateau
        lr = 2e-4

        direction = 'AtoB'              # AtoB: RGB->Depth
        loss = ['CLS', 'SEMANTIC']      # remove 'CLS' if trained with unlabeled data
        no_upsample = False             # True for removing Decoder network
        unlabeled = False               # True for training with unlabeled data
        content_model_path = 'resnet18_places365.pth'      # places model downloaded from http://places2.csail.mit.edu/
        content_layers = '0,1,2,3,4'    # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 10              # coefficient for content loss
        fix_grad = False

        # use generated data while training
        use_fake = False
        sample_path = os.path.join('trecg_AtoB_100.pth')    # path of saved TrecgNet model for generating fake images
        resume = False
        resume_path = os.path.join('/your_saved_model_path')    # path of loading TrecgNet model

        # if we do fusion, we need two tregnets
        resume_path_A = os.path.join('/home/dudapeng/workspace/trecgnet/', arch, 'sample_model/', content_pretrained,
                     'trecg_AtoB.pth')
        resume_path_B = os.path.join('/home/dudapeng/workspace/trecgnet/', arch, 'sample_model/', content_pretrained,
                     'trecg_BtoA.pth')

        # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
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
            'MODEL': model,
            'ARCH': arch,
            'SAVE_BEST': True,
            'NO_UPSAMPLE': no_upsample,
            'FIX_GRAD': fix_grad,

            # DATA
            'NUM_CLASSES': 19,
            'UNLABELED': unlabeled,
            'USE_FAKE_DATA': use_fake,
            'SAMPLE_MODEL_PATH': sample_path,
            'CONTENT_MODEL_PATH': content_model_path,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'RESUME_PATH_A': resume_path_A,
            'RESUME_PATH_B': resume_path_B,
            'LR_POLICY': lr_schedule,
            'LR': lr,

            'NITER': 20,
            'NITER_DECAY': 80,
            'NITER_TOTAL': 100,
            'EVALUATE': True, # True if you want to check the test result after each epoch

            # TRANSLATION TASK
            'WHICH_CONTENT_NET': content_model,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content
        }
