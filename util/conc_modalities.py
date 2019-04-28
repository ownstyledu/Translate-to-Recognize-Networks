import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='/dataset/sun_rgbd/data/images')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='/dataset/sun_rgbd/data/hha')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='/dataset/sun_rgbd/data_in_class/conc_data')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

splits = os.listdir(args.fold_A)

for sp in splits:
    print('processing phase {0} ...'.format(sp))
    fold_A = os.path.join(args.fold_A, sp)
    fold_B = os.path.join(args.fold_B, sp)
    catogaries = os.listdir(fold_A)
    for cato in catogaries:
        img_fold_A = os.path.join(fold_A, cato)
        img_fold_B = os.path.join(fold_B, cato)
        img_list_A = os.listdir(img_fold_A)
        img_list_B = os.listdir(img_fold_B)
        if len(img_list_A) != len(img_list_B):
            raise ValueError('number of images in A is not equal to B\'s, A\'s path is {0}'.format(img_fold_A))

        img_fold_AB = os.path.join(args.fold_AB, sp, cato)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)

        # print('split = %s, number of images = %d' % (sp, num_imgs))
        for n in range(len(img_list_A)):
            name_A = img_list_A[n]
            path_A = os.path.join(img_fold_A, name_A)
            if args.use_AB:
                name_B = name_A.replace('_A.', '_B.')
            else:
                name_B = name_A
            path_B = os.path.join(img_fold_B, name_B)
            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_AB = name_A
                if args.use_AB:
                    name_AB = name_AB.replace('_A.', '.') # remove _A
                path_AB = os.path.join(img_fold_AB, name_AB)
                im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
                im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)

print('finished!')
