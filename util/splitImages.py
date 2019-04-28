import os, shutil
from PIL import Image

## split images from train/val/..txt -> train/class1,class2 format

scene_dir = '/dataset/sun_rgbd/scene/'
data_dir = '/dataset/sun_rgbd/data/'
data_types = ['rgb/', 'hha/']
target_path = '/dataset/sun_rgbd/data_in_class/'
ext = '.png'


def sun_rgbd():
    for phase in [('train/', '19scenes_train.txt'), ('val/', '19scenes_val.txt'), ('test/', '19scenes_test.txt')]:
        for type in data_types:
            for image_name, label in read_annotation_sunrgbd(scene_dir, phase[1]):
                # class_dir = ''.join([target_path, label])
                source_image_path = os.path.join(data_dir, type, image_name, ext)
                target_folder = os.path.join(target_path, type, phase[0], label)
                target_image_path = os.path.join(target_folder, '/', image_name, ext)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                shutil.copyfile(source_image_path, target_image_path)
                print('copying {0} to {1}'.format(source_image_path, target_image_path))

def read_annotation_sunrgbd(data_dir, file_path):

    with open(os.path.join(data_dir, file_path)) as f:
        for line in f.readlines():
            [image_path, scene_label] = line.strip().split('\t')
            yield image_path, scene_label

def read_annotation_mit67(data_dir, file_path):

    with open(os.path.join(data_dir, file_path)) as f:
        for line in f.readlines():
            [scene_label, file_name] = line.strip().split('/')
            yield file_name, scene_label

def jpg2png():
    image_dir = os.path.join(data_dir, 'images')
    image_names = os.listdir(image_dir)
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        if '.jpg' in image_name:
            print('processing {0} ...'.format(image_name))
            im = Image.open(image_path)
            image_path_new = image_path.replace('.jpg', '.png')
            im.save(image_path_new)


if __name__ == '__main__':
    sun_rgbd()
    print('finishied!')
