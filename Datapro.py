import os
import cv2 as cv
import torch
import random
import os.path as osp
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

'''
To Process data
'''
class Data_process:
    def __init__(self):
        self.img_type = ['png', 'jpg']
        self.splits = 0.7
        self.save_dir = r'/data/private/zyf/mae'
        self.apo_scren_filenames = self.get_apo_scren()
        self.bdd_img_filenames = self.get_bdd_img()
        self.carla_img_filenames = self.get_carla_img()

    def show_img(self, filenames = [], index=0, step=0):
        fig, axes = plt.subplots(figsize=(20,20), nrows = 2, ncols=2, sharey=True, sharex=True)
        imgs = [cv.imread(filenames[index + i * step]) for i in range(4)]
        imgs = np.array(imgs)
        for nums, (ax, img) in enumerate(zip(axes.flatten(), imgs)):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img)
        plt.pause(0.5)
        pass


    def get_apo_scren(self, base_path = r'/share/apolloscap/Scene-Parsing/ColorImage'): # /share/apolloscap/Scene-Parsing/ColorImage/Record006/Camera 5
        recordings = [i for i in os.listdir(base_path)]
        cameras = ['Camera 5'] #, 'Camera 6']
        filenames = []
        for recording in recordings:
            for camera in cameras:
                file_dir = osp.join(base_path, recording, camera)
                filename = [osp.join(file_dir, i) for nums, i in enumerate(os.listdir(file_dir)) if (i.split('.')[-1] in self.img_type) and (nums % 5 == 0)]
                filenames += filename
        return filenames

    def get_bdd_img(self, base_path = r'/share/bdd100k/data/bdd100k/images/100k'):
        data_types = ['train', 'val', 'test']
        filenames = []
        for data_type in data_types:
            file_dir = osp.join(base_path, data_type)
            filename = [osp.join(file_dir, i) for i in os.listdir(file_dir) if i.split('.')[-1] in self.img_type]
            filenames += filename
        return filenames

    def get_carla_img(self, base_path = r'/share/carla_seg'):
        file_types = ['train', 'test', 'val']
        base_dir = osp.join(base_path, 'images')
        filenames = []
        for file_type in file_types:
            file_dir = osp.join(base_dir, file_type)
            filename = [osp.join(file_dir, i) for nums, i in enumerate(os.listdir(file_dir)) if (i.split('.')[-1] in self.img_type) and (nums % 20 == 0)]
            filenames += filename
        return filenames

    def save_imagesets(self, data_ty = 'Train'):
        filenames = self.carla_img_filenames + self.apo_scren_filenames + self.bdd_img_filenames
        random.shuffle(filenames)
        with open(osp.join(self.save_dir, '{}.txt'.format(data_ty)), 'w') as f:
            if data_ty == 'Trains':
                filenames = filenames[0: int(len(filenames) * self.splits)]
            elif data_ty == 'Vals':
                filenames = filenames[int(len(filenames) * self.splits) : int(len(filenames) * (self.splits + 0.2))]
            elif data_ty == 'Tests':
                filenames = filenames[int(len(filenames) * (self.splits + 0.2)):]
            print('{} sets has {} files'.format(data_ty, len(filenames)))
            for filename in filenames:
                f.write(filename + '\n')

    def main(self):
        self.save_imagesets('Trains')
        self.save_imagesets('Vals')
        self.save_imagesets('Tests')


if __name__ == '__main__':
    test = Data_process()
    test.main()


