import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
from data.dataset_utils import *
import os
from os.path import join, dirname
import re
import pdb
import cv2

class JigsawDataset(data.Dataset):
    def __init__(self, name, split='train', jig_classes=100,
            img_transformer=None, tile_transformer=None, patches=False, bias_whole_image=None):

        data_path = join(dirname(__file__), '..', 'datasets', name)
        test_split = 0        

        self.split = split

        if split == 'train':
            is_train = True
            disp_msg = 'Train Loader :'
        else:
            is_train = False
            disp_msg = 'Validation Loader :'

        # Create a dictionary of classes, videos and respective frames
        self.data_list = []

        # Assumed folder structure
        # UAH
        # |-SplitID (eg. Split0)
        #      |- ClassName (eg. adenoma, hyper, serr)
        #           |- ClassVideoIDMode (eg. adenoma1nbi)
        #                   |- FrameNum (eg. 0.png)
        
        # Iterate through all split folders
        for split_folder in os.listdir(data_path): 
            # Split string into text & numbers 
            # https://www.geeksforgeeks.org/python-splitting-text-and-number-in-string/
            temp = re.compile("([a-zA-Z]+)([0-9]+)") 
            split_id = temp.match(split_folder).groups()           
            split_id = int(split_id[-1])
            
            # Skip split folder if test split during training or train split during testing
            if (is_train and (split_id == test_split)) or (not is_train and (split_id != test_split)):
                continue
            
            print(disp_msg + split_folder + ' used')

            split_path = os.path.join(data_path, split_folder)

            # Iterate through all class folders
            for class_name in os.listdir(split_path):               
                # Iterate through all videos 
                class_path = os.path.join(split_path, class_name)

                for video_name in os.listdir(class_path):
                    video_path = os.path.join(class_path,video_name)

                    if not os.path.isdir(video_path):
                        continue
                    label, lesion_id, mode = self.get_video_label_uah(video_name)

                    # Create list of all frames in the video folder
                    for frame_name in os.listdir(video_path):
                        frame_path = os.path.join(video_path, frame_name)
                        self.data_list.append(dict(lesion_id = lesion_id,
                                                   frame_path = frame_path,
                                                   label = label,
                                                   mode = mode
                                                   ))

        if len(self.data_list):
            print("{} total frames collected".format(len(self.data_list)))
        else:
            print("No images were loaded from {}".format(data_path))
        
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        if patches:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):
        frame_dict = self.data_list[index]
        framename = frame_dict['frame_path']
        label = frame_dict['label']

        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), label

    def __getitem__(self, index):
        img, class_label = self.get_image(index)

        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0 or self.split != 'train':
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        sample = {'images': self.returnFunc(data),
                'aux_labels': int(order),
                'class_labels': int(class_label)}
        return sample

    def __len__(self):
        return len(self.data_list)

    def __retrieve_permutations(self, classes):
        all_perm = np.load(join(dirname(__file__), 'permutations_%d.npy' % (classes)))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def get_video_label_uah(self, filename):
        #Get class label from video's filename 
        #hyperplastic = 0, adenoma = 1, serrated = 2  
        #filename format : classFileNumberMode 
        
        #Split filename 
        #REF : https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number
        match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", filename, re.I)
        if match:
            className, fileId, mode = match.groups()
        else:
            print("Issue spilting file {}".format(filename))			

        if className == 'hyper':
            class_label = 0
        elif className == 'adenoma':
            class_label = 1
        elif className == 'serr':
            class_label = 2

        return (class_label, fileId, mode)

class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        frame_dict = self.data_list[index]
        framename = frame_dict['frame_path']
        label = frame_dict['label']
        img = Image.open(framename).convert('RGB')
        
        sample = {'images': self._image_transformer(img),
                'aux_labels': 0,
                'class_labels': int(label)}
        return sample
