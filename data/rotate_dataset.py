import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import random
import os
from os.path import join, dirname
import re
import pdb
# from data.dataset_utils import *

class RotateDataset(data.Dataset):
    def __init__(self, name, split='train', val_size=0, rot_classes=3,
            img_transformer=None, bias_whole_image=None):
        
        data_path = join(dirname(__file__), '..', 'datasets', name)
        test_split = 0        

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
        
        self.rot_classes = rot_classes
        self.bias_whole_image = bias_whole_image
        self._image_transformer = img_transformer

    def rotate_all(self, img):
        """Rotate for all angles"""
        img_rts = []
       
        for lb in range(self.rot_classes + 1):
            img_rt = self.rotate(img, rot=lb * 90)
            img_rts.append(img_rt)

        return img_rts

    def rotate(self, img, rot):
        if rot == 0:
            img_rt = img
        elif rot == 90:
            img_rt = img.transpose(Image.ROTATE_90)
        elif rot == 180:
            img_rt = img.transpose(Image.ROTATE_180)
        elif rot == 270:
            img_rt = img.transpose(Image.ROTATE_270)
        else:
            raise ValueError('Rotation angles should be in [0, 90, 180, 270]')
        return img_rt

    def get_image(self, index):
        frame_dict = self.data_list[index]
        framename = frame_dict['frame_path']
        label = frame_dict['label']

        img = Image.open(framename).convert('RGB')
        return img, label

    def __getitem__(self, index):
        img, label = self.get_image(index)
        rot_imgs = self.rotate_all(img)

        order = np.random.randint(self.rot_classes + 1)  # added 1 for class 0: unrotated
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0

        data = rot_imgs[order]
        data = self._image_transformer(data)
        sample = {'images': data,
                'aux_labels': int(order),
                'class_labels': label}
        return sample

    def __len__(self):
        return len(self.data_list)
    
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

# class RotateTestDataset(RotateDataset):
#     def __init__(self, *args, **xargs):
#         super().__init__(*args, **xargs)

#     def __getitem__(self, index):
#         framename = self.data_path + '/' + self.names[index]
#         img = Image.open(framename).convert('RGB')

#         sample = {'images': self._image_transformer(img),
#                 'aux_labels': 0,
#                 'class_labels': int(self.labels[index])}
#         return sample
