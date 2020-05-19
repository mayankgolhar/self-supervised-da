from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import torch
import random
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

class Data_Manager(Dataset):
    def __init__(self, data_path, num_views = 7, random_seed=1000, test_split = 0, is_train = True):
        super(Data_Manager, self).__init__()
        self.data_path = data_path
        self.num_views = num_views
        self.is_train = is_train
        # self.num_folds = num_folds # Number of folds in cross validation

        if is_train:
            # TO-DO : Add more transformations
            self.transform = transforms.Compose([transforms.ColorJitter(brightness=0.2*np.random.rand(1)[0],
                                                               contrast=0.2*np.random.rand(1)[0]), 
                                        transforms.RandomAffine(degrees=(0,180), 
                                                               translate=(0.3,0.3),
                                                               scale=(0.5,2),                                                              
                                                               fillcolor=0),
                                        transforms.RandomHorizontalFlip(p=0.2),
                                        transforms.RandomVerticalFlip(p=0.2),
                                        transforms.RandomCrop(224, 224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Using Pre trained model normalisation
                                                             std=[0.229, 0.224, 0.225])])
            disp_msg = 'Train Loader :'
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomCrop(224, 224),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using pre-trained model normalisation
                                     std=[0.229, 0.224, 0.225])
            ])
            disp_msg = 'Test Loader :'

        # Create a dictionary of classes, videos and respective frames
        self.videos = []
        video_idx = 0
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
                    frame_list = []
                    label, lesion_id, mode = self.get_video_label_uah(video_name)

                    if mode == 'nbi':
                        continue

                    # Create list of all frames in the video folder
                    for frame_name in os.listdir(video_path):
                        if self.is_image_file(frame_name):
                            frame_path = os.path.join(video_path, frame_name)
                            frame_list.append(frame_path)                        

                    self.videos.append(dict(lesion_id = lesion_id,
                                            video_path = video_path,
                                            frame_list = frame_list,
                                            label = label,
                                            mode = mode,
                                            ))                    		

        #Do random shuffling of video indices
        self.vid_idxes = np.arange(0,len(self.videos))
        np.random.seed(random_seed)
        np.random.shuffle(self.vid_idxes)

        if len(self.vid_idxes):
            frames = 0
            for video_idx in self.vid_idxes:
                video = self.videos[video_idx]
                frames += len(video['frame_list'])
            print("{} total frames collected".format(frames))
        else:
            print("No images were loaded from {}".format(self.data_path))

    def __len__(self):
        return len(self.vid_idxes)

    def __getitem__(self, index):
        # Select video from the index list               
        vid_index = self.vid_idxes[index]
        vid = self.videos[vid_index]
        tot_frames = len(vid['frame_list'])
        
        # # Select equi-spaced frames from the video
        # frames = np.linspace(0, tot_frames-1, self.num_views)

        if self.is_train:
            # Select frames randomly from the video
            frames = np.random.randint(0, tot_frames, self.num_views)
            
            # Order them in temporally increasing order
            frames.sort()
        else:
            # Select equi-spaced frames from the video
            frames = np.linspace(0, tot_frames-1, self.num_views)
            frames = np.around(frames)

        n = 0
        for frame in frames:
            im = Image.open(vid['frame_list'][int(frame)])
            im = self.transform(im)
            im = im.unsqueeze(0)
            if n is 0:
                imgs = im
            else:
                imgs = torch.cat((imgs, im), 0)
            n += 1

        #get labels
        cls_label = vid['label']      
        return dict(inp_data = imgs, cls_label = cls_label)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

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


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    im_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.3),
                                        transforms.RandomVerticalFlip(p=0.3),                                        
                                        transforms.RandomCrop(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    dataset_train = Data_Manager(data_path = "D:/Research/EndoDS/UAH_CrossValidation", num_views = 7,
                       random_seed=1000, test_split = 0, is_train = True)
    data_loader_train = DataLoader(dataset_train, batch_size = 6, shuffle = True)

    n = 0
    for batch in data_loader_train:

        if n == 1:
            exit() 