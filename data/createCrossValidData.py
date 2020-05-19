#### Create k-fold cross validation split of data 
# 1) Crawl through all the videos and make a class wise dictionary.
# 2) Assign split labels to each video within each class
# 3) Subsample n frames from each polyp video 

# Create a dictionary of videos and respective frames
from random import sample
import os
import shutil
import re
import math
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

if __name__ == "__main__":
    frames_per_video = 3
    k = 5
    data_path = 'D:/Research/EndoDS/UAH_256/UAH_256/UAH_AutoCrop_256/'
    out_path = '../datasets/UAH_CrossValidation/'
    frameSamplingRate = 5 #Sample every 5th frame
    rand_seed = 4949

    ds_dict = {}
    classNameList = []
    total_frames = 0
    # 1) Crawl through all the videos and make a class wise dictionary.
    for video_name in os.listdir(data_path):
        video_path = os.path.join(data_path, video_name)
        if not os.path.isdir(video_path):
            continue
       
        # get class name
        match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", video_name, re.I)
        if match:
            className, lesionId, mode = match.groups()
        else:
            print("Issue spliting file name {}".format(video_name))	
        
        if className not in ds_dict:
            ds_dict[className] = {}
            classNameList.append(className)

        # Create frame list for video
        frame_list = []
        for frame_name in os.listdir(video_path):
            if is_image_file(frame_name):
                frame_path = os.path.join(video_path, frame_name)
                frame_list.append(frame_path)

        total_frames += len(frame_list)

        if lesionId not in ds_dict[className]:
            ds_dict[className][lesionId] = {}

        ds_dict[className][lesionId][mode] = frame_list

    print('Total number of images : {}'.format(total_frames))

    # Assign split labels to each video within each class
    np.random.seed(rand_seed)
    for className in ds_dict:
        classLesion = ds_dict[className]
        lesionIdLst = [int(x) for x in classLesion.keys()]
        np.random.shuffle(lesionIdLst)
        splitLabels = [i%k for i in range(len(classLesion))]
        
        for i in range(len(classLesion)):
            classLesion[str(lesionIdLst[i])]['splitLabel'] = splitLabels[i]

    # Create split folders
    for splitIdx in range(k):
        splitFoldPath = os.path.join(out_path, 'Split'+str(splitIdx))
        if not os.path.exists(splitFoldPath):
            os.mkdir(splitFoldPath)
        for className in classNameList:
            classFoldPath = os.path.join(splitFoldPath, className)
            if not os.path.exists(classFoldPath):
                os.mkdir(classFoldPath)

    # Sample data from each video and put in folder
    for className in ds_dict:
        for lesionId in ds_dict[className]:
            # randomly select n number of frames from both the modes
            for mode in ['nbi', 'wl']:
                frameList = ds_dict[className][lesionId][mode]
                if len(frameList)<frames_per_video:
                    print(str(lesionId)+mode+' : Does not have enough frames')
                # # Sample randomly
                # frameIdxLst = sample(range(len(frameList)), frames_per_video)

                # # Sample uniformly from video
                # frameIdxLst = np.floor(np.linspace(0, len(frameList)-1, frames_per_video+2))
                # frameIdxLst = frameIdxLst[1:-1].astype(int)

                frameIdxLst = range(0, len(frameList), frameSamplingRate)

                splitLabel = ds_dict[className][lesionId]['splitLabel']

                # Create folder for the video
                video_name = className + lesionId + mode
                video_folder_path = os.path.join(out_path, 'Split'+str(splitLabel), className, video_name)
                if not os.path.exists(video_folder_path):
                    os.mkdir(video_folder_path)

                for frameIdx in frameIdxLst:
                    frame_src_path = frameList[frameIdx]
                    out_file_name = os.path.basename(frame_src_path)
                    frame_dest_path = os.path.join(video_folder_path, out_file_name)
                    
                    try:
                        shutil.copy(frame_src_path, frame_dest_path)
                    except IOError as e:
                        print("Unable to copy file. %s" % e)

    splitDataDict = {}
    # Print class label vs split
    for className in ds_dict:
        splitDataDict[className] = {}
        for lesionId in ds_dict[className]:
            splitId = ds_dict[className][lesionId]['splitLabel']
            if splitId not in splitDataDict[className]:
                splitDataDict[className][splitId] = []
            splitDataDict[className][splitId].append(lesionId)

    print(splitDataDict)
