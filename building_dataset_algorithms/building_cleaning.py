import json
import csv
import glob
import re
import pandas as pd
import os
from zsr import zsr_algorithm
from comparator import comparator_function

#get the current folder
environment = os.getcwd()

with open('dataset_elaborated_tmp.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['NoseX', 'NoseY', 'NoseC',
                                'LEyeX', 'LEyeY', 'LEyeC',
                                'REyeX', 'REyeY', 'REyeC',
                                'LEarX', 'LEarY', 'LEarC',
                                'REarX', 'REarY', 'REarC',
                                'LShoulderX', 'LShoulderY', 'LShoulderC',
                                'RShoulderX', 'RShoulderY', 'RShoulderC',
                                'LElbowX', 'LElbowY', 'LElbowC',
                                'RElbowX', 'RElbowY', 'RElbowC',
                                'LWristX', 'LWristY', 'LWristC',
                                'RWristX', 'RWristY', 'RWristC',
                                'LHipX', 'LHipY', 'LHipC',
                                'RHipX', 'RHipY', 'RHipC',
                                'LKneeX', 'LKneeY', 'LKneeC',
                                'RKneeX', 'RKneeY', 'RKneeC',
                                'LAnkleX', 'LAnkleY', 'LAnkleC',
                                'RAnkleX', 'RAnkleY', 'RAnkleC',
                                'UpperNeckX', 'UpperNeckY', 'UpperNeckC',
                                'HeadTopX', 'HeadTopY', 'HeadTopC',
                                'LBigToeX', 'LBigToeY', 'LBigToeC',
                                'LSmallToeX', 'LSmallToeY', 'LSmallToeC',
                                'LHeelX', 'LHeelY', 'LHeelC',
                                'RBigToeX', 'RBigToeY', 'RBigToeC',
                                'RSmallToeX', 'RSmallToeY', 'RSmallToeC',
                                'RHeelX', 'RHeelY', 'RHeelC',
                                'video_name', 'video_frame', 'skill_id'])
    


#order alfabetically the folder
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)
print("Starting the script...")

zeros_keypoints_counter = 0
total_keypoints_counter = 0
dataframe_local = pd.DataFrame()
percentage = 0.5
print("Minimum keypoints number threshold set to : {} %".format(percentage * 100))
global_frame_counter = 0
global_video_counter = 0
frame_counter = 0
excluded_videos = []
excluded_videos_frames = 0

#reading all the json files

input_json_folder = environment + "/temp_json/*"
print("Reading the json files from the folder : ", input_json_folder)

for i, folder in enumerate(glob.glob(input_json_folder)):
    
    #sorting frames basing on the name
    folder = natural_sort(glob.glob(folder + "/*"))
    global_video_counter += 1
    for file in folder:
        
        with open(file) as f:
            data = json.load(f)
        
        if data["people"] == []:
            continue

        keypoints = data["people"][0]["pose_keypoints_2d"]
        
        #from the name of the file extract the name of the video and the frame
        name = file.split("/")[-1]
        name = name.split("_")
        video_name = name[0]
        
        #extract the frame number without the 0 at the beginning

        frame_counter += 1

        #zeros check
        for i in range(len(keypoints)):
            if keypoints[i] == 0:
                zeros_keypoints_counter += 1
            total_keypoints_counter += 1
        

        video_frame = name[1]
        video_frame = video_frame.lstrip("0")
        if video_frame == "":
            video_frame = 0

        video_frame = int(video_frame)

        keypoints.append(video_name)
        keypoints.append(video_frame)
        

#---------------------------------------------------------------------------------------------------------------------
#comparing the labels using the id of the video
        
        comparator_function(video_name, video_frame, keypoints)

#---------------------------------------------------------------------------------------------------------------------
#Videos filtering algorithm

        dataframe_local = dataframe_local.append(pd.DataFrame([keypoints]), ignore_index=True)    
           # print(keypoints)
        #write the keypoints in the csv file

    global_frame_counter += frame_counter    
    #print("Processing the video... : ", video_name)
    #print("Total zeros keypoints : ", zeros_keypoints_counter)
    #print("Total keypoints : ", total_keypoints_counter)
    #print("Video frames : ", frame_counter)

    if total_keypoints_counter == 0:
        print("The current video has no frames!", video_name)
    else:    
        rate = zeros_keypoints_counter / total_keypoints_counter
    #print("Zeros percentage : {}%".format(rate * 100))
    if rate > percentage:
        print("The current video is excluded from the dataset\n")
        excluded_videos.append(video_name)
        excluded_videos_frames += frame_counter

    else:

#------------------------ Zero Sequences Reconstruction ----------------------
        print("Starting the zero sequences reconstruction...")
        local_dataframe_output = zsr_algorithm(dataframe_local)
        
#---------------------------------------------------------------------------------------------------------------------

        with open('dataset_elaborated_tmp.csv', 'a') as f:
            dataframe_local.to_csv(f, header=False, index=False)
        print("The current video has been added to dataset!\n")
    
    dataframe_local = pd.DataFrame()
    zeros_keypoints_counter = 0
    frame_counter = 0
    total_keypoints_counter = 0

print("Total frames processed: ", global_frame_counter)
print("Total videos processed: ", global_video_counter)
print("Total excluded videos: ", len(excluded_videos))
print("Excluded videos: ", excluded_videos)
print("Excluded videos frames: ", excluded_videos_frames)

f.close()

