import json
import csv
import glob
import re
import pandas as pd


with open('dataset_elaborated.csv', 'w') as f:
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
input_json_folder = "/home/coloranto/Documents/tesi/pre-post_processing_algorithms/pre_processing/elaboration_pre_processing_algorithms/json_source/*"
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
#labels filling...
        with open('dataset_video.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            sem = False
            for row in reader: 
                #print(row)
                
                #we are considering the _copy of the file 
                video_name_edit = video_name
                if video_name_edit.startswith("_"):
                    video_name_edit = video_name_edit[1:]

                #print("Compare ", video_name, " with ", row[6])
                if video_name_edit == row[6] and video_frame >= int(row[3]) and video_frame <= int(row[4]):
                    #print("Sono uguali")
                    keypoints.append(row[5])
                    sem = True
                    break                
            
            if sem == False:
                keypoints.append("none")
#---------------------------------------------------------------------------------------------------------------------
#Videos filtering algorithm

        dataframe_local = dataframe_local.append(pd.DataFrame([keypoints]), ignore_index=True)    
           # print(keypoints)
        #write the keypoints in the csv file

    global_frame_counter += frame_counter    
    print("Processing the video... : ", video_name)
    print("Total zeros keypoints : ", zeros_keypoints_counter)
    print("Total keypoints : ", total_keypoints_counter)
    print("Video frames : ", frame_counter)

    if total_keypoints_counter == 0:
        print("The current video has no frames!", video_name)
    else:    
        rate = zeros_keypoints_counter / total_keypoints_counter
    print("Zeros percentage : {}%".format(rate * 100))
    if rate > percentage:
        print("The current video is excluded from the dataset\n")
        excluded_videos.append(video_name)
        excluded_videos_frames += frame_counter

    else:

#---------------------------------------------------------------------------------------------------------------------
#Sequences reconstruction algorithm

        #create a temporary dataframe
        local_dataframe_output = pd.DataFrame()

        for col in dataframe_local.columns:
            #trasform the column into a list
            l1 = dataframe_local[col].tolist()
            
            if col == 75 or col == 76 or col == 77:
        
                local_dataframe_output[col] = l1
                continue
                        
            n_zeros = 0
            first_number = None
            last_number = None
            new_list = []

            # manage the zeros sequences at the beginning of the list
            while len(l1) > 0 and l1[0] == 0:
                new_list.append(0)
                l1.pop(0)

            for i in range(len(l1)):

                if l1[i] == 0:
                    n_zeros += 1
                    if n_zeros == 1:
                        first_number = l1[i-1]
                else:
                    if n_zeros == 0:
                        
                        new_list.append(l1[i])
                    else:
                        last_number = l1[i]
                        step = (last_number - first_number) / (n_zeros + 1)
                        for j in range(1, n_zeros + 1):
                            temp_step = round(first_number + j * step, 6)
                            new_list.append(temp_step)
                        
                        last_number = round(last_number, 6)
                        new_list.append(last_number)
                        n_zeros = 0

            # manage the zeros sequences at the ending of the list
            while len(l1) > 0 and l1[-1] == 0:
                new_list.append(0)
                l1.pop()

            
            
            local_dataframe_output[col] = new_list
        
        #mirroring the keypoints
        for index, row in local_dataframe_output.iterrows():
            mirrored_row = row
            for i in range(len(row)):
                if i == 75 or i == 76 or i == 77:
                    continue
                if i % 3 == 0:
                    if row[i] != 0:
                        mirrored_row[i] = 1 - row[i]

            local_dataframe_output = local_dataframe_output.append(mirrored_row, ignore_index=True)

#---------------------------------------------------------------------------------------------------------------------
        with open('dataset_elaborated.csv', 'a') as f:
            local_dataframe_output.to_csv(f, header=False, index=False)
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
