#how to extract data froma a local json and save it to a csv file
import json
import csv
import glob
import re
import pandas as pd


with open('dataset.csv', 'w') as f:
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
                                'nome_video', 'frame_video', 'skill_id'])
    


#order alfabetically the folder
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

zeros_frames_counter = 0
total_frames_counter = 0
dataframe_local = pd.DataFrame()

percentage = 0.5
print("La percentuale di keypoints minima per video è settata a : {} %".format(percentage * 100))


for i, folder in enumerate(glob.glob("/home/coloranto/Desktop/test/openpose/output/json/*")):
    
    folder = natural_sort(glob.glob(folder + "/*"))
    
    for file in folder:
        
        #print(file)

        with open(file) as f:
            data = json.load(f)
        
        if data["people"] == []:
            continue

        keypoints = data["people"][0]["pose_keypoints_2d"]
        
        #from the name of the file extract the name of the video and the frame
        name = file.split("/")[-1]
        name = name.split("_")
        nome_video = name[0]
        
        #extract the frame number without the 0 at the beginning


        #zeros check
        

        for i in range(len(keypoints)):
            if keypoints[i] == 0:
                zeros_frames_counter += 1
            total_frames_counter += 1

       
        

        frame_video = name[1]
        frame_video = frame_video.lstrip("0")
        if frame_video == "":
            frame_video = 0

        frame_video = int(frame_video)

        keypoints.append(nome_video)
        keypoints.append(frame_video)
        
        #print("keypoints vale: ", keypoints)

        #comparing the labels using the id of the video

        with open('dataset_video.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            sem = False
            for row in reader: 
                #print(row)


                #print("Compare ", nome_video, " with ", row[6])
                if nome_video == row[6] and frame_video >= int(row[3]) and frame_video <= int(row[4]):
                    #print("Sono uguali")
                    keypoints.append(row[5])
                    sem = True
                    break                
            
            if sem == False:
                keypoints.append("none")

        dataframe_local = dataframe_local.append(pd.DataFrame([keypoints]), ignore_index=True)    
           # print(keypoints)
        #write the keypoints in the csv file
        
    print("Processing the video... : ", nome_video)
    print("Total zeros frames : ", zeros_frames_counter)
    print("Total frames : ", total_frames_counter)
    if total_frames_counter == 0:
        print("The current video has no frames!", nome_video)
    else:    
        rate = zeros_frames_counter / total_frames_counter
    print("Zeros percentage : {} %".format(rate * 100))
    if rate > percentage:
        print("The current video is excluded from the dataset\n")
    else:
        
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
            

        with open('dataset.csv', 'a') as f:
            local_dataframe_output.to_csv(f, header=False, index=False)
        print("The current video has been added to dataset!\n")
    
    dataframe_local = pd.DataFrame()
    zeros_frames_counter = 0
    total_frames_counter = 0

#close the csv file
f.close()
