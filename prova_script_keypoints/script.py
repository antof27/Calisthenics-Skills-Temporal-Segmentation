#how to extract data froma a local json and save it to a csv file
import json
import csv
import glob
import re


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

#loop on all json file in the folder
for i, folder in enumerate(glob.glob("./all_video_keypoints/*")):
    
    print("folder: ", folder)

    folder = natural_sort(glob.glob(folder + "/*"))
    
    for file in folder:
        
        #print(file)

        #read the json file
        with open(file) as f:
            data = json.load(f)
        
        #print(file)

        if data["people"] == []:
            continue

        keypoints = data["people"][0]["pose_keypoints_2d"]
        
        #from the name of the file extract the name of the video and the frame
        name = file.split("/")[-1]
        name = name.split("_")
        nome_video = name[0]
        #print("Name vale : ", name)
        #extract the frame number without the 0 at the beginning

        #print("nome_video vale: ", nome_video)

        frame_video = name[1]
        frame_video = frame_video.lstrip("0")
        if frame_video == "":
            frame_video = 0

        frame_video = int(frame_video)


        keypoints.append(nome_video)
        keypoints.append(frame_video)
        
        #print("keypoints vale: ", keypoints)
        #extract the skill id from dataset.csv comparing the name of the video and the time of the frame

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
                keypoints.append("null_skill")

            
            print(keypoints)
        #write the keypoints in the csv file
        with open('dataset.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(keypoints)
            
#close the csv file
f.close()

