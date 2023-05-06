import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder
import json
import csv
import glob
import re
from scipy.stats import mode
from matplotlib import pyplot as plt
from mlp_post_edit import MLP
from paper_code import viterbi, SF1
from vsr import vsr_algorithm

#i want to create an enum that contain the corrispondence between coded labels and decoded labels
#i want 
def encoding(list):
    for i in range (len(list)):
        if list[i] == "bl":
            list[i] = 0
        elif list[i] == "fl":
            list[i] = 1
        elif list[i] == "flag":
            list[i] = 2
        elif list[i] == "ic":
            list[i] = 3
        elif list[i] == "mal":
            list[i] = 4
        elif list[i] == "none":
            list[i] = 5
        elif list[i] == "oafl":
            list[i] = 6
        elif list[i] == "oahs":
            list[i] = 7
        elif list[i] == "pl":
            list[i] = 8
    return list


#--------------------- INPUT VIDEO FROM TERMINAL, CONVERT TO 960X540 AT 24 FPS -----------------------------
try:
    input_video = sys.argv[1]

    video_converted_with_ext = "video_to_inference.mp4"

    video_converted = video_converted_with_ext.split('.')[0]
    os.system(f"ffmpeg -i {input_video} -r 24 -vf \"scale=w=960:h=540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2\" {video_converted_with_ext}")
except:
    print("Error in video conversion, check the video format or the video path")


#---------------------ELABORATING THE VIDEO WITH OPENPOSE ----------------------------

video_dir = "/home/coloranto/Documents/tesi/mlp/"
video_path = video_dir + video_converted_with_ext
json_dir = "/home/coloranto/Documents/tesi/mlp/inference_json/"

json_output_path = json_dir + video_converted + "/"
if os.path.exists(json_output_path):
    i = 1
    while True:
        new_path = json_dir + video_converted + "_" + str(i) + "/"
        if not os.path.exists(new_path):
            json_output_path = new_path
            break
        i += 1

os.mkdir(json_output_path)

video_output_path = f"/home/coloranto/Documents/tesi/mlp/{video_converted}.avi"
print(video_path)
print(json_output_path)
print(video_output_path)

os.chdir("/home/coloranto/Desktop/test/openpose/")


try : 
    os.system(f"./build/examples/openpose/openpose.bin \
    -keypoint_scale 3 \
    --model_pose BODY_25B \
    --net_resolution -1x208 \
    --video {video_path} \
    --write_json {json_output_path} --display 0 \
    --number_people_max 1 \
    --write_video {video_output_path}")

except:
    print("Error in openpose elaboration, check the video format or the video path")

print("Video correctly elaborated by openpose!\n")

os.chdir("/home/coloranto/Documents/tesi/mlp/")

#--------------------- EXTRACTING THE FEATURES FROM THE JSON FILES AND BUILDING THE DATASET -----------------------------
print("Provo a crearlo")

with open('/home/coloranto/Documents/tesi/mlp/video_to_predict.csv', 'w') as f:
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


print("Creato!")



#order alfabetically the folder
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


print("Starting the script...")
folder = json_output_path
print("Reading json files from: ", folder)


#i should open all the json files in the folder
folder = glob.glob(folder + "*.json")
folder = natural_sort(folder)
dataframe_local = pd.DataFrame()

total_frames = 0

for file in folder:
    total_frames += 1
    with open(file) as f:
        data = json.load(f)
    
    
    if data["people"] == []:
        keypoints = [0] * 75
        #continue
    else:
        keypoints = data["people"][0]["pose_keypoints_2d"]
    

    #extract the frame number from the file name
    frame_num = file.split("/")[-1]
    frame_num = frame_num.split("_")[3]

    frame_num = frame_num.lstrip("0")
    if frame_num == "":
        frame_num = 0

    frame_num = int(frame_num)


    keypoints.append(video_converted)
    keypoints.append(frame_num)
        
    dataframe_local = dataframe_local.append(pd.DataFrame([keypoints]), ignore_index=True)  

print("Dataframe created!\n", dataframe_local)

#--------------------- ZERO SEQUENCES RECONSTRUCTION -----------------------------


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
                range_ = last_number - first_number 
                if range_ < 0:
                    step = (last_number - first_number) / (n_zeros + 1)
                    for j in range(1, n_zeros + 1):
                        new_list.append(first_number + j * step)
                    new_list.append(last_number)
                else:
                    for j in range(1, n_zeros + 1):
                        new_list.append(0)
                    new_list.append(last_number)
                    
                n_zeros = 0
    # manage the zeros sequences at the ending of the list
    while len(l1) > 0 and l1[-1] == 0:
        new_list.append(0)
        l1.pop()


    local_dataframe_output[col] = new_list


with open('/home/coloranto/Documents/tesi/mlp/video_to_predict.csv', 'a') as f:
    dataframe_local.to_csv(f, header=False, index=False)


#--------------------- MLP INFERENCE -----------------------------

#encode the labels to number  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)



# load the model
model = MLP()
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()
data_to_predict = pd.read_csv('/home/coloranto/Documents/tesi/mlp/video_to_predict.csv')

X_pred = data_to_predict.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
X_pred = torch.FloatTensor(X_pred.values).to(device)

#make predictions, saving the index to predicted and without saving the gradient operations
with torch.no_grad():
    outputs = model(X_pred)
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)



# put the predicted labels in a csv file    
predicted_video = pd.DataFrame()
predicted_numerical = pd.DataFrame()

predicted_video['video_name'] = "video_name"
predicted_numerical['video_name'] = "video_name"
predicted_numerical["video_name"] = predicted.tolist()

raw_predicted = predicted.tolist()
print("len raw_predicted: ", len(raw_predicted))

predicted_labels = le.inverse_transform(predicted.tolist())
predicted_video['video_name'] = predicted_labels

predicted_video['probabilities'] = probabilities.tolist()

probabilities_matrix = np.zeros((total_frames, 9))
for i in range(len(predicted_video)):
    probabilities_matrix[i] = predicted_video.iloc[i, 1]

print(probabilities_matrix)


predicted_numerical.to_csv('predicted_numerical.csv', index=False)

predicted_video.to_csv('predicted_video.csv', index=False)


#--------------------- VIDEO SEGMENT RECONSTRUCTION -----------------------------

df = pd.read_csv("predicted_video.csv")

vsr_predicted, output, output_l = vsr_algorithm(df)


#--------------------- PLOTTING THE PREDICTED -----------------------------


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


x = []
y = []
for i in range(0, len(output)):
    x.append(output[i][1])
    x.append(output[i][2])
    y.append(output[i][0])
    y.append(output[i][0])


ax1.plot(x, y)
ax1.set_title("Predicted - frames")
ax1.set_xlabel("Frames")
ax1.set_ylabel("Skills")

x1 = []
y1= []
k = 0
for i in range(0, len(output_l)):
    x1.append(k)
    x1.append(k+output_l[i][1])
    y1.append(output_l[i][0])
    y1.append(output_l[i][0])
    k+=output_l[i][1]

print("x1", x1)
print("y1", y1)



ax2.plot(x1, y1, color='red')
ax2.set_title("Predicted - seconds")
ax2.set_xlabel("Seconds")
ax2.set_ylabel("Skills")


#--------------------- GROUND TRUTH COMPARISON -----------------------------

print("How many skills are there in the video?")
try: 
    n__skills = int(input())
except ValueError:
    print("Please insert a number")
    n__skills = int(input())

print("So.. there are", n__skills, "skills in the video")

skills_frames_ = []
skills_seconds_ = []

def name_corrector(name):
    #print("Name inserted: ", name)
    if name.startswith(" "):
        name = name[1:]
    if name.endswith(" "):
        name = name[:-1]

    if name == "planche" or name == "pl":
        name = "pl"
    elif name == "one arm handstand" or name == "one-arm-handstand" or name == "one_arm_handstand" or name == "oahs":
        name = "oahs"
    elif name == "front lever" or name == "front-lever" or name == "front_lever" or name == "front" or name == "frontlever" or name == "fl":
        name = "fl"
    elif name == "back lever" or name == "back-lever" or name == "back_lever" or name == "bl":
        name = "bl"
    elif name == "one arm front lever" or name == "one-arm-front-lever" or name == "one_arm_front_lever" or name == "oafl":
        name = "oafl"
    elif name == "human flag" or name == "human_flag" or name == "hf" or name == "human-flag" or name == "flag":
        name = "flag"
    elif name == "iron cross" or name == "iron_cross" or name == "iron-cross" or name == "cross" or name == "ic":
        name = "ic"
    elif name == "maltese" or name == "mal":
        name = "mal" 
    else:
        print("The name you inserted is not correct, please insert the correct name")
        name = input()
        name = name_corrector(name)
    #print("Name corrected: ", name, "")        
    return name


for i_skill in range(0, n__skills):
    print("Insert the name of the #", i_skill+1,"skill in the video")
    skill_name_ = input()
    skill_name_ = name_corrector(skill_name_)
    print("Insert the start frame of the #", i_skill+1,"skill in the video")
    start_frame_ = int(input())
    if start_frame_ < 0:
        start_frame_ = 0
    print("Insert the end frame of the #", i_skill+1,"skill in the video")
    end_frame_ = int(input())
    if end_frame_ > total_frames:
        end_frame_ = total_frames-1
    skills_frames_.append([skill_name_, start_frame_, end_frame_])



#--------------------- RECONSTRUCT NONE SEQUENCE -----------------------------
i = 0
print("k: ", k)
print("total frame: ", total_frames)

while True:
    if i == 0 and skills_frames_[i][1] != 0:
        skills_frames_.insert(0, ["none", 0, skills_frames_[i][1]-1])
        k = k+1
        
    if i != len(skills_frames_)-1:
        if skills_frames_[i][2]+1 != skills_frames_[i+1][1]:
            skills_frames_.insert(i+1, ["none", skills_frames_[i][2]+1, skills_frames_[i+1][1]-1])
            k = k+1
        
    
    if i == len(skills_frames_)-1:
        if skills_frames_[i][2]+1 < total_frames-1:
            skills_frames_.append(["none", skills_frames_[i][2]+1, total_frames-1]) 
        break

    i = i+1

print("Skills in frames: ", skills_frames_)

#Converting the frames in seconds into a new list
for i in range(0, len(skills_frames_)):
    skills_seconds_.append([skills_frames_[i][0], (skills_frames_[i][2]-skills_frames_[i][1])/24])

print("Skills in seconds: ", skills_seconds_)

gt_predicted = []
for i in range(0, len(skills_frames_)):
    for j in range(skills_frames_[i][1], skills_frames_[i][2]+1):
        gt_predicted.append(skills_frames_[i][0])

print("gt_predicted", gt_predicted)

gt_predicted = encoding(gt_predicted)

print("gt_predicted", gt_predicted)

#--------------------- PLOTTING THE GROUND TRUTH -----------------------------

x2 = []
y2 = []
for i in range(0, len(skills_frames_)):
    x2.append(skills_frames_[i][1])
    x2.append(skills_frames_[i][2])
    y2.append(skills_frames_[i][0])
    y2.append(skills_frames_[i][0])


ax3.plot(x2, y2)
ax3.set_title("Ground truth - frames")
ax3.set_xlabel("Frames")
ax3.set_ylabel("Skills")

x3 = []
y3= []
k = 0
for i in range(0, len(skills_seconds_)):
    x3.append(k)
    x3.append(k+skills_seconds_[i][1])
    y3.append(skills_seconds_[i][0])
    y3.append(skills_seconds_[i][0])
    k+=skills_seconds_[i][1]

print("x1", x2)
print("y1", y2)


ax4.plot(x3, y3, color='red')
ax4.set_title("Ground truth - seconds")
ax4.set_xlabel("Seconds")
ax4.set_ylabel("Skills")


fig.tight_layout()
fig.subplots_adjust(top=0.85)

fig.suptitle("Video segmentation timelines", fontsize=16, fontweight='bold')

plt.show()

# METRIC CALCULATION
raw_results, raw_value = SF1(gt_predicted, raw_predicted)
print("raw_results: ", raw_results)
print("raw_value: ", raw_value)

vsr_results, vsr_value = SF1(gt_predicted, vsr_predicted)
print("vsr_results: ", vsr_results)
print("vsr_value: ", vsr_value)

viterbi_output = viterbi(probabilities_matrix, 10e-10)
paper_results, paper_value = SF1(gt_predicted, viterbi_output)
print("paper_results: ", paper_results)
print("paper_value: ", paper_value)



f.close()
