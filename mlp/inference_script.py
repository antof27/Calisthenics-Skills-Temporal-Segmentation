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



for file in folder:
    with open(file) as f:
        data = json.load(f)
    

    if data["people"] == []:
        continue

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
    
with open('/home/coloranto/Documents/tesi/mlp/video_to_predict.csv', 'a') as f:
    local_dataframe_output.to_csv(f, header=False, index=False)


#--------------------- MLP INFERENCE -----------------------------
input_size = 75
hidden_units = 512
num_classes = 9

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super(MLP, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_units)
        self.activation = nn.ReLU()
        
        self.hl2 = nn.Linear(hidden_units, hidden_units)
        self.activation2 = nn.ReLU()
        
        self.hl3 = nn.Linear(hidden_units, hidden_units)
        self.activation3 = nn.ReLU()
        
        self.hl4 = nn.Linear(hidden_units, num_classes)
        self.output_layer = nn.LogSoftmax(dim=1)

    
    def forward(self,x):
        hidden_representation = self.hl1(x)
        hidden_representation = self.activation(hidden_representation)
        
        hidden_representation = self.hl2(hidden_representation)
        hidden_representation = self.activation2(hidden_representation)

        hidden_representation = self.hl3(hidden_representation)
        hidden_representation = self.activation3(hidden_representation)

        hidden_representation = self.hl4(hidden_representation)
        scores = self.output_layer(hidden_representation)
        return scores


# Load the model
model = MLP(input_size, hidden_units, num_classes)
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load the label encoder


# Load the data to predict
data_to_predict = pd.read_csv('/home/coloranto/Documents/tesi/mlp/video_to_predict.csv')

# Preprocess the data
X_pred = data_to_predict.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
X_pred = torch.FloatTensor(X_pred.values).to(device)

# Make predictions
with torch.no_grad():
    outputs = model(X_pred)
    _, predicted = torch.max(outputs.data, 1)

# Decode the predicted labels
predicted_labels = le.inverse_transform(predicted.tolist())

# put the predicted labels in a csv file    
predicted_video = pd.DataFrame()
predicted_video['video_name'] = "video_name"
predicted_video['video_name'] = predicted_labels

predicted_video.to_csv('predicted_video.csv', index=False)

#--------------------- VIDEO SEGMENTATION -----------------------------

df = pd.read_csv("predicted_video.csv")

modes = []
modes_2_temp = []
#set the default size to 3
window_size = 3
temp_mode = []
i = window_size
while i <= len(df):
    values = df[i-window_size:i].iloc[:, 0]
    current_mode = values.mode()[0]

    if len(set(values)) == window_size:
        window_size += 1
        i+=1
       
    else:
        mode_value = mode(values)[0][0]

        min_index = values[values==mode_value].index.min()
        max_index = values[values==mode_value].index.max()

        temp_mode = [current_mode, min_index, max_index] 
        modes.append(temp_mode)
        modes_2_temp.append(current_mode)

        arr = values.values
        max_mode_index = np.where(arr == current_mode)[0][-1]
        window_size = 3
        i += window_size-1 
        
    
print(modes)
print(modes_2_temp)

print("Second step : \n")

final_array = []
print(len(modes))
for i in range(0, len(modes)):
    patch_mode = []
    if i != 0 and i != len(modes)-1:
        patch_mode.append(modes[i-1][0])
        patch_mode.append(modes[i][0])
        patch_mode.append(modes[i+1][0])
        moda_ = mode(patch_mode)[0][0]
        modes[i][0] = moda_
        #final_array.append([mode(patch_mode)[0][0], modes[i][1], modes[i][2]])
    elif i == 0:
        patch_mode.append(modes[i][0])
        patch_mode.append(modes[i+1][0])
        patch_mode.append(modes[i+2][0])
        moda_ = mode(patch_mode)[0][0]
        modes[i][0] = moda_

        #final_array.append([mode(patch_mode)[0][0], modes[i][1], modes[i][2]])
    else:
        patch_mode.append(modes[i-2][0])
        patch_mode.append(modes[i-1][0])
        patch_mode.append(modes[i][0])
        moda_ = mode(patch_mode)[0][0]
        modes[i][0] = moda_
        
        #final_array.append([mode(patch_mode)[0][0], modes[i][1], modes[i][2]])
    
print(modes)
#print(final_array)




output = []
output_l = []
i = 0
breakp = False
while i < len(modes)-1:
    skill = modes[i][0]
    start = modes[i][1]
    while i < len(modes)-1 and modes[i][0] == skill:
        i += 1

        if i == len(modes)-1:
            end = modes[i][2]
            breakp = True

    
    if breakp == False:
        end = (modes[i][1]-1)
    
    output.append([skill, start, end])
    milliseconds = ((end+1)-start)*(1/24)
    output_l.append([skill, milliseconds])

print("\nLista finale: \n")
print(output)

print("\nLista finale in millisecondi: \n")
print(output_l)

total_frame = (output[-1][2])+1
total_length = total_frame*(1/24)
print("total length in seconds: ", total_length)


#create a subplot with 2 rows and 2 columns




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
    print("Name inserted: ", name)
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
    print("Name corrected: ", name, "")        
    return name


for i_skill in range(0, n__skills):
    print("Insert the name of the #", i_skill+1,"skill in the video")
    skill_name_ = input()
    skill_name_ = name_corrector(skill_name_)
    print("Insert the start frame of the #", i_skill+1,"skill in the video")
    start_frame_ = int(input())
    print("Insert the end frame of the #", i_skill+1,"skill in the video")
    end_frame_ = int(input())
    skills_frames_.append([skill_name_, start_frame_, end_frame_])


#in the other frames, out of the interval, the skill is not performed so the skill_name is "none"
i = 0
k = len(skills_frames_)

while True:
    if i == 0 and skills_frames_[i][1] != 0:
        skills_frames_.insert(0, ["none", 0, skills_frames_[i][1]-1])
        k = k+1
        
    if i != k-1:
        if skills_frames_[i][2]+1 != skills_frames_[i+1][1]:
            skills_frames_.insert(i+1, ["none", skills_frames_[i][2]+1, skills_frames_[i+1][1]-1])
            k = k+1

    else:
        skills_frames_.append(["none", skills_frames_[i][2]+1, total_frame-1]) 
        break

    i = i+1

print("Skills in frames: ", skills_frames_)

#convert the frames in seconds
for i in range(0, len(skills_frames_)):
    skills_seconds_.append([skills_frames_[i][0], (skills_frames_[i][2]-skills_frames_[i][1])/24])

print("Skills in seconds: ", skills_seconds_)




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

f.close()
