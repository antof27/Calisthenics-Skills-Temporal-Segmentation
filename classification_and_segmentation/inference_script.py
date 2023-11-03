import os
import sys
import json
import csv
import glob
import re
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from mlp_new_dataset import MLP
from paper_code import viterbi, SF1

from vsr import vsr_algorithm
from visual_vsr import visualize

from vsr4modifies import vsr_algorithm_display
from openpose_script import openpose_script
from zsr import zsr_algorithm
from gtc import gtc_algorithm
from plot import predicted_plotting, gt_plotting
from demo_timeline import create_timeline_image
from gt_lists_reconstructor import count_sequences
from demo_matrix import draw_horizontal_bars
from demo_temporal import print_element_with_counter
from image_generator import create_timeline_img


environment = os.getcwd()
#--------------------- INPUT VIDEO FROM TERMINAL, CONVERT TO 960X540 AT 24 FPS -----------------------------

#if the number of arguments is not 2, the script will stop
if len(sys.argv) != 2:
    print("Error in the input, please insert the video path")
    sys.exit(1)

try:
    input_video = sys.argv[1]
    video_converted_with_ext = "video_to_inference.mp4"
    video_converted = video_converted_with_ext.split('.')[0]
    os.system(f"ffmpeg -i {input_video} -r 24 -vf \"scale=w=960:h=540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2\" {video_converted_with_ext}")
except:
    print("Error in video conversion, check the video format or the video path")

#---------------------ELABORATING THE VIDEO WITH OPENPOSE ----------------------------

json_output_path = openpose_script(video_converted_with_ext, video_converted)

#--------------------- EXTRACTING THE FEATURES FROM THE JSON FILES AND BUILDING THE DATASET -----------------------------

print("Creating the labels file...")

with open(environment + '/video_to_predict.csv', 'w') as f:
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

#ZERO SEQUENCES RECONSTRUCTION

zsr_dataframe = zsr_algorithm(dataframe_local)

with open(environment + '/video_to_predict.csv', 'a') as f:
    dataframe_local.to_csv(f, header=False, index=False)

#MLP INFERENCE

#encode the labels to number  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)

# load the model
model = MLP()
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()
data_to_predict = pd.read_csv(environment + '/video_to_predict.csv')

X_pred = data_to_predict.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
X_pred = torch.FloatTensor(X_pred.values).to(device)

#make predictions, saving the index to predicted and without saving the gradient operations
with torch.no_grad():
    outputs = model(X_pred)
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)


# put the predicted labels in a csv file    
predicted_video = pd.DataFrame()
predicted_video['video_name'] = "video_name"

raw_predicted = predicted.tolist()

predicted_labels = le.inverse_transform(predicted.tolist())
predicted_video['video_name'] = predicted_labels

probabilities_matrix = probabilities.cpu().numpy()

print(probabilities_matrix)

predicted_video.to_csv('predicted_video.csv', index=False)


#VIDEO SEGMENT RECONSTRUCTION

df = pd.read_csv("predicted_video.csv")
#create a copy of the dataframe
df_copy = df.copy()
#vsr_predicted, output, output_l = vsr_algorithm(df, 13)
vsr_predicted, output, output_l = vsr_algorithm(df)

#--------------------- PLOTTING THE PREDICTED -----------------------------

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
seconds_length = predicted_plotting(output, output_l, ax1, ax2)

#--------------------- GROUND TRUTH COMPARISON -----------------------------

gt_predicted, skills_frames_, skills_seconds_ = gtc_algorithm(seconds_length, total_frames)

gt_l = count_sequences(gt_predicted)
gt_l = [x for x in gt_l if x[0] != 'none']



#--------------------- PLOTTING THE GROUND TRUTH -----------------------------

gt_plotting(skills_frames_, skills_seconds_, ax3, ax4)

fig.tight_layout()
fig.subplots_adjust(top=0.85)

fig.suptitle("Video segmentation timelines", fontsize=16, fontweight='bold')

#plt.show()

#--------------------- METRIC EVALUATION -----------------------------
raw_results, raw_value = SF1(gt_predicted, raw_predicted)
print("raw_results: ", raw_results)
print("raw_value: ", raw_value)

vsr_results, vsr_value = SF1(gt_predicted, vsr_predicted)
print("vsr_results: ", vsr_results)
print("vsr_value: ", vsr_value)

viterbi_output = viterbi(probabilities_matrix, 10e-20)
paper_results, paper_value = SF1(gt_predicted, viterbi_output)
print("paper_results: ", paper_results)
print("paper_value: ", paper_value)
#--------------------- METRIC EVALUATION -----------------------------

# Create the timeline image
# Example list names and skills


list_names = ["Raw", "Heuristic", "Probabilistic", "GT"]
skills = [raw_predicted, vsr_predicted, viterbi_output, gt_predicted]

# Create the timeline image
num_frames = len(raw_predicted)

# Create a folder to save the timeline images
timeline_folder = './demo/timeline'


#from output_l can you remove that elements in which the first element, is none

output_l = [x for x in output_l if x[0] != 'none']


print("output_l: ", output_l)
print("gt_l: ", gt_l)

#save the probabilities matrix into a txt file 

folder_path1 = "./demo/matrix/"

# Generate images with horizontal bars and save them in the folder
for i, row in enumerate(probabilities_matrix):
    image = draw_horizontal_bars([row], vsr_predicted[i], gt_predicted[i])
    
    image.save(os.path.join(folder_path1, f"matrix_{i:04}.png"))


print_element_with_counter(vsr_predicted, gt_predicted)


# Create a loop to generate images for each frame
for frame in range(num_frames):
    # Call the create_timeline_image function with the current frame
    timeline_image = create_timeline_image(list_names, skills, frame)

    # Save the image with a filename that includes the frame number in the specified folder
    filename = f"timeline_frame_{frame:04}.png"
    file_path = os.path.join(timeline_folder, filename)
    timeline_image.save(file_path)



#vsr_algorithm_display(df, gt_predicted)
#visualize(df_copy, gt_predicted)



'''

timeline_image = create_timeline_img(list_names, skills)
# Display the image
timeline_image.show()
#save the image
timeline_image.save("inf.png")
'''



f.close()
