import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from encoding import encoding

# Read the csv file with pandas
df = pd.read_csv("dataset_elaborated6N.csv")

total_videos = 0
total_frames = 0
video_skills = [0,0,0,0,0,0,0,0,0]
frame_skills = [0,0,0,0,0,0,0,0,0]
video_skills_container = []


def processing(video_name, video_skills_container, video_skills, string):
   
    if video_name not in video_skills_container:
        video_skills_container.append(video_name)
        video_name = string   
        video_name = encoding(video_name)
        video_skills[video_name] += 1
    else:
        video_name = string    
        video_name = encoding(video_name)

    return video_name


for i in range(0, len(df)):
    #check the 75th columns
    video_name = df.iloc[i, 75]
    if video_name.startswith("pl"):
        video_name = processing(video_name, video_skills_container, video_skills, "pl")
    elif video_name.startswith("flag"):
        video_name = processing(video_name, video_skills_container, video_skills, "flag")
    elif video_name.startswith("fl"):
        video_name = processing(video_name, video_skills_container, video_skills, "fl")
    elif video_name.startswith("ic"):
        video_name = processing(video_name, video_skills_container, video_skills, "ic")
    elif video_name.startswith("mal"):
        video_name = processing(video_name, video_skills_container, video_skills, "mal")
    elif video_name.startswith("oafl"):
        video_name = processing(video_name, video_skills_container, video_skills, "oafl")
    elif video_name.startswith("oahs"):
        video_name = processing(video_name, video_skills_container, video_skills, "oahs")
    elif video_name.startswith("bl"):
        video_name = processing(video_name, video_skills_container, video_skills, "bl")
    
    if df.iloc[i, 77] == "none":
        frame_skills[5] += 1
    else:
        frame_skills[video_name] += 1
    
    total_frames += 1



print(total_frames)
print("video_skills: ", video_skills)
print("frame_skills: ", frame_skills)

plt.figure(1)
plt.pie(frame_skills, labels=["bl", "fl", "flag", "ic", "mal", "none", "oafl", "oahs", "pl"], autopct='%1.1f%%')
plt.show()