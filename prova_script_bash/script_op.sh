#bash script to loop on a folder and print the name of the files

# Path: script_op.sh
#!/bin/bash

# Specifica la cartella contenente i file video

import os

video_dir="/content/drive/MyDrive/datasetRaw/video_to_render/"

# Cicla attraverso tutti i file video nella cartella

for video_file in os.listdir(video_dir):
    
    # Esegui un'operazione su ogni file video (ad esempio, riprodurlo)
    #!echo "$(basename "$video_file")"

    video_name = video_file.split(".")[0]
    
    !mkdir -p /content/drive/MyDrive/datasetRaw/json_frames/$video_name    
        
    video_path = "/content/drive/MyDrive/datasetRaw/video_to_render/"+video_file
    json_output_path = "/content/drive/MyDrive/datasetRaw/json_frames/"+video_name+"/"
    video_output_path = "/content/drive/MyDrive/datasetRaw/out_video/"+video_file

    !./build/examples/openpose/openpose.bin \
    -keypoint_scale 3\
    --model_pose BODY_25B \
    --video {video_path} \
    --write_json {json_output_path} --display 0 \
    --number_people_max 1 \
    --write_video {video_output_path}

