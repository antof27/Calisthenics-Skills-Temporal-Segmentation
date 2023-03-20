#!/bin/bash

video_dir="/home/coloranto/Documents/tesi/dataset_video/"

for video_file in $video_dir*; do

    video_name=$(basename "$video_file" | cut -f 1 -d '.')
    echo "$video_name"
    mkdir -p "/home/coloranto/Desktop/test/openpose/output/json/$video_name"
        
    video_path="$video_file"
    json_output_path="/home/coloranto/Desktop/test/openpose/output/json/$video_name/"
    video_output_path="/home/coloranto/Desktop/test/openpose/output/videos/$video_name.avi"
    echo "$video_path"
    echo "$json_output_path"
    echo "$video_output_path"

    ./build/examples/openpose/openpose.bin \
    -keypoint_scale 3 \
    --model_pose BODY_25B \
    --net_resolution -1x208 \
    --video "$video_path" \
    --write_json "$json_output_path" --display 0 \
    --number_people_max 1 \
    --write_video "$video_output_path"

done
