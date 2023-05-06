import os 


def openpose_script(video_converted_with_ext, video_converted):
    
    environment = os.getcwd()
    video_dir = environment + "/"
    print("video_dir", video_dir)
    video_path = video_dir + video_converted_with_ext
    json_dir = environment + "/inference_json/"

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
    video_output_path = environment + video_converted + ".avi"
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

    os.chdir(environment)
    return json_output_path
