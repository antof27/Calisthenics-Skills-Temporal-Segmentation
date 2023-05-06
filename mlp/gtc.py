from name_corrector import name_corrector
from encoding import encoding


def gtc_algorithm(seconds_length, total_frames):
    print("How many skills are there in the video?")
    try: 
        n__skills = int(input())
    except ValueError:
        print("Please insert a number")
        n__skills = int(input())

    print("So.. there are", n__skills, "skills in the video")

    skills_frames_ = []
    skills_seconds_ = []


    for i_skill in range(0, n__skills):
        print("Insert the name of the #", i_skill+1,"skill in the video")
        skill_name_ = input()
        skill_name_ = name_corrector(skill_name_)
        print("Insert the start frame of the #", i_skill+1,"skill in the video")
        #if it's not a number, ask again
        try:
            start_frame_ = int(input())
        except ValueError:
            print("Please insert a number")
            start_frame_ = int(input())
        if start_frame_ < 0:
            start_frame_ = 0

        print("Insert the end frame of the #", i_skill+1,"skill in the video")
        try:
            end_frame_ = int(input())
        except ValueError:
            print("Please insert a number")
            end_frame_ = int(input())

        if end_frame_ > total_frames:
            end_frame_ = total_frames-1
        skills_frames_.append([skill_name_, start_frame_, end_frame_])



    #--------------------- RECONSTRUCT NONE SEQUENCE -----------------------------
    i = 0
    print("Seconds length: ", seconds_length)
    print("total frame: ", total_frames)

    while True:
        if i == 0 and skills_frames_[i][1] != 0:
            skills_frames_.insert(0, ["none", 0, skills_frames_[i][1]-1])
            seconds_length += 1
            
        if i != len(skills_frames_)-1:
            if skills_frames_[i][2]+1 != skills_frames_[i+1][1]:
                skills_frames_.insert(i+1, ["none", skills_frames_[i][2]+1, skills_frames_[i+1][1]-1])
                seconds_length += 1
            
        
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

    return gt_predicted, skills_frames_, skills_seconds_