import csv

def comparator_function(video_name, video_frame, keypoints):
    with open('dataset_video.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader)
                sem = False
                for row in reader: 
                    
                    #we are considering the _copy of the file 
                    video_name_edit = video_name
                    if video_name_edit.startswith("_"):
                        video_name_edit = video_name_edit[1:]

                    #print("Compare ", video_name, " with ", row[6])
                    if video_name_edit == row[6] and video_frame >= int(row[3]) and video_frame <= int(row[4]):
                        #print("Sono uguali")
                        keypoints.append(row[5])
                        sem = True
                        break                
                
                if sem == False:
                    keypoints.append("none")