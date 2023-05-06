#--------------------- VIDEO SEGMENT RECONSTRUCTION -----------------------------
from statistics import mode
import pandas as pd


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



def filtering(pointer, patch_mode, modes, index1, index2, index3, index4, index5=None):
    patch_mode.append(modes[index1][0])
    patch_mode.append(modes[index2][0])
    patch_mode.append(modes[index3][0])
    
    #if all the values have the same occurences
    if len(set(patch_mode)) == len(patch_mode):
        patch_mode.append(modes[index4][0])
        if(index5 != None):
            patch_mode.append(modes[index5][0])

    moda_ = mode(patch_mode)
    
    modes[pointer][0] = moda_
    return modes



#--------------------- FIRST STEP -----------------------------
def vsr_algorithm(raw_predicted):
    #if raw_predicted is a list, convert it to a dataframe
    if type(raw_predicted) == list:
        raw_predicted = pd.DataFrame(raw_predicted)
    
    total_frames = len(raw_predicted)

    
    
    
    
    modes = []
    #modes_2_temp = []
    #set the default size to 15
    window_size = 12
    temp_mode = []

    i = window_size
    while i <= len(raw_predicted):
        values = raw_predicted[i-window_size:i].iloc[:, 0]
        current_mode = values.mode()[0]
        if len(set(values)) == window_size:
            window_size += 1
            i+=1
        
        else:
            
            min_index = values[values==current_mode].index.min()
            window_size = 12
            i += window_size-2
            max_index = values[values==current_mode].index.max()
            
            if i >= len(raw_predicted):
                max_index = len(raw_predicted)-1
            
            temp_mode = [current_mode, min_index, max_index] 
            modes.append(temp_mode)
            
            #modes_2_temp.append(current_mode)

    #--------------------- SECOND STEP -----------------------------
    
    if len(modes) < 4:
        element = []
        for i in range(0, len(modes)):
            element.append(modes[i][0])
        
        return element
    
    for p in range(0, len(modes)):
        patch_mode = []
        if p == 0:
            filtering(p, patch_mode, modes, p, p+1, p+2, p+3)
        elif p == 1:
            filtering(p, patch_mode, modes, p-1, p, p+1, p+2)
        elif p == len(modes)-2:
            filtering(p, patch_mode, modes, p-1, p, p+1, p-2)
        elif p == len(modes)-1:
            filtering(p, patch_mode, modes, p-2, p-1, p, p-3)
        else:
            filtering(p, patch_mode, modes, p-1, p, p+1, p-2, p+2)
        
    #print(modes)

    #--------------------- THIRD STEP -----------------------------

    output = []
    output_l = []
    j = 0
    breakp = False
    while j < len(modes)-1:
        skill = modes[j][0]
        if j == 0:
            start = 0
        else:
            start = modes[j][1]
        while j < len(modes)-1 and modes[j][0] == skill:
            j += 1

            if j == len(modes)-1:
                end = modes[j][2]
                breakp = True

        
        if breakp == False:
            end = (modes[j][1]-1)
        
        output.append([skill, start, end])
        milliseconds = ((end+1)-start)*(1/24)
        output_l.append([skill, milliseconds])
    
    print("\nLista finale: \n")
    print(output)

    #print("\nLista finale in millisecondi: \n")
    #print(output_l)
    
    #total_length = total_frames*(1/24)
    #print("total length in seconds: ", total_length)

    vsr_predicted = []
    for i in range(0, len(output)):
        for j in range(output[i][1], output[i][2]+1):
            vsr_predicted.append(output[i][0])


    vsr_predicted = encoding(vsr_predicted)

    #print("vsr_predicted", vsr_predicted)

    return vsr_predicted, output, output_l

'''
list1 = ["none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"none",
"flag", 
"none",
"none",
"none",
"flag", 
"flag", 
"flag", 
"flag", 
"none",
"none",
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"none",
"flag", 
"none",
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"flag", 
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none",
"none","none","none"]

final_list = vsr_algorithm(list1)
print(final_list)
'''
