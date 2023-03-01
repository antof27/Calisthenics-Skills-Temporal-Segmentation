import pandas as pd
from scipy.stats import mode
import numpy as np


df = pd.read_csv("labels_sample.csv")


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
        end = (modes[i][1])-1
    
    output.append([skill, start, end])
    milliseconds = (end-start)*(1/24)
    output_l.append([skill, milliseconds])

print("\nLista finale: \n")
print(output)
print("\nLista finale in millisecondi: \n")
print(output_l)

