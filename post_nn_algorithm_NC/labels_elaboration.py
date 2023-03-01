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



























'''
df = pd.read_csv("labels_sample.csv")

# inizializza l'array in cui salvare le mode delle finestre
modes = []
modes_2_temp = []

# inizializza la dimensione della finestra
window_size = 3
temp_mode = []


# inizia a iterare le righe del dataframe partendo dalla riga con indice window_size
i = window_size

while i <= len(df):

    # prendi i valori della finestra corrente
    values = df[i-window_size:i].iloc[:, 0]
    # calcola la moda dei valori della finestra
    print("i valori sono : \n", values)

    current_mode = values.mode()[0]
    
    
    print("window_size", window_size)
    #print("len(set(values))", len(set(values)))
    #print("set(values)", set(values))

    if len(set(values)) == window_size:
        # aumenta la dimensione della finestra
        print("nei valori non c'è la moda, aumento la finestra di uno")
        window_size += 1
        i+=1
    # altrimenti, salva la moda della finestra e vai avanti alla prossima finestra
    else:
        print("la moda c'è ed è : ", current_mode)
        #print("inserisco la moda nell'array")
        

        mode_value = mode(values)[0][0]

        # Indice più piccolo dell'elemento moda
        min_index = values[values==mode_value].index.min()

        # Indice più grande dell'elemento moda
        max_index = values[values==mode_value].index.max()

        print("Moda:", mode_value)
        print("Indice più piccolo della moda:", min_index)
        print("Indice più grande della moda:", max_index)

        temp_mode = [current_mode, min_index, max_index]
        modes.append(temp_mode)
        modes_2_temp.append(current_mode)


        # prendi l'indice massimo della moda


        arr = values.values
        print("arr vale: ", arr)


        max_mode_index = np.where(arr == current_mode)[0][-1]
        print("max_mode_index", max_mode_index)
        # iterazione a partire dal valore dopo l'indice massimo della moda
        window_size = 3
        i += window_size-1 #rate di incremento 
        print("\ni DOPO vale: ", i)
      
# stampa le mode trovate

print(modes)
'''


'''
patch_mode.append(modes[0][0])
patch_mode.append(modes[1][0])
patch_mode.append(modes[2][0])
making_array.append(mode(patch_mode)[0][1])
making_array.append(mode(patch_mode)[0][2])
final_array.append(mode(patch_mode)[0][0])

for i in range(1, len(modes)-1):
    patch_mode = []
    patch_mode.append(modes[i-1][0])
    patch_mode.append(modes[i][0])
    patch_mode.append(modes[i+1][0])
    final_array.append(mode(patch_mode)[0][0])

patch_mode = []
patch_mode.append(modes[-3][0])
patch_mode.append(modes[-2][0])
patch_mode.append(modes[-1][0])
final_array.append(mode(patch_mode)[0][0])

print("final_array", final_array)
'''








'''
patch_mode.append(modes_2_temp[0])
patch_mode.append(modes_2_temp[1])
patch_mode.append(modes_2_temp[2])
final_array.append(mode(patch_mode)[0][0])

for i in range(1, len(modes_2_temp)-1):
    patch_mode = []
    patch_mode.append(modes_2_temp[i-1])
    patch_mode.append(modes_2_temp[i])
    patch_mode.append(modes_2_temp[i+1])
    final_array.append(mode(patch_mode)[0][0])

patch_mode = []
patch_mode.append(modes_2_temp[-3])
patch_mode.append(modes_2_temp[-2])
patch_mode.append(modes_2_temp[-1])
final_array.append(mode(patch_mode)[0][0])

print("final_array", final_array)
'''