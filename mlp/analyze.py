import numpy as np
'''
bl 0
fl 1
flag 2
ic 3
mal 4
none 5
oafl 6
oahs 7
pl 8
'''


def edge_analyzer(l1):
#save the index of the first element not equal to 5

    first_index = 0
    last_index = 0

    if l1[0] == 5:
        for i in range(len(l1)):
            if l1[i] != 5:
                first_index = i
                break
        

    if l1[-1] == 5:
        for i in range(len(l1)-1, -1, -1):
            if l1[i] != 5:
                last_index = i
                break
 
    #initialize l2 with the same length as l1
    if first_index == 0 and last_index == 0:
        return

    if first_index == 0 and last_index != 0:
        first_index = last_index

    if first_index != 0 and last_index == 0:
        last_index = first_index
    
    print("First index", first_index)
    print("Last index", last_index)
    l2 = index_list_filler(l1, first_index, last_index)
    return l2


        
def index_list_filler(l1, first_index, last_index):
    l2 = np.ones(len(l1))
    l2[first_index] = 0
    l2[last_index] = 0


    for i in range(len(l2)):
        for j in range(i, first_index):
            l2[j] = first_index - j

        k = first_index+1
        y = last_index-1
        fill = 1
        
        while k <= y:
            l2[k] = fill
            l2[y] = fill
            fill+=1
            k+=1
            y-=1

        for h in range(last_index+1, len(l2)):
            l2[h] = h - last_index
    
    return l2
  


def range_accuracy_checker(raw_predicted, gt_test, lista_finale, interval):
    total = 0
    true = 0
    max_interval = 0
    for i in range(len(lista_finale)):
        if lista_finale[i] == 0.0:
            
            counter = 1
            while counter <= interval:
                
                #print("i-counter", gt_test[i-counter], "[",i-counter,"]", "i+counter", gt_test[i+counter], "[",i+counter,"]")
                #check if the index is in the range
                if i-counter >= 0 and i+counter < len(lista_finale):
                    if lista_finale[i-counter] == lista_finale[i+counter]:
                        #print("lista[i-counter]", lista_finale[i-counter], "lista[i+counter]", lista_finale[i+counter]," minore di ", interval)
                        
                        if raw_predicted[i-counter] == gt_test[i-counter]:
                            true+=1
                        
                        if raw_predicted[i+counter] == gt_test[i+counter]:
                            true+=1
                        
                        total +=2
                    else:
                        max_interval = counter
                        
                        
                        break
                    counter+=1

    return true/total, max_interval
                    

            
#call the function with the range
def analyzer(raw_predicted, gt_test, limit = 10):
    
    accuracy_list = np.zeros(limit)
    #build the lista_finale
    lista_finale = edge_analyzer(gt_test)
    print("Lista finale: ", lista_finale)
    print("Lista gt: ", gt_test)
    print("Lista raw: ", raw_predicted)

    for j in range(1, limit):
        accuracy, max_interval = range_accuracy_checker(raw_predicted, gt_test, lista_finale, j)
        if j >= max_interval and max_interval != 0:
            break

        accuracy_list[j-1]=accuracy

    print("Accuracy list: ", accuracy_list)

    return accuracy_list
    



'''
gt_test = [5,5,5,5,5,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5] 
raw_predicted = [5,5,5,5,5,6,7,6,6,7,6,6,7,6,5,5,5,5,5,5,5,5,5]
accuracy_list = analyzer(raw_predicted, gt_test)
'''



    










