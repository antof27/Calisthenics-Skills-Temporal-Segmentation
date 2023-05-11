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
def none_sequence(l1, k):
    first_index = 0
    last_index = 0
    
    if k-1 >= 0:
        first_index = k-1
    else:
        first_index = 0

    #read all the sequence of 5
    while k < len(l1) and l1[k] == 5:
        k+=1
    #save the index of the last element not equal to 5
    last_index = k

    return first_index, last_index

def edge_analyzer(l1):
    indeces = []
    k = 0
    while k < len(l1):
        
        if l1[k] != 5:
            k+=1
            continue

        f,l= none_sequence(l1, k)
        if f != 0 :
            indeces.append(f)
        if l != len(l1):
            indeces.append(l)
        
        
        
        k = l+1
        
    if len(indeces) == 0:
        return []
    
    l2 = index_list_filler(l1, indeces)
    return l2


        
def index_list_filler(l1, indeces):
    l2 = np.ones(len(l1))

    for i in range(len(l2)):
        if i in indeces:
            l2[i] = 0

    for i in range(len(l2)):

        for j in range(i, indeces[0]):
            l2[j] = indeces[0] - j
        
        for y in range(0, len(indeces)):
            s = indeces[y]+1
            #add a control to avoid out of range
            if y+1 >= len(indeces):
                break

            e = indeces[y+1]-1
            
            fill = 1
            while s <= e:
                
                if e >= len(l2):
                    break
                l2[s] = fill
                l2[e] = fill
                fill+=1
                s+=1
                e-=1
        
        
        
        for h in range(indeces[-1]+1, len(l2)):
            l2[h] = h - indeces[-1]
    
    return l2
  


def range_accuracy_checker(raw_predicted, gt_test, lista_finale, limit):
    
    total = 0
    true = 0
    max_interval = 0
    truetotal_list = [[0, 0] for _ in range(limit)]


    for i in range(len(lista_finale)):
        if lista_finale[i] == 0.0:
            
            counter = 0
            
            while counter < limit:
                total = 0
                true = 0
                if counter == 0:
                    #print("Compare : ", raw_predicted[i], " with ", gt_test[i])
                    if raw_predicted[i] == gt_test[i]:
                        true+=1
                    total+=1

                    truetotal_list[counter][0] += true
                    truetotal_list[counter][1] += total 
                    counter+=1
                    continue

                if i-counter >= 0 and i+counter < len(lista_finale):

                    if lista_finale[i-counter] == counter and lista_finale[i+counter] == counter:
                        
                        if raw_predicted[i-counter] == gt_test[i-counter]:
                            true+=1
                        
                        if raw_predicted[i+counter] == gt_test[i+counter]:
                            true+=1
                        
                        total +=2


                        truetotal_list[counter][0] += true
                        truetotal_list[counter][1] += total 

                    else:
                        max_interval = counter-1
                        break
                    counter+=1
                else:
                    max_interval = counter-1
                    break

    return truetotal_list
                    

            
#call the function with the wanted range
def analyzer(raw_predicted, gt_test, limit):
    accuracy_list = np.zeros(limit)
    lista_finale = edge_analyzer(gt_test)    
    if len(lista_finale) == 0:
        return accuracy_list
    
    truetotal_list = range_accuracy_checker(raw_predicted, gt_test, lista_finale, limit)

    return truetotal_list
 
'''
gt_ =  [5,5,5,5,5,5,5,6,6,6,6,6,6,6,5,5,5,5,5,5,3,3,3,3,3,3,3,3,5,5,5,5,5]
raw_ = [5,5,5,5,5,5,5,7,6,8,6,5,6,1,5,5,5,5,5,5,2,4,3,9,3,2,3,3,5,5,5,5,5]



print(raw_)
print(gt_)

acc1 = analyzer(raw_, gt_, 6)
print("Accuracy: ", acc1)
'''






    










