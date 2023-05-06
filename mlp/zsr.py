import pandas as pd


def zsr_algorithm(dataframe_local):

    local_dataframe_output = pd.DataFrame()

    for col in dataframe_local.columns:
        #trasform the column into a list
        l1 = dataframe_local[col].tolist()
        
        if col == 75 or col == 76 or col == 77:

            local_dataframe_output[col] = l1
            continue
                    
        n_zeros = 0
        first_number = None
        last_number = None
        new_list = []

        # manage the zeros sequences at the beginning of the list
        while len(l1) > 0 and l1[0] == 0:
            new_list.append(0)
            l1.pop(0)

        for i in range(len(l1)):
            if l1[i] == 0:
                n_zeros += 1
                
                if n_zeros == 1:
                    first_number = l1[i-1]
            else:
                if n_zeros == 0:
                    new_list.append(l1[i])
                else:
                    last_number = l1[i]
                    range_ = last_number - first_number 
                    if range_ < 0.1 and n_zeros < 5:
                        step = (last_number - first_number) / (n_zeros + 1)
                        for j in range(1, n_zeros + 1):
                            new_list.append(first_number + j * step)
                        new_list.append(last_number)
                    else:
                        for j in range(1, n_zeros + 1):
                            new_list.append(0)
                        new_list.append(last_number)
                        
                    n_zeros = 0
        # manage the zeros sequences at the ending of the list
        while len(l1) > 0 and l1[-1] == 0:
            new_list.append(0)
            l1.pop()


        local_dataframe_output[col] = new_list

    return local_dataframe_output