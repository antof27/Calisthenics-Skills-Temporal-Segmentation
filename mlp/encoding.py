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
