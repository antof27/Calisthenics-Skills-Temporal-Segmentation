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
        elif list[i] == "vsit":
            list[i] = 9
    return list

def decoding(list):
    for i in range (len(list)):
        if list[i] == 0:
            list[i] = "bl"
        elif list[i] == 1:
            list[i] = "fl"
        elif list[i] == 2:
            list[i] = "flag"
        elif list[i] == 3:
            list[i] = "ic"
        elif list[i] == 4:
            list[i] = "mal"
        elif list[i] == 5:
            list[i] = "none"
        elif list[i] == 6:
            list[i] = "oafl"
        elif list[i] == 7:
            list[i] = "oahs"
        elif list[i] == 8:
            list[i] = "pl"
        elif list[i] == 9:
            list[i] = "vsit"
    return list