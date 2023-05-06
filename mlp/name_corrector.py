def name_corrector(name):
    #print("Name inserted: ", name)
    if name.startswith(" "):
        name = name[1:]
    if name.endswith(" "):
        name = name[:-1]

    if name == "planche" or name == "pl":
        name = "pl"
    elif name == "one arm handstand" or name == "one-arm-handstand" or name == "one_arm_handstand" or name == "oahs":
        name = "oahs"
    elif name == "front lever" or name == "front-lever" or name == "front_lever" or name == "front" or name == "frontlever" or name == "fl":
        name = "fl"
    elif name == "back lever" or name == "back-lever" or name == "back_lever" or name == "bl":
        name = "bl"
    elif name == "one arm front lever" or name == "one-arm-front-lever" or name == "one_arm_front_lever" or name == "oafl":
        name = "oafl"
    elif name == "human flag" or name == "human_flag" or name == "hf" or name == "human-flag" or name == "flag":
        name = "flag"
    elif name == "iron cross" or name == "iron_cross" or name == "iron-cross" or name == "cross" or name == "ic":
        name = "ic"
    elif name == "maltese" or name == "mal":
        name = "mal" 
    else:
        print("The name you inserted is not correct, please insert the correct name")
        name = input()
        name = name_corrector(name)
    #print("Name corrected: ", name, "")        
    return name
