def name_checker(name):
    names = ['flag', 'fl', 'bl', 'oafl', 'oahs', 'pl', 'ic', 'mal', 'vsit']
    if name.startswith(" "):
        name = name[1:]
    if name.endswith(" "):
        name = name[:-1]
        
    if name not in names:
        print("The name you inserted is not correct, please insert the correct name")
        print("Choose one between these : {'flag', 'fl', 'bl', 'oafl', 'oahs', 'pl', 'ic', 'mal', 'vsit'}")
        name = input()
        name = name_checker(name)
    return name

