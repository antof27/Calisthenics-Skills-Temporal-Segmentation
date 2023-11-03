from PIL import Image, ImageDraw, ImageFont
import os 

def decoding(value):
    if value == 0:
        value = "bl"
    elif value == 1:
        value = "fl"
    elif value == 2:
        value = "flag"
    elif value == 3:
        value = "ic"
    elif value == 4:
        value = "mal"
    elif value == 5:
        value = "none"
    elif value == 6:
        value = "oafl"
    elif value == 7:
        value = "oahs"
    elif value == 8:
        value = "pl"
    elif value == 9:
        value = "vsit"
    return value

def get_color(item):
    
    skill_colors = [(220, 20, 60), (21, 176, 26), (3, 67, 223), (255, 170, 0), (19, 185, 191), (60, 60, 60), (169,86,30), (249, 115, 6), (218, 112, 214), (128, 0, 128)]
    if item == "bl" or item == 0:
        clr = skill_colors[0]
    elif item == "fl" or item == 1:
        clr = skill_colors[1]
    elif item == "flag" or item == 2:
        clr = skill_colors[2]
    elif item == "ic" or item == 3:
        clr = skill_colors[3]
    elif item == "mal" or item == 4:
        clr = skill_colors[4]
    elif item == "none" or item == 5:
        clr = skill_colors[5]
    elif item == "oafl" or item == 6:
        clr = skill_colors[6]
    elif item == "oahs" or item == 7:
        clr = skill_colors[7]
    elif item == "pl" or item == 8:
        clr = skill_colors[8]
    elif item == "vsit" or item == 9:
        clr = skill_colors[9]
    return clr


def print_element_with_counter(list1, list2):
    width = 410
    height = 540
    font_size = 30
    environment = os.getcwd()
    font_path = environment + "/arial.ttf"
    font = ImageFont.truetype(font_path, font_size)
    classes_text = "Classes"
    gt_text = "Heuristic"
    est_text = "GT"
    bar_list = []
    l1 = []
    l2 = []
    
    val1_sec = 0
    val2_sec = 0
    counter1 = 0
    counter2 = 0
   
    counter_val1 = 0
    counter_val2 = 0
    prev_value1 = 10
    prev_value2 = 10
    prev_bar = 10
    

    for i, (val1, val2) in enumerate(zip(list1, list2), start=0):
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        draw.text((10, 5), classes_text, fill=(0,0,0), font=font)
        draw.text((155, 5), gt_text, fill=(0,0,0), font=font)
        draw.text((310, 5), est_text, fill=(0,0,0), font=font)

        if val1 != 5 and val1 not in bar_list:
            bar_list.append(val1)
            #label = decoding(val1)
            
        elif val2 != 5 and val2 not in bar_list:
            bar_list.append(val2)
            #label = decoding(val2)
            
        
       
        if val1 != 5 and val1 != prev_value1:
            l1.append(val1)
            
        
        if val2 != 5 and val2 != prev_value2:
            l2.append(val2)
            


        #check if val1 is equal to the previous value
        if val1 != 5:
            if val1 == prev_value1:
                counter1 += 1
                val1_sec = counter1/24
                l1[counter_val1] = val1_sec

            else:
                #counter_val1 += 1
                counter1 = 1
                val1_sec = 1/24
                #print("len l1", len(l1))
                if len(l1) > 1:
                    counter_val1+=1
                l1[counter_val1] = val1_sec
            
        #check if val2 is equal to the previous value
        if val2 != 5:
            if val2 == prev_value2:
                counter2 += 1
                val2_sec = counter2/24
                l2[counter_val2] = val2_sec

            else:
                
                #counter_val2 += 1
                counter2 = 1
                val2_sec = 1/24
                if len(l2) > 1:
                    counter_val2+=1
                l2[counter_val2] = val2_sec

        
        y_bar = 75
        for j in range(len(bar_list)):

            color = get_color(bar_list[j])
            draw.rectangle([(5, y_bar), (400, y_bar+35)], color)


            label_text = decoding(bar_list[j])
            label_text = label_text.upper()
            draw.text((40, y_bar), label_text, fill=(250,250,250), font=font)
            y_bar += 75
        
        y1 = 75
        for k in range(len(l1)):
            v1 = round(l1[k], 2)
            text1 = str(v1)
            #print("text1", text1)
            draw.text((185, y1), text1, fill=(250, 250, 250), font=font)
            y1 += 75
        
        y2 = 75
        for l in range(len(l2)):
            v2 = round(l2[l], 2)
            text2 = str(v2)
            #print("text2", text2)
            draw.text((300, y2), text2, fill=(250, 250, 250), font=font)
            y2 += 75

        print("l1", l1)
        print("l2", l2)
        print("bar_list", bar_list)
        image.save(f"./demo/lists/image_{i:04}.png")
        prev_value1 = val1
        prev_value2 = val2
        prev_bar = val1
        
        
    
    
    

list1 = [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,7,7]
list2 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,7,7]


#print_element_with_counter(list1, list2)










        