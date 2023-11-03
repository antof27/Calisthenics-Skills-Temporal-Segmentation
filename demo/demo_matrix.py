from PIL import Image, ImageDraw, ImageFont
import os

def decoding(idx):
    skill_colors = [(220, 20, 60), (21, 176, 26), (3, 67, 223), (255, 170, 0), (19, 185, 191), (60, 60, 60), (169,86,30), (249, 115, 6), (218, 112, 214), (128, 0, 128)]
    if idx == 0:
        clr = skill_colors[0]
    elif idx == 1:
        clr = skill_colors[1]
    elif idx == 2:
        clr = skill_colors[2]
    elif idx == 3:
        clr = skill_colors[3]
    elif idx == 4:
        clr = skill_colors[4]
    elif idx == 5:
        clr = skill_colors[5]
    elif idx == 6:
        clr = skill_colors[6]
    elif idx == 7:
        clr = skill_colors[7]
    elif idx == 8:
        clr = skill_colors[8]
    elif idx == 9:
        clr = skill_colors[9]

    return clr


def encoding(item):
    if item == 0:
        item = "bl"
    elif item == 1:
        item = "fl"
    elif item == 2:
        item = "flag"
    elif item == 3:
        item = "ic"
    elif item == 4:
        item = "mal"
    elif item == 5:
        item = "none"
    elif item == 6:
        item = "oafl"
    elif item == 7:
        item = "oahs"
    elif item == 8:
        item = "pl"
    elif item == 9:
        item = "vsit"
    return item

def draw_horizontal_bars(matrix, est_list, gt_list):
    # Dimensions of the image
    image_width = 450
    image_height = 540
    environment = os.getcwd()
    font_size = 25
    font_size2 = 20
    # Number of bars and height of each bar
    font_path = environment + "/arial.ttf"
    font = ImageFont.truetype(font_path, font_size)
    font2 = ImageFont.truetype(font_path, font_size2)
    raw_text = "Raw probabilities"
    est_text = "Heuristic"
    gt_text = "GT"
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((10, 5), raw_text, fill=(0,0,0), font=font)
    draw.text((260, 5), est_text, fill=(0,0,0), font=font)
    draw.text((390, 5), gt_text, fill=(0,0,0), font=font)
    
    num_bars = len(matrix[0])
    
    bar_height = (image_height-50) // num_bars

    # Create an image with white background
    

    # Draw bars for each row
    for row_idx, row in enumerate(matrix):
        # Draw bars for each column
        #draw a circle for the estimated value
        color_e = decoding(est_list)
        shape = [(300, est_list*bar_height+62), (324, est_list*bar_height+86)]
        draw.ellipse(shape, color_e)

        color_gt = decoding(gt_list)
        shape2 = [(394, gt_list*bar_height+62), (418, gt_list*bar_height+86)]
        draw.ellipse(shape2, color_gt)


        for col_idx, value in enumerate(row):
            # Calculate the width of the bar
            bar_width = int(value * (image_width-200))

            # Calculate the position of the bar
            x1 = 5
            y1 = col_idx * bar_height +50
            x2 = bar_width+5
            y2 = (col_idx + 1) * bar_height+40

            # Draw the bar
            color = decoding(col_idx)
            draw.rectangle([(x1, y1), (x2, y2)], fill=color)
            #draw the text of the skill
            text_encoded = encoding(col_idx)
            text_encoded = text_encoded.upper()
            draw.text((x1+10, y1+7), text_encoded, fill=(255,255,255), font=font)

    return image
'''
# Example usage
matrix = [
    [0.1, 0.05, 0.5, 0.02, 0.1, 1],
    [4.8928469e-1, 6.1021371e-16, 1.8277907e-14, 2.1894228e-15, 2.1912887e-09, 2.4131657e-15],
    [7.9680637e-11, 4.9604531e-16, 1.1222097e-14, 7.0357717e-16, 8.1406737e-10, 1.8995133e-15],
    [5.3185567e-15, 1.1295596e-16, 3.7382036e-16, 9.6017569e-11, 7.7980032e-11, 9.5400077e-17],
    [7.7963183e-15, 1.1918470e-16, 4.6175255e-16, 1.2543247e-10, 4.1831982e-11, 9.2644248e-17],
    [8.2702984e-15, 3.4756981e-16, 7.8200078e-16, 1.3178444e-10, 9.5712521e-11, 2.0128077e-16]
]

est_list = [0,0,0,0,5,5]
gt_list = [5,0,0,3,4,5]

# Create the folder if it doesn't exist

folder_path = "./demo/matrix/"

# Generate images with horizontal bars and save them in the folder
for i, row in enumerate(matrix):
    image = draw_horizontal_bars([row], est_list, gt_list)
    image.save(os.path.join(folder_path, f"image_{i}.png"))
'''