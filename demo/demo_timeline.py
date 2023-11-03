from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import os

def create_timeline_image(list_names, skills, current_frame):
    # Define the dimensions and colors for the image
    width = 1820  # Image width in pixels
    height = 290  # Image height in pixels
    bg_color = (255, 255, 255)  # Background color (white)
    legend_width = 80  # Width of the legend
    list_name_width = 120  # Width of the list name section

    # Define the skill names and colors
    skill_names = ["BL", "FL", "FLAG", "IC", "MAL", "NONE", "OAFL", "OAHS", "PL", "VSIT"]
    skill_colors = [(220, 20, 60), (21, 176, 26), (3, 67, 223), (255, 170, 0), (19, 185, 191), (60, 60, 60), (169, 86, 30), (249, 115, 6), (218, 112, 214), (128, 0, 128)]

    # Calculate the width of each frame based on the number of frames and the image width
    frame_width = (width - list_name_width - (legend_width)) // len(skills[0])

    # Create a new image with extended width
    image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    # Set the font for drawing text
    environment = os.path.dirname(os.path.abspath(__file__))  # Get the current file directory
    font_path = os.path.join(environment, "arial.ttf")
    font_size1 = 20
    font_size2 = 17
    font1 = ImageFont.truetype(font_path, size=font_size1)
    font2 = ImageFont.truetype(font_path, size=font_size2)

    # Draw the list names on the left side
    list_name_height = height // len(list_names)
    for i, name in enumerate(list_names):
        list_name_y = i * list_name_height
        draw.text((5, list_name_y+30), name, fill=(0, 0, 0), font=font1, angle=90)

    # Loop over each frame and assign the corresponding color
    for i, skills_per_frame in enumerate(zip(*skills)):
        # Calculate the position of the frame
        x = list_name_width + i * frame_width
        y = 0

        # Define the color and name based on the skill values
        for j, skill in enumerate(skills_per_frame):
            if skill >= 0 and skill < len(skill_names):
                color = skill_colors[skill]
                name = skill_names[skill]
            else:
                color = (0, 0, 0)  # Default color for invalid skill values
                name = ""

            frame_y = y + (j * (height // len(list_names)))
            frame_height = height // len(list_names)
            image.paste(color, (x, frame_y+2, x + frame_width, frame_y + frame_height-2))

        # Draw the cursor on the current frame
        if i == current_frame:
            cursor_x = x
            cursor_y = y
            cursor_height = height
            draw.rectangle([(cursor_x-1, cursor_y), (cursor_x + frame_width+1, cursor_y + cursor_height)], fill=(255, 255, 255))

    # Draw the legend on the right side
    legend_x = x + (frame_width + 2)
    legend_y = 0
    legend_height = height - 4
    draw.rectangle([(legend_x, legend_y+2), (width, legend_y + legend_height+2)], fill=bg_color)

    # Draw the legend labels
    label_height = legend_height // len(skill_colors)
    for i, color in enumerate(skill_colors):
        label_y = legend_y + (i * label_height)
        draw.rectangle([(legend_x, label_y+2), (width, label_y + label_height+2)], fill=color)
        draw.text((legend_x + label_height-15, label_y+5), skill_names[i], fill=(255, 255, 255), font=font2)

    return image