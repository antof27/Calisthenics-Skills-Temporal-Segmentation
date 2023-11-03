from matplotlib import pyplot as plt

def predicted_plotting(output, output_l, ax1, ax2):
    x = []
    y = []
    x1 = []
    y1= []

    for i in range(0, len(output)):
        x.append(output[i][1])
        x.append(output[i][2])
        y.append(output[i][0])
        y.append(output[i][0])

    ax1.plot(x, y)
    ax1.set_title("Predicted - frames")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Skills")
    
    seconds_length = 0
    for i in range(0, len(output_l)):
        x1.append(seconds_length)
        x1.append(seconds_length+output_l[i][1])
        y1.append(output_l[i][0])
        y1.append(output_l[i][0])
        seconds_length+=output_l[i][1]

    print("x1", x1)
    print("y1", y1)

    ax2.plot(x1, y1, color='orange')
    ax2.set_title("Predicted - seconds")
    ax2.set_xlabel("Seconds")
    ax2.set_ylabel("Skills")

    return seconds_length


def gt_plotting(skills_frames_, skills_seconds_, ax3, ax4):
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    k = 0

    for i in range(0, len(skills_frames_)):
        x2.append(skills_frames_[i][1])
        x2.append(skills_frames_[i][2])
        y2.append(skills_frames_[i][0])
        y2.append(skills_frames_[i][0])
    ax3.plot(x2, y2)
    ax3.set_title("Ground truth - frames")
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("Skills")

    for i in range(0, len(skills_seconds_)):
        x3.append(k)
        x3.append(k+skills_seconds_[i][1])
        y3.append(skills_seconds_[i][0])
        y3.append(skills_seconds_[i][0])
        k+=skills_seconds_[i][1]

    print("x1", x2)
    print("y1", y2)

    ax4.plot(x3, y3, color='red')
    ax4.set_title("Ground truth - seconds")
    ax4.set_xlabel("Seconds")
    ax4.set_ylabel("Skills")

