import seaborn as sns
import os
import json
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)
# Initialize a list to store the data from each JSON file
temp_recall = []
temp_precision = []
temp_f1 = []
temp_loss_list = []
temp_conf_matrix = []
temp_accuracy = []
temp_raw_results = []
temp_raw_value = []
temp_vsr_results = []
temp_vsr_value = []
temp_paper_results = []
temp_paper_value = []
temp_fusion_results = []
temp_fusion_value = []
temp_average_windows = []
temp_edge_accuracy = []

mean_recall = 0
mean_precision = 0
mean_f1 = 0
mean_loss_list = []
mean_conf_matrix = []
mean_accuracy = 0
mean_raw_results = []
mean_raw_value = []
mean_vsr_results = []
mean_vsr_value = []
mean_paper_results = []
mean_paper_value = []
mean_fusion_results = []
mean_fusion_value = []
mean_average_windows = []
mean_edge_accuracy = []



environment = os.getcwd()
environment = os.path.join(environment,"models_results/conf_mat_updated")

#iterate on environment folder
for files in os.listdir(environment):
    file_path = os.path.join(environment, files)

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get the data from the JSON file
    temp_recall.append(data["recall"])
    temp_precision.append(data["precision"])
    temp_f1.append(data["f1"])
    temp_loss_list.append(data["loss"])
    temp_conf_matrix.append(data["confusion_matrix"])
    temp_accuracy.append(data["accuracy"])
    temp_raw_results.append(data["raw_results"])
    temp_raw_value.append(data["raw_value"])
    temp_vsr_results.append(data["vsr_results"])
    temp_vsr_value.append(data["vsr_value"])
    temp_paper_results.append(data["paper_results"])
    temp_paper_value.append(data["paper_value"])
    temp_average_windows.append(data["means_windows"])
    temp_edge_accuracy.append(data["accuracy_edge_raw"])


# Calculate the mean of the data values
mean_recall = np.mean(temp_recall)
mean_precision = np.mean(temp_precision)
mean_f1 = np.mean(temp_f1)
mean_loss_list = np.mean(temp_loss_list, axis=0)
mean_conf_matrix = np.mean(temp_conf_matrix, axis=0)
mean_accuracy = np.mean(temp_accuracy)
mean_raw_results = np.mean(temp_raw_results, axis=0)
mean_raw_value = np.mean(temp_raw_value, axis=0)
mean_asf1_raw = np.mean(mean_raw_value)
mean_vsr_results = np.mean(temp_vsr_results, axis=0)
mean_vsr_value = np.mean(temp_vsr_value, axis=0)
mean_asf1_vsr = np.mean(mean_vsr_value)
mean_paper_results = np.mean(temp_paper_results, axis=0)
mean_paper_value = np.mean(temp_paper_value, axis=0)
mean_asf1_paper = np.mean(mean_paper_value)
mean_average_windows = np.mean(temp_average_windows, axis=0)
mean_edge_accuracy = np.mean(temp_edge_accuracy, axis=0)

#save the results in a json file
results = {
    "recall": mean_recall,
    "precision": mean_precision,
    "f1": mean_f1,
    "loss_list": mean_loss_list.tolist(),
    "confusion_matrix": mean_conf_matrix.tolist(),
    "accuracy": mean_accuracy,
    "raw_results": mean_raw_results.tolist(),
    "raw_value": mean_raw_value.tolist(),
    "asf1_raw": mean_asf1_raw,
    "vsr_results": mean_vsr_results.tolist(),
    "vsr_value": mean_vsr_value.tolist(),
    "asf1_vsr": mean_asf1_vsr,
    "paper_results": mean_paper_results.tolist(),
    "paper_value": mean_paper_value.tolist(),
    "asf1_paper": mean_asf1_paper,
    "means_windows": mean_average_windows.tolist(),
    "accuracy_edge_raw": mean_edge_accuracy.tolist()
}

with open('mean_results_ideal_2.json', 'w') as f:
    json.dump(results, f)




#create a plot for the loss
f1, ax8 = plt.subplots(1, 1, figsize=(10, 6))
x3 = np.linspace(1, len(mean_loss_list), len(mean_loss_list))
y3 = mean_loss_list
ax8.plot(x3, y3)
ax8.set_xlabel('Epochs')
ax8.set_ylabel('Loss')
ax8.set_title('Loss')
f1.tight_layout()
f1.savefig('loss.png', dpi=300)


#--------------------------CONFUSION MATRIX-------------------------------
#convert the confusion matrix to scientific to standard notation




#mean_conf_matrix = np.mean(temp_conf_matrix, axis=0)


conf_mat = mean_conf_matrix * 100

conf_mat_percent = np.around((conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]), decimals=4)

# Display it with percentage symbols
f3, ax10 = plt.subplots(1, 1, figsize=(15, 15))
    
sns.heatmap(conf_mat_percent, annot=True, fmt=".2%", cmap="Blues", ax=ax10, xticklabels=le.classes_, yticklabels=le.classes_)
'''
# Display it
f3, ax10 = plt.subplots(1, 1, figsize=(10, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_mat_percent, 
    display_labels=le.classes_,
)

disp.plot(cmap=plt.cm.Blues, ax=ax10, colorbar=True, xticks_rotation=90)
'''
ax10.set_xlabel('Predicted label')
ax10.set_ylabel('True label')
ax10.set_title('Confusion matrix')

f3.tight_layout()
f3.savefig('confusion_matrix.png', dpi=300)

#--------------------------ACCURACY-------------------------------

print("Accuracy: ", mean_accuracy)
print("Recall: ", mean_recall)
print("Precision: ", mean_precision)
print("F1: ", mean_f1)
print("mean_average_windows: ", mean_average_windows)

#--------------------------EDGE ANALYZER-------------------------------
f4, ax11 = plt.subplots(1, 1, figsize=(10, 6))
x5 = np.linspace(1, len(mean_edge_accuracy), len(mean_edge_accuracy))
y5 = mean_edge_accuracy
ax11.bar(x5, y5)
ax11.set_xlabel('Edge distances', fontsize=14)
ax11.set_ylabel('Local accuracy', fontsize=14)
ax11.set_ylim(0.5, 0.7)
ax11.tick_params(axis='x', labelsize=14)  # You can adjust the fontsize as needed
ax11.tick_params(axis='y', labelsize=14)
f4.tight_layout()
f4.savefig('accuracy_edge_raw.png', dpi=300)

#--------------------------METRICS-------------------------------

print("raw_results: ", mean_raw_results)
print("raw_value: ", mean_raw_value)
average_raw_value = np.mean(mean_raw_value)
print("raw_value_mean: ", average_raw_value)

print("vsr_results: ", mean_vsr_results)
print("vsr_value: ", mean_vsr_value)
average_vsr_value = np.mean(mean_vsr_value)
print("vsr_value_mean: ", average_vsr_value)

print("paper_results: ", mean_paper_results)
print("paper_value: ", mean_paper_value)
average_paper_value = np.mean(mean_paper_value)
print("paper_value_mean: ", average_paper_value)

max_window = 26
#plot the means
x7 = np.linspace(6, max_window, max_window-6)
y7 = mean_average_windows

fig5 = plt.figure()
plt.plot(x7, y7, color = 'blue', label='mASF1')
plt.legend(loc='lower left')

plt.scatter(x7[np.argmax(y7)], y7[np.argmax(y7)], color='red', marker='*', s=100)
# Add axis labels and title

x7_even = x7.astype(int)
x7_even = x7_even[x7_even % 2 == 0]
plt.xticks(x7_even)

plt.xlabel('Windows_size')
plt.ylabel('SF1 Score')
#plt.title('Windows_size - mASF1 curves')
fig5.savefig('window_size_comparison.png', dpi=300)

# PLOT THE SF1 CURVES
x = np.linspace(0, 1, 100)
y1 = mean_paper_results
y2 = mean_vsr_results
y3 = mean_raw_results

# Create the plot
fig = plt.figure()
plt.plot(x, y1, color = 'blue', label='Probabilistic')
plt.plot(x, y2, color = 'orange', label='Heuristic')
plt.plot(x, y3, color = 'green', label='Raw')

plt.legend(loc='lower left')

# Add axis labels and title
plt.xlabel('Threshold')
plt.ylabel('SF1 Score')
#plt.title('Threshold - SF1 curves')
fig.savefig('threshold-sf1.png', dpi=300)



#PLOT THE SF1 VALUES
x1 = np.linspace(0, 1, len(mean_paper_value))
y3 = mean_paper_value
y4 = mean_vsr_value
y5 = mean_raw_value

# Create the plot
fig2 = plt.figure()
y3_label = 'PLS (mASF1: ' + str(round(average_paper_value, 2)) + ')'
plt.plot(x1, y3, color = 'blue', label= y3_label, marker='o')
y4_label = 'VSR (mASF1: ' + str(round(average_vsr_value, 2)) + ')'
plt.plot(x1, y4, color = 'orange', label= y4_label, marker='v')
y5_label = 'RAW (mASF1: ' + str(round(average_raw_value, 2)) + ')'
plt.plot(x1, y5, color = 'green', label= y5_label, marker='s')

plt.legend(loc='lower left')


# Add axis labels and title
plt.xlabel('Threshold')
plt.ylabel('ASF1 Score')
#plt.title('Threshold - mASF1 curves')
fig2.savefig('threshold-masf1.png', dpi=300)

plt.show()


