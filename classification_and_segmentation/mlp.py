import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
import random as rd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from vsr import vsr_algorithm
from algorithms_fusion import combining
#from updated_vsr import vsr_algorithm
from paper_code import viterbi, SF1
from augment_function import augmented
from analyze import analyzer
import json
import csv
import seaborn as sns


writer = SummaryWriter('runs/MLP')

#----------------- set seed ------------------
torch.manual_seed(42)
np.random.seed(42)


def hidden_blocks(input_size, output_size, activation_function):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activation_function,
    )


class MLP(nn.Module):
    def __init__(self, input_size = 75, hidden_units = 512, num_classes = 10, activation_function=nn.LeakyReLU()):
        super(MLP, self).__init__()
        
        self.architecture = nn.Sequential(
            hidden_blocks(input_size, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function), 
            hidden_blocks(hidden_units, hidden_units, activation_function),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self,x):
        return self.architecture(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Current device:", device)

    le = LabelEncoder()
    data = pd.read_csv('dataset_v8.50.csv')

    X = data.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
    y = data['skill_id']

    #encode the labels
    y = le.fit_transform(y)

    '''
    print(le.classes_)
    for i in range(len(le.classes_)):
        print(le.classes_[i], i)
    '''
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #splitting and grouping by video_name
    train_idx, test_idx = next(gss.split(X, y, groups=data['video_name']))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    '''
    print("Starting data augmentation...")
    X_train, y_train = augmented(X_train, y_train)
    print("Dataset augmented!")
    '''
    X_train.to_csv('Xtrain_output.csv', index=False)
    y_train_df = pd.DataFrame(y_train, columns=['skill_id'])
    y_train_df.to_csv('Ytrain_output.csv', index=False)

    X_test.to_csv('Xtest_output.csv', index=False)
    y_test_df = pd.DataFrame(y_test, columns=['skill_id'])
    y_test_df.to_csv('Ytest_output.csv', index=False)
    

    # Set the hyperparameters
    input_size = len(data.columns) - 3 # exclude 'id_video', 'frame', 'skill_id'
    hidden_units = 512
    num_classes = len(data['skill_id'].unique())
    lr = 0.0001
    n_epochs = 500
    batch_size = 512


    model = MLP(input_size, hidden_units, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.RMSprop(model.parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.Adagrad(model.to(device).parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
     # Load the training data
    '''
    X_train = pd.read_csv('x_train_fold_3.csv')
    y_train = pd.read_csv('y_train_fold_3.csv')
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), 
                                  torch.LongTensor(y_train['skill_id'].values).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    '''



    raw_pred_test = []
    vsr_pred_test = []
    fusion_pred_test = []
    loss_list = []
    viterbi_pred_test = []
    gt_test = []

    print("Now training the model...")
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
           
            if (i + 1) % int(batch_size/4) == 0:
                acc = 100 * correct / total
                #print the loss and the accuracy for batch_size = 512
                loss_list.append(running_loss / batch_size)
                print('[Epoch %d, Batch %d] Loss: %.3f Accuracy: %.3f%%' % 
                    (epoch + 1, i + 1, running_loss / batch_size, acc))
                #writer.add_scalar('Loss/train', running_loss / batch_size, epoch * len(train_loader) + i)
                #writer.add_scalar('Accuracy/train', acc, epoch * len(train_loader) + i)

                running_loss = 0.0
                correct = 0
                total = 0
        
        
    # Test the model
    print("Now testing the model...")
    correct = 0
    total = 0

    '''
    # Load the test data
    X_test = pd.read_csv('x_test_fold_3.csv')
    y_test = pd.read_csv('y_test_fold_3.csv')
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), 
                                 torch.LongTensor(y_test['skill_id'].values).to(device))
    test_loader = DataLoader(test_dataset, int(batch_size/2), shuffle=False)
    '''    
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), torch.LongTensor(y_test).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    y_pred = []
    y_true = []
    
    limit = 6
    accuracy_list_raw = [[0, 0] for _ in range(limit)]
    
    max_window = 26
    list_of_temp_vsr = [[] for _ in range(max_window-6)] 
    list_of_total_vsr = [[] for _ in range(max_window-6)]


    test_writer = SummaryWriter('runs/test')

    with torch.no_grad():
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(test_loader):
            
            #if the input row has a lot of zeros, it is not considered

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()

            
            gt_labels = labels.tolist()
            raw_predicted = predicted.tolist()
            #check if raw_predicted is an empty list
            
            #create a csv with the first column of ground truth labels and the second column of raw predicted labels
            '''
            with open('confusion_raw_predicted.csv', 'a') as f:
                writer_raw = csv.writer(f)
                writer_raw.writerows(zip(gt_labels, raw_predicted))
            '''            


            vsr_output = vsr_algorithm(raw_predicted, 13)  
            vsr_predicted = vsr_output[0]          
            viterbi_predicted = viterbi(probabilities, 10e-20)

            fusion_predicted = combining(vsr_predicted, viterbi_predicted)
            #print("listgt: ", gt_labels)

            

            gt_test.extend(gt_labels)
            raw_pred_test.extend(raw_predicted)
            vsr_pred_test.extend(vsr_predicted)
            viterbi_pred_test.extend(viterbi_predicted)
            fusion_pred_test.extend(fusion_predicted)
            

            #VSR window_size creation
            for w in range(0, max_window-6):
                vsr_current = vsr_algorithm(raw_predicted, w+6)
                vsr_current = vsr_current[0]

                list_of_temp_vsr[w] = vsr_current

            
            for y in range(len(list_of_temp_vsr)):
                list_of_total_vsr[y].extend(list_of_temp_vsr[y])

            
            #edge analyzer
            
            accuracy_temp_raw = analyzer(raw_predicted, gt_labels, limit)


            #if every element of accuracy_list_temp is 0, continue
            if all(elem == 0 for elem in accuracy_temp_raw):
                continue
                
            for i in range(len(accuracy_list_raw)):
                accuracy_list_raw[i][0] += accuracy_temp_raw[i][0]
                accuracy_list_raw[i][1] += accuracy_temp_raw[i][1]

            
            


            
            '''
            #put them into a csv
            temp = pd.DataFrame({'gt_labels': gt_labels, 'raw_predicted': raw_predicted})

            temp.to_csv('temp_seeing.csv', mode='a', header=False, index=False)
            '''

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = loss.item()
            acc = 100 * correct / total

            #print all the curve in tensorboard
            test_writer.add_scalar('Loss/test', running_loss / batch_size, epoch * len(test_loader) + i)
            test_writer.add_scalar('Accuracy/test', acc, i)

        
    data_loss_test = {"loss": loss_list}

    #calculate the recall, precision and f1 score

    recall_raw = recall_score(gt_test, raw_pred_test, average='weighted')    
    precision_raw = precision_score(gt_test, raw_pred_test, average='weighted')
    f1_raw = f1_score(gt_test, raw_pred_test, average='weighted')

    data_recall = {"recall": recall_raw}
    data_precision = {"precision": precision_raw}
    data_f1 = {"f1": f1_raw}

    
    

    
    #CONFUSION MATRIX
    #i want a normalized confusion matrix in percentage

    # Create your confusion matrix
    '''
    conf_mat = confusion_matrix(gt_test, raw_pred_test)

    conf_mat_percent = np.around((conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]), decimals=4)

    # Display it with percentage symbols
    f3, ax10 = plt.subplots(1, 1, figsize=(15, 15))
    
    sns.heatmap(conf_mat_percent, annot=True, fmt=".2%", cmap="Blues", ax=ax10, xticklabels=le.classes_, yticklabels=le.classes_)
    '''

    conf_mat = confusion_matrix(gt_test, raw_pred_test)

    conf_mat_percent = np.around((conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]), decimals=4)

    # Display it with percentage symbols
    f3, ax10 = plt.subplots(1, 1, figsize=(15, 15))
    
    sns.heatmap(conf_mat_percent, annot=True, fmt=".4f", cmap="Blues", ax=ax10, xticklabels=le.classes_, yticklabels=le.classes_)


    ax10.set_xlabel('Predicted label')
    ax10.set_ylabel('True label')
    ax10.set_title('Confusion matrix')

    f3.tight_layout()
    f3.savefig('confusion_matrix.png', dpi=300)
    
    data_conf = {"confusion_matrix": conf_mat_percent.tolist()}
    #write to a json file the confusion matrix values with key = 'confusion_matrix'
    


    



    
    #print("Accuracy list RAW: ", accuracy_list_raw)
    #EDGE ANALYZER
    accuracy_edge_raw = []
    for i in range(1, len(accuracy_list_raw)):
        if accuracy_list_raw[i][1] != 0:
            accuracy_edge_raw.append(accuracy_list_raw[i][0] / accuracy_list_raw[i][1])
    
    print("Accuracy edge RAW: ", accuracy_edge_raw)


    data_edge_analyzer = {"accuracy_edge_raw": accuracy_edge_raw}
    '''
    f4, ax11 = plt.subplots(1, 1, figsize=(10, 6))
    x5 = np.linspace(1, len(accuracy_edge_raw), len(accuracy_edge_raw))
    y5 = accuracy_edge_raw
    ax11.bar(x5, y5)
    ax11.set_xlabel('Edge distances')
    ax11.set_ylabel('Local accuracy')
    ax11.set_ylim(0.5, 0.7)
    ax11.set_title('Raw - Edge / Accuracy correlation')
    f4.tight_layout()
    f4.savefig('accuracy_edge_raw.png', dpi=300)
    '''
    #ACCURACY ON TEST DATA
    acc_test = 100 * correct / total
    print('Accuracy on test data: %d %%' % (acc_test))
    
    data_acc = {"accuracy": acc_test}
    #save the accuracy on a json file





    
    # METRIC CALCULATION
    raw_results, raw_value = SF1(gt_test, raw_pred_test)
    #print("raw_results: ", raw_results)
    #print("raw_value: ", raw_value)
    average_raw_value = np.mean(raw_value)
    #print("raw_value_mean: ", average_raw_value)

    data_raw_results = {"raw_results": raw_results.tolist()}
    data_raw_value = {"raw_value": raw_value.tolist()}

    #save the raw_results and raw_value on a json file



    vsr_results, vsr_value = SF1(gt_test, vsr_pred_test)
    #print("vsr_results: ", vsr_results)
    #print("vsr_value: ", vsr_value)
    average_vsr_value = np.mean(vsr_value)
    #print("vsr_value_mean: ", average_vsr_value)

    data_vsr_results = {"vsr_results": vsr_results.tolist()}
    data_vsr_value = {"vsr_value": vsr_value.tolist()}

    #save the vsr_results and vsr_value on a json file

    fusion_results, fusion_value = SF1(gt_test, fusion_pred_test)
    #print("vsr_results: ", vsr_results)
    #print("vsr_value: ", vsr_value)
    average_fusion_value = np.mean(fusion_value)
    #print("vsr_value_mean: ", average_vsr_value)

    data_fusion_results = {"fusion_results": fusion_results.tolist()}
    data_fusion_value = {"fusion_value": fusion_value.tolist()}

    #save the vsr_results and vsr_value on a json file




    paper_results, paper_value = SF1(gt_test, viterbi_pred_test)
    #print("paper_results: ", paper_results)
    #print("paper_value: ", paper_value)
    average_paper_value = np.mean(paper_value)
    #print("paper_value_mean: ", average_paper_value)

    data_paper_results = {"paper_results": paper_results.tolist()}
    data_paper_value = {"paper_value": paper_value.tolist()}

    #save the paper_results and paper_value on a json file



    
    #Metric calculation on VSR with different window sizes
    vsr_sf1_results = [[] for _ in range(max_window-6)]
    vsr_values = [0] * (max_window-6)
    means = [0] * (max_window-6)
    for l in range(len(list_of_total_vsr)):
        vsr_sf1_results[l] = SF1(gt_test, list_of_total_vsr[l])
        
        vsr_values[l] = vsr_sf1_results[l][1]
        #print("Window size: ", l+2)
        #print(vsr_values[l])
        #put the mean inside means list
        means[l] = np.mean(vsr_values[l])
        #print("Mean: ", means[l])
    
    data_vsr_window = {"means": means}
    #save the means on a json file
   


    '''
    #plot the means
    x7 = np.linspace(6, max_window, max_window-6)
    y7 = means
    

    fig5 = plt.figure()
    plt.plot(x7, y7, color = 'blue', label='mASF1')
    plt.legend(loc='lower left')
    
    plt.scatter(x7[np.argmax(y7)], y7[np.argmax(y7)], color='red', marker='*', s=100)
    # Add axis labels and title
    plt.xlabel('Windows_size')
    plt.ylabel('SF1 Score')
    plt.title('Windows_size - mASF1 curves')


   # PLOT THE SF1 CURVES
    x = np.linspace(0, 1, 100)
    y1 = paper_results
    y2 = vsr_results
    
    # Create the plot
    fig = plt.figure()
    plt.plot(x, y1, color = 'blue', label='PC')
    plt.plot(x, y2, color = 'orange', label='PROP')
    plt.legend(loc='lower left')

    # Add axis labels and title
    plt.xlabel('Threshold')
    plt.ylabel('SF1 Score')
    plt.title('Threshold - SF1 curves')
    



    #PLOT THE SF1 VALUES
    x1 = np.linspace(0, 1, len(paper_value))
    y3 = paper_value
    y4 = vsr_value
    
    # Create the plot
    fig2 = plt.figure()
    y3_label = 'PC (mASF1: ' + str(round(average_paper_value, 2)) + ')'
    plt.plot(x1, y3, color = 'blue', label= y3_label, marker='o')
    y4_label = 'PROP (mASF1: ' + str(round(average_vsr_value, 2)) + ')'
    plt.plot(x1, y4, color = 'orange', label= y4_label, marker='v')

    plt.legend(loc='lower left')


    # Add axis labels and title
    plt.xlabel('Threshold')
    plt.ylabel('SF1 Score')
    plt.title('Threshold - mASF1 curves')

    #plt.show()
    '''
    data = {
        "recall": recall_raw,
        "precision": precision_raw,
        "f1": f1_raw,
        "loss": loss_list,
        "confusion_matrix": conf_mat_percent.tolist(),
        "accuracy": acc_test,
        "raw_results": raw_results.tolist(),
        "raw_value": raw_value.tolist(),
        "vsr_results": vsr_results.tolist(),
        "vsr_value": vsr_value.tolist(),
        "fusion_results": fusion_results.tolist(),
        "fusion_value": fusion_value.tolist(),
        "paper_results": paper_results.tolist(),
        "paper_value": paper_value.tolist(),
        "means_windows": means,
        "accuracy_edge_raw": accuracy_edge_raw
    }
    

    with open('fold_4.json', 'w') as outfile:
        json.dump(data, outfile)
    
    
    


    np.save('classes.npy', le.classes_)

    torch.save(model.state_dict(), 'model.pth')
    
if __name__ == '__main__':
    main()