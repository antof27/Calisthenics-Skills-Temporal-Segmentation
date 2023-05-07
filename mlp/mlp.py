import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import random as rd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from vsr import vsr_algorithm
from paper_code import viterbi, SF1
from augment_function import augmented



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
    def __init__(self, input_size = 75, hidden_units = 1024, num_classes = 9, activation_function=nn.LeakyReLU()):
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
    data = pd.read_csv('dataset_elaboratedv5SI.csv')

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

    X_train.to_csv('Xtrain_output.csv', index=False)
    y_train_df = pd.DataFrame(y_train, columns=['skill_id'])
    y_train_df.to_csv('Ytrain_output.csv', index=False)

    X_test.to_csv('Xtest_output.csv', index=False)
    y_test_df = pd.DataFrame(y_test, columns=['skill_id'])
    y_test_df.to_csv('Ytest_output.csv', index=False)
    '''

    # Set the hyperparameters
    input_size = len(data.columns) - 3 # exclude 'id_video', 'frame', 'skill_id'
    hidden_units = 1024
    num_classes = len(data['skill_id'].unique())
    lr = 0.001
    n_epochs = 10
    batch_size = 512


    model = MLP(input_size, hidden_units, num_classes)
    criterion = nn.CrossEntropyLoss()

    #criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.RMSprop(model.parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.Adagrad(model.parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    '''
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    '''
     # Load the training data
    
    X_train = pd.read_csv('Xtrain_output.csv')
    y_train = pd.read_csv('Ytrain_output.csv')
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), 
                                  torch.LongTensor(y_train['skill_id'].values).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    



    raw_pred_test = []
    vsr_pred_test = []
    viterbi_pred_test = []
    gt_test = []

    model.to(device)
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
           
            if i % batch_size == batch_size - 1:
                acc = 100 * correct / total
                #print the loss and the accuracy for batch_size = 512

                print('[Epoch %d, Batch %d] Loss: %.3f Accuracy: %.3f%%' % 
                    (epoch + 1, i + 1, running_loss / batch_size, acc))
                writer.add_scalar('Loss/train', running_loss / batch_size, epoch * len(train_loader) + i)
                writer.add_scalar('Accuracy/train', acc, epoch * len(train_loader) + i)

                running_loss = 0.0
                correct = 0
                total = 0
     


    # Test the model
    print("Now testing the model...")
    correct = 0
    total = 0

    
    # Load the test data
    X_test = pd.read_csv('Xtest_output.csv')
    y_test = pd.read_csv('Ytest_output.csv')
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), 
                                 torch.LongTensor(y_test['skill_id'].values).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    '''
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), torch.LongTensor(y_test).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    '''
    y_pred = []
    y_true = []
    test_writer = SummaryWriter('runs/test')

    with torch.no_grad():
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(test_loader):
            
            #if the input row has a lot of zeros, it is not considered
            if ((inputs == 0).sum() / inputs.numel()) > 0.5:
                continue

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()

            
            gt_labels = labels.tolist()
            raw_predicted = predicted.tolist()
            #check if raw_predicted is an empty list
            
            print("raw_predicted: ", raw_predicted)
            vsr_output = vsr_algorithm(raw_predicted)  
            vsr_predicted = vsr_output[0]          
            viterbi_predicted = viterbi(probabilities, 10e-20)
            

            gt_test.extend(gt_labels)
            raw_pred_test.extend(raw_predicted)
            vsr_pred_test.extend(vsr_predicted)
            viterbi_pred_test.extend(viterbi_predicted)
            
            
            #put them into a csv
            temp = pd.DataFrame({'gt_labels': gt_labels, 'raw_predicted': raw_predicted})

            temp.to_csv('temp_seeing.csv', mode='a', header=False, index=False)


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = loss.item()
            acc = 100 * correct / total

            #print all the curve in tensorboard
            test_writer.add_scalar('Loss/test', running_loss / batch_size, epoch * len(test_loader) + i)
            test_writer.add_scalar('Accuracy/test', acc, i)

            

    print('Accuracy on test data: %d %%' % (100 * correct / total))

    #confusion 
    conf_mat = confusion_matrix(gt_test, raw_pred_test)

    # Display it
    f,ax = plt.subplots(1,1,figsize=(15,15))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, 
        display_labels= le.classes_
    )

    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix')

    f.tight_layout()
    f.savefig('confusion_matrix.png', dpi=300)

    
    # METRIC CALCULATION
    raw_results, raw_value = SF1(gt_test, raw_pred_test)
    print("raw_results: ", raw_results)
    print("raw_value: ", raw_value)
    average_raw_value = np.mean(raw_value)
    print("raw_value_mean: ", average_raw_value)


    vsr_results, vsr_value = SF1(gt_test, vsr_pred_test)
    print("vsr_results: ", vsr_results)
    print("vsr_value: ", vsr_value)
    average_vsr_value = np.mean(vsr_value)
    print("vsr_value_mean: ", average_vsr_value)

    paper_results, paper_value = SF1(gt_test, viterbi_pred_test)
    print("paper_results: ", paper_results)
    print("paper_value: ", paper_value)
    average_paper_value = np.mean(paper_value)
    print("paper_value_mean: ", average_paper_value)

   


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

    plt.show()



    np.save('classes.npy', le.classes_)

    torch.save(model.state_dict(), 'model.pth')
    
if __name__ == '__main__':
    main()
