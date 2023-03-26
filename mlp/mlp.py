import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import random as rd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/MLP')


def hidden_blocks(input_size, output_size, activation_function):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activation_function,
    )


class MLP(nn.Module):
    def __init__(self, input_size = 75, hidden_units = 512, num_classes = 9, activation_function=nn.LeakyReLU()):
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

    print(le.classes_)
    #print all the unique labels in y
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #splitting and grouping by video_name
    train_idx, test_idx = next(gss.split(X, y, groups=data['video_name']))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    def augmented(X_train, y_train):
        X_train_augmented = X_train.copy()
        y_train_augmented = y_train.copy()

        for _, row in X_train_augmented.iterrows(): 
            augmentation = rd.choice(['mirror', 'shift'])
            
            if augmentation == 'mirror':
                horizzontal_mirror = [k for k in range(len(X_train_augmented.columns)) if k % 3 == 0 and row[k] != 0]
                row[horizzontal_mirror] = 1 - row[horizzontal_mirror]

            else:
                shift_factor = rd.uniform(-0.10, 0.10)
                shift = [k for k in range(len(X_train_augmented.columns)) if row[k] != 0 and k % 3 != 2]
                row[shift] += shift_factor
                row[row < 0] = 0
                row[row > 1] = 0
 
        X_train_augmented = X_train_augmented.append(X_train)
        y_train_augmented = np.append(y_train_augmented, y_train)
        
        return X_train_augmented, y_train_augmented


    X_train, y_train = augmented(X_train, y_train)
    
    X_train.to_csv('Augmented_dataset.csv', index=False)


    # Set the hyperparameters
    input_size = len(data.columns) - 3 # exclude 'id_video', 'frame', 'skill_id'
    hidden_units = 512
    num_classes = len(data['skill_id'].unique())
    lr = 0.001
    n_epochs = 50
    batch_size = 128


    model = MLP(input_size, hidden_units, num_classes)
    criterion = nn.CrossEntropyLoss()

    #criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    



    model.to(device)
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

    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), torch.LongTensor(y_test).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, labels in test_loader:
            
            #if the input row has a lot of zeros, it is not considered

            if ((inputs == 0).sum() / inputs.numel()) > 0.4:
                continue
            # Forward pass

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test data: %d %%' % (100 * correct / total))

    #print the confusion matrix
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.tolist())
    #print confusion matrix with index as labels
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(le.classes_)))))

    np.save('classes.npy', le.classes_)

    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()
