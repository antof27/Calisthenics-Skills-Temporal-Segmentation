import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

#how to train the model in cuda

torch.random.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Current device:", device)


le = LabelEncoder()
data = pd.read_csv('dataset_elaboratedv4K.csv')

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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super(MLP, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_units)
        self.activation1 = nn.LeakyReLU()
        
        self.hl2 = nn.Linear(hidden_units, hidden_units)
        self.activation2 = nn.LeakyReLU()
        
        self.hl3 = nn.Linear(hidden_units, hidden_units)
        self.activation3 = nn.LeakyReLU()
        
        self.hl4 = nn.Linear(hidden_units, num_classes)
        self.output_layer = nn.LogSoftmax(dim=1)

    
    def forward(self,x):
        hidden_representation = self.hl1(x)
        hidden_representation = self.activation1(hidden_representation)
        
        hidden_representation = self.hl2(hidden_representation)
        hidden_representation = self.activation2(hidden_representation)

        hidden_representation = self.hl3(hidden_representation)
        hidden_representation = self.activation3(hidden_representation)

        hidden_representation = self.hl4(hidden_representation)
        scores = self.output_layer(hidden_representation)
        return scores



# Set the hyperparameters
input_size = len(data.columns) - 3 # exclude 'id_video', 'frame', 'skill_id'
hidden_units = 1024
num_classes = len(data['skill_id'].unique())
lr = 0.0001
n_epochs = 80
batch_size = 128

# Initialize the model, loss function, and optimizer
model = MLP(input_size, hidden_units, num_classes)
criterion = nn.NLLLoss()
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
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print statistics
        running_loss += loss.item()
        if i % batch_size == batch_size - 1:
            acc = 100 * correct / total
            print('[Epoch %d, Batch %d] Loss: %.3f Accuracy: %.3f%%' % 
                  (epoch + 1, i + 1, running_loss / batch_size, acc))
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

print(confusion_matrix(y_true, y_pred))

np.save('classes.npy', le.classes_)

torch.save(model.state_dict(), 'model.pth')
