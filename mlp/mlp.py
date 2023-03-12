import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
data = pd.read_csv('dataset_elaborated.csv')

X = data.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
y = data['skill_id']

# Encode the labels
y = le.fit_transform(y)

# Define the GroupShuffleSplit object with the desired parameters
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split the data into train and test sets, using the 'skill_id' as labels and the 'video_name' as groups
train_idx, test_idx = next(gss.split(X, y, groups=data['video_name']))

X_train, y_train = X.iloc[train_idx], y[train_idx]
X_test, y_test = X.iloc[test_idx], y[test_idx]

X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hl2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.hl1(x)
        out = self.relu(out)
        out = self.hl2(out)
        out = self.softmax(out)
        return out

# Set the hyperparameters
input_size = len(data.columns) - 3 # exclude 'id_video' and 'class' columns
hidden_size = 128
num_classes = len(data['skill_id'].unique())
lr = 0.01
n_epochs = 50
batch_size = 32

# Initialize the model, loss function, and optimizer
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Convert inputs and labels to PyTorch tensors
        outputs = model(inputs)
        loss = criterion(outputs, labels)


        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % batch_size == batch_size - 1:
            print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_size))
            running_loss = 0.0

# Test the model

correct = 0
total = 0

test_dataset = TensorDataset(torch.FloatTensor(X_test.values), torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for inputs, labels in test_loader:

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test data: %d %%' % (100 * correct / total))


torch.save(model.state_dict(), 'model.pth')
