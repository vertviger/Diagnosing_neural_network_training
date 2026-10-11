import csv 
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

random.seed(42)
# reading data from the file
data = []
with open('dataset.csv', mode = 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        if row:
            values = [float(val) for val in row[:-1]]
            label = row[-1]
            data.append((values, label))
# converting string labels to integers
unique_labels = []
for item in data:
    label_str = item[1]
    if label_str not in unique_labels:
        unique_labels.append(label_str)

unique_labels.sort()

label_to_id = {}
current_id = 0
for label_str in unique_labels:
    label_to_id[label_str] = current_id
    current_id = current_id + 1

processed_data = []

for item in data:
    label_str = item[1]
    features = item[0]
    label_int = label_to_id[label_str]
    processed_data.append((features, label_int))
# splitting into training, validation and test sets
classes = {}
for features, label_int in processed_data:
    if label_int not in classes: 
        classes[label_int] = []
    classes[label_int].append((features, label_int))

train_data = []
val_data = []
test_data = []

for label_int, samples in classes.items():
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n*0.6)
    n_val = int(n*0.2)
    train_data.extend(samples[:n_train])
    val_data.extend(samples[n_train:n_train+n_val])
    test_data.extend(samples[n_train+n_val :])
# check
print(
    f"Train size: {len(train_data)} | Val size: {len(val_data)} | Test size: {len(test_data)}"
)
# normalizing
features_count = len(train_data[0][0])

means = []
stand_d = []

for feature_idx in range(features_count):
    column_values = [sample[0][feature_idx] for sample in train_data]
    
    mean = sum(column_values)/len(column_values)
    
    squared_diffs = []
    for x in column_values:
        diff = (x - mean) ** 2
        squared_diffs.append(diff)
    
    variance = sum(squared_diffs)/len(column_values)
    stand_d_val = math.sqrt(variance)

    means.append(mean)
    stand_d.append(stand_d_val)

def standardize_dataset(dataset, means, stand_d):
    standardized_features = []
    labels = []
    for features, label in dataset:
        temp_std_feat = []
        for i in range(len(features)):
            temp_std_feat.append((features[i]-means[i])/stand_d[i])
        standardized_features.append(temp_std_feat)
        labels.append(label)
    return standardized_features, labels

# normilizing each set
V_train, l_train = standardize_dataset(train_data, means, stand_d)
V_val, l_val = standardize_dataset(val_data, means, stand_d) 
V_test, l_test = standardize_dataset(test_data, means, stand_d) 

# pytorch
V_train_t = torch.tensor(V_train, dtype = torch.float32)
l_train_t = torch.tensor(l_train, dtype = torch.long)

V_val_t = torch.tensor(V_val, dtype = torch.float32)
l_val_t = torch.tensor(l_val, dtype = torch.long)

V_test_t = torch.tensor(V_test, dtype = torch.float32)
l_test_t = torch.tensor(l_test, dtype = torch.long)

# Multi Layer Perceptron
class BaselineMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = BaselineMLP(input_dim=4, hidden_dim=8, output_dim=3)

# Loss function
criterion = nn.CrossEntropyLoss()
# Gradient for parameter updates
optimizer = optim.SGD(model.parameters(), lr = 0.05)

# Training setup
epochs = 150
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training
for epoch in range(epochs):
    model.train()
    train_logits = model(V_train_t)
    loss = criterion(train_logits, l_train_t)
    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # accuracy
    _, training_prediction = torch.max(train_logits, dim=1)
    training_accuracy = (training_prediction == l_train_t).float().mean().item()
    # evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(V_val_t)
        val_loss = criterion(val_logits, l_val_t)
        _, val_predictions = torch.max(val_logits, dim = 1)
        val_accuracy = (val_predictions == l_val_t).float().mean().item()
    # saving history of metrics
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accuracies.append(training_accuracy)
    val_accuracies.append(val_accuracy)

    # debug
    if (epoch + 1) % 30 == 0:
        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
            f"Train Acc: {training_accuracy*100:.1f}% | Val Acc: {val_accuracy*100:.1f}%"
        )

# plotting the accuracy and loss metrics

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

ax1.plot(train_losses, label = "Train Loss", color = "blue")
ax1.plot(val_losses, label = "Validation Loss", color = "green")
ax1.set_title("Loss changes through epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot([acc * 100 for acc in train_accuracies], label = "Train Accuracy", color = "blue")
ax2.plot([acc * 100 for acc in val_accuracies], label = "Validation Accuracy", color = "green")
ax2.set_title("Accuracy changes through epochs")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy %")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()