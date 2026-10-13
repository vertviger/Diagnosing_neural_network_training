import csv 
import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

random.seed(42)
torch.manual_seed(42)
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

# batches setup

train_dataset = TensorDataset(V_train_t, l_train_t)
batch_size = 30
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Multi Layer Perceptron
class BaselineMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_P):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_P)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = BaselineMLP(input_dim=4, hidden_dim=16, output_dim=3, dropout_P=0.10)

# Loss function
criterion = nn.CrossEntropyLoss()
# Gradient for parameter updates
optimizer = optim.AdamW(model.parameters(), lr = 0.05, weight_decay=1e-3)

# Training setup
epochs = 200
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training
for epoch in range(epochs):
    model.train()
    batch_loss = 0.0
    correct_batch_train = 0
    total_train = 0

    for batch_features, batch_labels in train_loader:
        train_logits = model(batch_features)
        loss = criterion(train_logits, batch_labels)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accuracy
        batch_loss += loss.item() * batch_features.size(0)
        _, training_prediction = torch.max(train_logits, dim=1)
        correct_batch_train += (training_prediction == batch_labels).sum().item()
        total_train += batch_labels.size(0)
    # epoch training metrics
    epoch_training_loss = batch_loss/total_train
    training_accuracy = correct_batch_train/total_train

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
    if ((epoch + 1) % 50 == 0) or epoch + 1 == epochs:
        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
            f"Train Acc: {training_accuracy*100:.1f}% | Val Acc: {val_accuracy*100:.1f}%"
        )

# testing

model.eval()
with torch.no_grad():
    test_logits = model(V_test_t)
    test_loss = criterion(test_logits, l_test_t)
    _, test_predictions = torch.max(test_logits, dim = 1)
    test_accuracy = (test_predictions == l_test_t).float().mean().item()

print(f"Test Acc: {test_accuracy*100:.1f}%")

# plotting the accuracy and loss metrics

plt.style.use('dark_background')

fig, axes = plt.subplots(2, 2, figsize = (12, 12))
ax1, ax2, ax3, ax4 = axes.flatten()

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

feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

val_preds_np = val_predictions.numpy()
true_val_labels = np.array(l_val)
X_val = np.array(V_val)

def plot_dual_layer_points(ax, feat_x, feat_y, title):
  ax.scatter(
      X_val[:, feat_x],
      X_val[:, feat_y],
      c=val_preds_np,
      s=140,
      alpha=0.4,
  )

  ax.scatter(
      X_val[:, feat_x],
      X_val[:, feat_y],
      c=true_val_labels,
      s=50,
  )

  ax.set_title(title)
  ax.set_xlabel(f"Feature {feat_x} ({feature_names[feat_x]})")
  ax.set_ylabel(f"Feature {feat_y} ({feature_names[feat_y]})")
  ax.grid(True)


# Validation Set - Sepal Dimensions (Features 0 & 1)
plot_dual_layer_points(
    ax3, 0, 1, "Validation set plot by features 0 and 1"
)

# Validation Set - Petal Dimensions (Features 2 & 3)
plot_dual_layer_points(
    ax4, 2, 3, "Validation set plot by features 2 and 3"
)

plt.tight_layout()
plt.show()