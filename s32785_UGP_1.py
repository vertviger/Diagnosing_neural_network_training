import csv 
import math
import random
import torch

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
    train_data = samples[:n_train]
    val_data = samples[n_train:n_train+n_val]
    test_data = samples[n_train+n_val :]
# check
print(
    f"Train size: {len(train_data)} | Val size: {len(val_data)} | Test size: {len(test_data)}"
)
# normalizing training set
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
    stand_d = math.sqrt(variance)

    means.append(mean)
    stand_d.append(stand_d)