# ======================================================
# --- LIBRARIES ----------------------------------------
# ======================================================
import os
import csv
from random import shuffle
import numpy as np
# pytorch
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# ======================================================
# --- HELPER VARS --------------------------------------
# ======================================================
# devices
CUDA_0 = torch.device('cuda:0')
# paths
WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIR, "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, "train_labels.csv")
# data
NUM_CLASSES = 20
FEATURES_PER_INPUT = 450
VALID_PCT = 0.20
# network hyperparams
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 250

# ======================================================
# --- CLASSES ------------------------------------------
# ======================================================
# Custom Dataset
class UserCoordsDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        input = torch.from_numpy(self.inputs[idx]).cuda()
        label = torch.tensor(self.labels[idx], dtype=torch.long, device=CUDA_0)
        return input, label

    def __len__(self):
        return len(self.labels)

# ======================================================
# --- LOAD DATA ----------------------------------------
# ======================================================

# inputs loader function:
def load_inputs(path):
    inputs = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), mode='r') as feature_file:
            csv_reader = csv.reader(feature_file)
            coords = [float(coord) for coords in csv_reader for coord in coords]
        # add/remove coords if length is not valid
        length = len(coords)
        if length != FEATURES_PER_INPUT:
            error = FEATURES_PER_INPUT - length
            # remove lines if negative, otherwise add extra lines with zeros
            if error > 0:
                coords.extend([0 for amt in range(error)])
            else:
                coords = coords[:error]
        # normalize coords
        coords = np.array(coords, dtype=np.float32)
        coords = np.divide(coords, np.linalg.norm(coords, ord=2))
        inputs.append(coords)

    return inputs

# load the train inputs
inputs = load_inputs(TRAIN_DATA_DIR)

# load the train labels
with open(TRAIN_LABELS_FILE, mode='r') as labels_file:
    csv_reader = csv.reader(labels_file)
    next(csv_reader)  # skip the header
    labels = [int(label) - 1 for _, label in csv_reader]

# put the loaded data into a dataset
train_data = UserCoordsDataset(inputs, labels)

# split the data between train and validation
num_train = len(inputs)
indices = list(range(num_train))
shuffle(indices)
split = int(VALID_PCT * num_train)
train_indices, valid_indices = indices[split:], indices[:split]

# show lengths
print("Train Data Len: ", len(train_data))
print("Train Len: ", len(train_indices))
print("Validation Len: ", len(valid_indices))

# put the data into a torch DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
validation_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(valid_indices))
# ======================================================
# --- NETWORK ------------------------------------------
# ======================================================
model = torch.nn.Sequential(
    nn.Linear(FEATURES_PER_INPUT, 800),
    nn.Tanh(),
    nn.Dropout(0.3),
    nn.Linear(800, 800),
    nn.Tanh(),
    nn.Dropout(0.3),
    nn.Linear(800, NUM_CLASSES)
)
model.cuda()

# ======================================================
# --- TRAIN NETWORK ------------------------------------
# ======================================================

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    # train
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        # move data to the cuda device
        inputs = inputs.cuda()
        labels = labels.cuda()
        # reset gradients
        optimizer.zero_grad()
        # forward prop
        predictions = model.forward(inputs)
        # calculate loss
        loss = criterion(predictions, labels)
        # perform backprop
        loss.backward()
        # update weights
        optimizer.step()

        train_loss += loss.item()

    # validation and accuracy test
    model.eval()
    validation_loss = 0
    accuracy = 0
    # stop calculating the gradients
    with torch.no_grad():
        for inputs, labels in validation_loader:

            inputs = inputs.cuda()
            labels = labels.cuda()

            predictions = model.forward(inputs)
            loss = criterion(predictions, labels)

            validation_loss += loss.item()

            # get the classes with the highest probability
            _, top_classes = predictions.topk(1, dim=1)
            # calculate accuracy for this batch
            equality = top_classes == labels.view(*top_classes.shape)
            accuracy += torch.mean(equality.type(torch.float))

    print(
        f"Epoch: {epoch + 1}/{NUM_EPOCHS}... "
        f"Training Loss: {train_loss / len(train_loader):.3f}... "
        f"Validation Loss: {validation_loss / len(validation_loader):.3f}... "
        f"Validation Accuracy: {accuracy / len(validation_loader):.3f}"
    )

# show model structure, parameters, loss function, etc.
print("Learning Rate: ", LEARNING_RATE)
print("Batch Size: ", BATCH_SIZE)
print("Epochs: ", NUM_EPOCHS)
print("Criterion: ", criterion)
print("Optimizer: ", optimizer)
print("Model Structure:\n", model)