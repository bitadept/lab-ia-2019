# ======================================================
# --- LIBRARIES ----------------------------------------
# ======================================================
import os
import csv
from random import shuffle
import numpy as np
# pytorch
import torch
import torch.nn.functional as F
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
MODEL_SAVE_PATH = "S:\\models\\model.pt"
# data
NUM_CLASSES = 20
VALID_PCT = 0.20
# network hyperparams
LEARNING_RATE = 0.00062
BATCH_SIZE = 75
NUM_EPOCHS = 1000


# ======================================================
# --- CLASSES ------------------------------------------
# ======================================================
# Custom Dataset
class UserCoordsDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        input = self.inputs[idx].cuda()
        label = self.labels[idx].cuda()
        return input, label

    def __len__(self):
        return len(self.labels)


# Network
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 550)
        self.fc2 = nn.Linear(550, 550)
        # self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(550, NUM_CLASSES)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        # self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.0)

    def forward(self, x):
        # add hidden layer, with relu activation function
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        # x = self.dropout3(torch.relu(self.fc3(x)))
        x = self.dropout4(self.fc4(x))

        return x


# ======================================================
# --- FUNCTIONS ----------------------------------------
# ======================================================
def coords_450(reader):
    coords = [float(coord) for coords in reader for coord in coords]
    # add/remove coords if length is not valid
    length = len(coords)
    if length != 450:
        error = 450 - length
        # remove lines if negative, otherwise add extra lines with zeros
        if error > 0:
            coords.extend([0 for amt in range(error)])
        else:
            coords = coords[:error]
    # normalize coords
    coords = torch.tensor(coords, dtype=torch.float)
    coords.div_(coords.norm(p=2))

    return coords

# def coords_3(reader):
#     coords = [[] for i in range(3)]
#     for coords_row in reader:
#         for i, coord in enumerate(coords_row):
#             coords[i].append(float(coord))
#     coords = torch.tensor(coords, dtype=torch.float)
#     # get the mean of every coords column
#     coords = torch.mean(coords, dim=1)
#     coords.div_(coords.norm(p=2))
#     return coords

def load_inputs(path, testing=False):
    inputs = []

    # if in testing mode, create a indice -> name of file dict
    if testing:
        idx_to_name = {}

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), mode='r') as feature_file:
            reader = csv.reader(feature_file)
            # get coords
            coords = coords_450(reader)
        # associate idx to name if testing
        if testing:
            idx_to_name[len(inputs)] = filename.split('.')[0]  # without the extension
        inputs.append(coords)

    if testing:
        return inputs, idx_to_name
    return inputs


if __name__ == "__main__":
    # ======================================================
    # --- LOAD DATA ----------------------------------------
    # ======================================================
    # load the train inputs
    inputs = load_inputs(TRAIN_DATA_DIR)

    # load the train labels
    with open(TRAIN_LABELS_FILE, mode='r') as labels_file:
        csv_reader = csv.reader(labels_file)
        next(csv_reader)  # skip the header
        labels = torch.tensor([int(label) - 1 for _, label in csv_reader], dtype=torch.long)

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
    # --- TRAIN NETWORK ------------------------------------
    # ======================================================

    # init network
    model = Net(len(inputs[0])).cuda()

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # save the model when minimum loss is found
    validation_loss_min = np.Inf
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

        validation_loss = validation_loss / len(validation_loader)
        print(
            f"Epoch: {epoch + 1}/{NUM_EPOCHS}... "
            f"Training Loss: {train_loss / len(train_loader):.3f}... "
            f"Validation Loss: {validation_loss:.3f}... "
            f"Validation Accuracy: {accuracy / len(validation_loader):.3f}"
        )

        if validation_loss <= validation_loss_min:
            print(
                f"New minimum validation loss found: {validation_loss:.3f}. (Old: {validation_loss_min:.3f}). Saving model...")
            validation_loss_min = validation_loss
            # save model
            torch.save({
                'model_state_dict': model.state_dict(),
            }, MODEL_SAVE_PATH)

    print(f"--- Training completed. Minimum loss: {validation_loss_min:.3f} ---")
    # show model structure, parameters, loss function, etc.
    print("Learning Rate: ", LEARNING_RATE)
    print("Batch Size: ", BATCH_SIZE)
    print("Epochs: ", NUM_EPOCHS)
    print("Criterion: ", criterion)
    print("Optimizer: ", optimizer)
    print("Model Structure:\n", model)
