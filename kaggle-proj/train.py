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
COLUMN_LEN = 150
ROW_LEN = 3
# network hyperparams
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0000
BATCH_SIZE = 25
NUM_EPOCHS = 10000


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


# Networks
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_final = nn.Linear(512, NUM_CLASSES)

        self.activation = F.relu
        self.dropout1 = nn.Dropout(p=0.8)
        self.dropout2 = nn.Dropout(p=0.9)

    def forward(self, x):
        x = self.dropout1(self.activation(self.fc1(x)))
        x = self.dropout2(self.activation(self.fc2(x)))
        x = self.fc_final(x)

        return x


class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4352, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_final = nn.Linear(64, NUM_CLASSES)

        self.activation = F.relu
        self.dropout = nn.Dropout(p=0.12)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.pool(self.activation(self.conv2(x))))
        x = self.dropout(self.activation(self.conv3(x)))
        x = self.dropout(self.pool(self.activation(self.conv4(x))))
        x = self.dropout(self.activation(self.conv5(x)))
        x = self.dropout(self.pool(self.activation(self.conv6(x))))

        # print(x.shape)
        x = x.view(x.shape[0], 4352)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.fc_final(x))

        return x


# ======================================================
# --- FUNCTIONS ----------------------------------------
# ======================================================
# normalizations functions
def norm_standardize(tensor_1d):
    return tensor_1d.sub(tensor_1d.mean()).div(tensor_1d.std())


def norm_l2(tensor_1d):
    return tensor_1d.div(tensor_1d.norm(p=2))


def get_coords_from_file(file):
    reader = csv.reader(file)

    # store coords by column
    coord_cols = [[] for _ in range(3)]
    for coord_row in reader:
        for i, coord in enumerate(coord_row):
            coord_cols[i].append(float(coord))

    # add/remove coords if column length is not 150
    col_len = len(coord_cols[0])
    error_len = COLUMN_LEN - col_len
    # add elements if positive, otherwise remove
    for i, coord_col in enumerate(coord_cols):
        if error_len > 0:
            # calculate mean of this column
            mean = sum(coord_col) / len(coord_col)
            # append the calculated mean for error_len times
            coord_col.extend([mean for i in range(error_len)])
        elif error_len < 0:
            del coord_col[(col_len + error_len):]
        # convert to torch tensor & normalize
        coord_cols[i] = norm_standardize(torch.tensor(coord_col, dtype=torch.float, device=CUDA_0))
        # coord_cols[i] = torch.tensor(coord_col, dtype=torch.float, device=CUDA_0)

    # convert cols to tensor, concat and return it
    return torch.cat(coord_cols).view(ROW_LEN, COLUMN_LEN)


def load_inputs(path, testing=False):
    inputs = []

    # if in testing mode, create a indice -> name of file dict
    if testing:
        idx_to_name = {}

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), mode='r') as feature_file:
            # get coords
            coords = get_coords_from_file(feature_file)
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
        # substract 1 because classes start from 0
        labels = torch.tensor([int(label) - 1 for _, label in csv_reader], dtype=torch.long, device=CUDA_0)

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
    # model = Net(len(inputs[0])).cuda()
    model = NetCNN().cuda()
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH)['model_state_dict'])

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # show model structure, parameters, loss function, etc.
    print("Learning Rate: ", LEARNING_RATE)
    print("Weight Decay: ", WEIGHT_DECAY)
    print("Batch Size: ", BATCH_SIZE)
    print("Activation: ", model.activation)
    print("Epochs: ", NUM_EPOCHS)
    print("Criterion: ", criterion)
    print("Optimizer: ", optimizer)
    print("Model Structure:\n", model)

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
            # perform backprop, compute gradients
            loss.backward()
            # update weights by applying gradients
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
