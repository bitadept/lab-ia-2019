# Conv1d implementation inspired from https://www.kaggle.com/purplejester/pytorch-deep-time-series-classification

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
MODEL_SAVE_PATH = "S:\\models\\model.pt"
# data
NUM_CLASSES = 20
VALID_PCT = 0.20
COLUMN_LEN = 150
ROW_LEN = 3
# network hyperparams
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0000
BATCH_SIZE = 64
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


# Network
class _SepConv1d(nn.Module):
    """
    Separable 1d convolution
    """

    def __init__(self, in_channels, out_channels, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel, stride, padding=pad, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepConv1d(nn.Module):
    """
    The adds optional activation function and dropout layers right after
    a separable convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel, stride, pad, drop=None, bn=True,
                 activation=lambda: nn.PReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(in_channels, out_channels, kernel, stride, pad)]
        if activation:
            layers.append(activation())
        if bn:
            layers.append(nn.BatchNorm1d(out_channels))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Flatten(nn.Module):
    """
    Flattens tensors
    """

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class Net(nn.Module):

    def __init__(self, in_channels, num_classes, drop=.5):
        super().__init__()

        self.out = nn.Sequential(

            SepConv1d(in_channels, 32, 8, 2, 3, drop=drop),
            SepConv1d(32, 64, 8, 4, 2, drop=drop),
            SepConv1d(64, 128, 8, 4, 2, drop=drop),
            SepConv1d(128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(drop),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs):
        return self.out(inputs)


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
    model = Net(ROW_LEN, NUM_CLASSES, drop=0.22).cuda()
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH)['model_state_dict'])

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.out.parameters(), lr=LEARNING_RATE)

    # show model structure, parameters, loss function, etc.
    print("Learning Rate: ", LEARNING_RATE)
    print("Weight Decay: ", WEIGHT_DECAY)
    print("Batch Size: ", BATCH_SIZE)
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
