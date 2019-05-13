# generate a csv from the testing samples
import os
import torch

from train import load_inputs
from train import Net

WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIR, "data")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
MODELS_PATH = "S:\\models"
MODEL_FILE = "model515_tloss209.pt"

inputs, idx_to_name = load_inputs(TEST_DATA_DIR, testing=True)

model = Net(len(inputs[0])).cuda()

checkpoint = torch.load(os.path.join(MODELS_PATH, MODEL_FILE))

model.load_state_dict(checkpoint['model_state_dict'])

f_out = open(os.path.join(MODELS_PATH, MODEL_FILE.split('.')[0] + ".csv"), mode='w')
f_out.write("id,class\n")

model.eval()
with torch.no_grad():
    for idx, input in enumerate(inputs):

        input = input.cuda()

        prediction = model.forward(input)

        _, top_class = prediction.topk(1)

        f_out.write(f"{idx_to_name[idx]},{top_class.item() + 1}\n")

f_out.close()